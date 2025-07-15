import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path

import torch
from scipy.stats import skew
from collections import Counter

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def i2t(npts, sims, return_ranks=False, mode="coco"):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr

def t2i(npts, sims, return_ranks=False, mode="coco"):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr

def sims2rank(t2i_sims, i2t_sims=None):
    toc2 = time.time()
    logger.info("videos:{:d}, captions:{:d}.".format(*t2i_sims.shape))
    (r1a, r5a, r10a, r50a, medra, meanra) = t2i(npts=t2i_sims.shape[0], sims=t2i_sims)
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))
    if i2t_sims is not None:
        (r1, r5, r10, r50, medr, meanr) = i2t(npts=i2t_sims.shape[0], sims=i2t_sims)
        logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                    (r1, r5, r10, r50, medr, meanr))
    toc3 = time.time()
    logger.info("time profile: metrics {:.5f}s".format(toc3 - toc2))
    return


def Inverted_Softmax(sims, v_tau, t_tau):
    v_hubness = v_tau*np.log(np.exp(sims/v_tau).sum(axis=0,keepdims=True))
    t_hubness = t_tau*np.log(np.exp(sims/t_tau).sum(axis=1,keepdims=True))
    return v_hubness, t_hubness

def Sinkhorn_Normalization(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True)
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

def otnorm(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[0]
    P = ot.sinkhorn(r, c, 1-sims, reg=tau)
    return P

def qb_norm(train_test, test_test, k=1, tau=0.05):
    def get_retrieved_videos(sims, k):
        argm = np.argsort(-sims, axis=1)
        topk = argm[:,:k].reshape(-1)
        retrieved_videos = np.unique(topk)
        return retrieved_videos

    # Returns list of indices to normalize from sims based on videos
    def get_index_to_normalize(sims, videos):
        argm = np.argsort(-sims, axis=1)[:,0]
        result = np.array(list(map(lambda x: x in videos, argm)))
        result = np.nonzero(result)
        return result

    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test/tau)
    test_test = np.exp(test_test/tau)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized

def Skewness(sims, topk=1):
    if isinstance(sims, np.ndarray):
        sims = torch.tensor(sims)
    _, indices = sims.topk(k=topk, dim=1)
    p = torch.zeros_like(sims)
    row_indices = torch.arange(sims.size(0)).unsqueeze(1)
    p[row_indices, indices] = 1
    Nk = p.sum(dim=0)
    Sk = skew(Nk)
    return Sk

def get_skewness(sim_matrix, topk=10):
    if type(sim_matrix) == torch.Tensor:
        retrieve_index = (-sim_matrix).argsort(axis=1)[:, :topk].reshape(-1).numpy().tolist()
    else:
        retrieve_index = (-sim_matrix).argsort(axis=1)[:, :topk].flatten().tolist()
    count = Counter(retrieve_index)
    l = [count[k] for k in count.keys()]
    return skew(l)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--dataset', default='f30k', help='coco, f30k')
    parser.add_argument('--sims_path', type=Path, help='path to checkpoint for evaluation',
                        default='./runs/sims/zs_f30k_sims.npy')
    args = parser.parse_args()

    try:
        sims = np.load(args.sims_path, allow_pickle=True).tolist()["sims"]
    except:
        sims = np.load(args.sims_path, allow_pickle=True)
        sims = sims[:,:sims.shape[0]*5]
    
    logger.info("\n%s"%args.sims_path)
    logger.info("Baseline: ")
    sims2rank(sims, sims)
    skewness = Skewness(sims.T,10)
    # skewness = get_skewness(sims,10)
    logger.info("Skewness:%.3f"%(skewness))
    
    logger.info("Inverted_Softmax")
    v_hubness, t_hubness = Inverted_Softmax(sims, v_tau=0.01, t_tau=0.02)
    sims2rank(sims-t_hubness, sims-v_hubness, )
    is_sims = sims-t_hubness
    skewness = Skewness(is_sims.T,10)
    # skewness = get_skewness(sims,10)
    logger.info("Skewness:%.3f"%(skewness))

    # logger.info("DIS")
    # sims2rank(qb_norm(sims.T.copy(), sims.T.copy(), k=1, tau=0.0195).T, qb_norm(sims.copy(), sims.copy(), k=1, tau=0.0095))

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(sims, tau=0.01)
    sims2rank(sims-t_hubness[:,np.newaxis], sims-v_hubness, )
    sn_sims = sims-t_hubness[:,np.newaxis]
    skewness = Skewness(sn_sims.T,10)
    # skewness = get_skewness(sims,10)
    logger.info("Skewness:%.3f"%(skewness))
    logger.info("\n")

if __name__=="__main__":
    main()
