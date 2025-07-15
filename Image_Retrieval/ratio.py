import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
from scipy.stats import skew

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

# def Sinkhorn_Normalization(sims, tau=0.01):
#     r = np.ones(sims.shape[0])/sims.shape[0]
#     c = np.ones(sims.shape[1])/sims.shape[1]
#     P = ot.sinkhorn(r,c,1-sims,reg=tau)
#     return P

def Sinkhorn_Normalization(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True, numItermax=10)
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

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

def emd(sims):
    a = np.ones(sims.shape[0])/sims.shape[0]
    b = np.ones(sims.shape[1])/sims.shape[1]
    return ot.emd2(a,b,1-sims)


def main():
    # parser = argparse.ArgumentParser(description='PyTorch Template')
    # parser.add_argument('--dataset', default='coco', help='coco, f30k')
    # parser.add_argument('--sims_path', default='runs/sims/zs_f30k_sims.npy', type=Path, help='path to checkpoint for evaluation')
    # args = parser.parse_args()

    class args:
        q_t_sims_path = './runs/sims/zs_f30k_sims.npy' ## test_imgs @ test_txts.T
        gq_t_sims_path = "./runs/sims/f30k_train_txts_test_imgs_sims.npy" ## test_imgs @ train_txts.T
        gq_q_sims_path = "./runs/sims/f30k_train_txts_test_txts_sims.npy" ## train_txts @ test_txts.T
        gq_gt_sims_path = "./runs/sims/f30k_train_txts_train_imgs_sims.npy" ## train_imgs @ train_txts.T

        q_embs_path = "runs/querybank/f30k_test_txts.npy"
        t_embs_path = "runs/querybank/f30k_test_imgs.npy"
        gq_embs_path = "runs/querybank/f30k_train_txts.npy"
        gt_embs_path = "runs/querybank/f30k_train_imgs.npy"


    #################### load embs ####################
    q_embs = np.load(args.q_embs_path, allow_pickle=True)
    logger.info("load q_embs.")
    t_embs = np.load(args.t_embs_path, allow_pickle=True)[::5]
    logger.info("load t_embs.")
    gq_embs = np.load(args.gq_embs_path, allow_pickle=True)
    logger.info("load gq_embs.")
    gt_embs = np.load(args.gt_embs_path, allow_pickle=True)[::5]
    logger.info("load gt_embs.")

    q_t_sims = t_embs @ q_embs.T
    gq_t_sims = t_embs @ gq_embs.T
    gq_gt_sims = gt_embs @ gq_embs.T
    #################### load sims ####################
    # try:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True).tolist()["sims"] 
    # except:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True)
    # logger.info("load q_t_sims.")

    # try:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True).tolist()["sims"]
    # except:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True)
    # logger.info("load gq_t_sims.")

    # try:gq_q_sims = np.load(args.gq_q_sims_path, allow_pickle=True).tolist()["sims"]
    # except:gq_q_sims = np.load(args.gq_q_sims_path, allow_pickle=True)
    # logger.info("load gq_q_sims.")

    # try:gq_gt_sims = np.load(args.gq_gt_sims_path, allow_pickle=True).tolist()["sims"]
    # except:gq_gt_sims = np.load(args.gq_gt_sims_path, allow_pickle=True)
    # logger.info("load gq_gt_sims.")

    # logger.info("\n%s"%args.q_t_sims_path)
    # logger.info("Baseline: ")
    # sims2rank(q_t_sims)
    # logger.info("Skewness:%.3f"%(Skewness(q_t_sims.T,10)))
    # # logger.info("mean gt similarity:%.4f"%emd(gq_t_sims[:,:nqb]))

    # logger.info("Sinkhorn_Normalization")
    # v_hubness, t_hubness = Sinkhorn_Normalization(q_t_sims, tau=0.01)
    # hr_sims = q_t_sims-t_hubness[:,np.newaxis]
    # sims2rank(hr_sims)
    # logger.info("Skewness:%.3f"%(Skewness(hr_sims.T,10)))
    # logger.info("mean gt similarity:%.4f"%emd(q_t_sims))
    # logger.info("mean gq similarity:%.4f"%0)

    rounts = 3
    for round in range(rounts):
        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<< round %d >>>>>>>>>>>>>>>>>>>>>>>>>>>"%round)
        for ratio in range(3,11,1):
            r = 0.5**ratio
            logger.info("ratio:%.5f"%r)
            nqb = int(r*gq_t_sims.shape[1])
            tb = gq_t_sims.shape[1]
            ids = np.random.choice(tb, nqb, replace=False)
            norm_sims = gq_t_sims[:,ids]

            logger.info("Inverted_Softmax")
            tau_gq = 0.02
            v_hubness, t_hubness = Inverted_Softmax(norm_sims, v_tau=tau_gq, t_tau=tau_gq)
            sn_sims = q_t_sims-t_hubness
            sims2rank(sn_sims)
            logger.info("Skewness:%.3f"%(Skewness(sn_sims.T,10)))
            # logger.info("mean gt similarity:%.4f"%emd(norm_sims))

            logger.info("Sinkhorn_Normalization")
            v_hubness, t_hubness = Sinkhorn_Normalization(norm_sims, tau=0.01)
            sn_sims = q_t_sims-t_hubness[:,np.newaxis]
            sims2rank(sn_sims)
            logger.info("Skewness:%.3f"%(Skewness(sn_sims.T,10)))
            logger.info("mean gt similarity:%.4f"%emd(norm_sims))
            # logger.info("mean gq similarity:%.4f"%emd(gq_q_sims[ids,:]))

            logger.info("Dual Bank Sinkhorn_Normalization")
            # v_hubness, t_hubness = Sinkhorn_Normalization(gq_t_sims[:,ids], tau=0.01)
            norm_sims = np.concatenate([gq_t_sims[:,ids], gq_gt_sims[:,ids]], axis=0)
            v_hubness, t_hubness = Sinkhorn_Normalization(norm_sims, tau=0.01)
            dbsn_sims = q_t_sims-t_hubness[:gq_t_sims.shape[0],np.newaxis]
            sims2rank(dbsn_sims)
            logger.info("Skewness:%.3f"%(Skewness(dbsn_sims.T,10)))
            logger.info("mean gt similarity:%.4f"%emd(norm_sims))
            # logger.info("mean gq similarity:%.4f"%emd(gq_q_sims[ids,:]))
            logger.info("\n")

if __name__=="__main__":
    main()
