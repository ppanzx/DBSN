import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path
import torch
from scipy.stats import skew
from scipy.optimize import linear_sum_assignment

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
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True)
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

def OT_Normalization(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    ot_sims = ot.emd(r, c, 1-sims)
    return ot_sims

def Hungarian_Normalization(sims):
    row_ind, col_ind = linear_sum_assignment(1-sims)
    hm_sims = np.zeros_like(sims)
    hm_sims[row_ind, col_ind] = 1
    return hm_sims

def Sparse_Normalization(sims, tau=0.05, max_nz=5):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    sn_sims = ot.smooth.smooth_ot_dual(r, c, 1-sims, tau, 
                                 reg_type='sparsity_constrained', max_nz=max_nz)
    return sn_sims


def L2_Normalization(sims, tau=0.05):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    l2_sims = ot.smooth.smooth_ot_dual(r, c, 1-sims, tau, reg_type='l2')
    return l2_sims


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

    try:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True).tolist()["sims"] 
    except:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True)
    logger.info("load q_t_sims.")

    try:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True).tolist()["sims"]
    except:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True)
    logger.info("load gq_t_sims.")

    sims2rank(q_t_sims)
    logger.info("Skewness:%.3f"%(Skewness(q_t_sims.T,10)))

    logger.info("Inverted_Softmax")
    v_hubness, t_hubness = Inverted_Softmax(q_t_sims, v_tau=0.02, t_tau=0.01)
    # sims2rank(q_t_sims-v_hubness)
    sims2rank(q_t_sims-t_hubness)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(q_t_sims, tau=0.01)
    # sims2rank(q_t_sims-v_hubness)
    sims2rank(q_t_sims-t_hubness[:,np.newaxis])

    logger.info("OT_Normalization")
    ot_sims = OT_Normalization(q_t_sims)
    sims2rank(ot_sims)
    sparsity = (ot_sims==0).sum()/np.size(ot_sims)
    logger.info("sparsity: %.8f"%sparsity)

    logger.info("Hungarian_Normalization")
    hn_sims = Hungarian_Normalization(q_t_sims)
    sims2rank(hn_sims)
    sparsity = (hn_sims==0).sum()/np.size(hn_sims)
    logger.info("sparsity: %.8f"%sparsity)

    logger.info("L2_Normalization")
    l2_sims = L2_Normalization(q_t_sims)
    sims2rank(l2_sims)
    sparsity = (l2_sims==0).sum()/np.size(l2_sims)
    logger.info("sparsity: %.8f"%sparsity)

    logger.info("Sparse_Normalization: k=1")
    sn1_sims = Sparse_Normalization(q_t_sims, max_nz=1)
    sims2rank(sn1_sims)
    sparsity = (sn1_sims==0).sum()/np.size(sn1_sims)
    logger.info("sparsity: %.8f"%sparsity)

    logger.info("Sparse_Normalization: k=5")
    sn5_sims = Sparse_Normalization(q_t_sims, max_nz=5)
    sims2rank(sn5_sims)
    sparsity = (sn5_sims==0).sum()/np.size(sn5_sims)
    logger.info("sparsity: %.8f"%sparsity)

    logger.info("\n")
    pass

if __name__=="__main__":
    main()
