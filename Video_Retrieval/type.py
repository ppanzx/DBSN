import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def v2t(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    mode = npts != sims.shape[1]
    tov = 20 ## t/v


    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode:
            rank = 1e20
            for i in range(tov * index, tov * index + tov, 1):
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

    # if return_ranks:
    #     return r1, r5, r10, r50, medr, meanr, ranks, top1
    # else:
    #     return r1, r5, r10, r50, medr, meanr
    metrics = {}
    metrics['R1'] = r1
    metrics['R5'] = r5
    metrics['R10'] = r10
    metrics['MR'] = meanr
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = medr
    # metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def t2v(sims, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    mode = npts != sims.shape[1]
    tov = 20 ## t/v

    if mode:
        ranks = np.zeros(tov * npts)
        top1 = np.zeros(tov * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode:
            for i in range(tov):
                inds = np.argsort(sims[tov * index + i])[::-1]
                ranks[tov * index + i] = np.where(inds == index)[0][0]
                top1[tov * index + i] = inds[0]
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

    # if return_ranks:
    #     return r1, r5, r10, r50, medr, meanr, ranks, top1
    # else:
    #     return r1, r5, r10, r50, medr, meanr
    metrics = {}
    metrics['R1'] = r1
    metrics['R5'] = r5
    metrics['R10'] = r10
    metrics['MR'] = meanr
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = medr
    # metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def sims2rank(t2v_sims, v2t_sims=None):
    toc2 = time.time()
    logger.info("videos:{:d}, captions:{:d}.".format(*t2v_sims.shape))
    tv_metrics = t2v(sims=t2v_sims)
    
    toc3 = time.time()
    
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Mean R: {:.1f} - Median R: {:.1f} ".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MeanR'], tv_metrics['MR']))
    if v2t_sims is not None:
        vt_metrics = v2t(sims=v2t_sims)
        logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Mean R: {:.1f} - Median R: {:.1f}".format(
            vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MeanR'], vt_metrics['MR']))
    
    logger.info("time profile: metrics {:.5f}s".format(toc3 - toc2))
    return


def Inverted_Softmax(sims, v_tau, t_tau):
    t_hubness = v_tau*np.log(np.exp(sims/v_tau).sum(axis=0,keepdims=True))
    v_hubness = t_tau*np.log(np.exp(sims/t_tau).sum(axis=1,keepdims=True))
    return t_hubness,v_hubness

# def Sinkhorn_Normalization(sims, tau=0.01):
#     r = np.ones(sims.shape[0])/sims.shape[0]
#     c = np.ones(sims.shape[1])/sims.shape[1]
#     P = ot.sinkhorn(r,c,1-sims,reg=tau)
#     return P

def Sinkhorn_Normalization(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True, numItermax=10) #, numItermax=10
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

def Hungarian_Normalization(sims):
    row_ind, col_ind = linear_sum_assignment(1-sims)
    hm_sims = np.zeros_like(sims)
    hm_sims[row_ind, col_ind] = 1
    return hm_sims

def L2_Normalization(sims, tau=0.05):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    l2_sims = ot.smooth.smooth_ot_dual(r, c, 1-sims, tau, reg_type='l2')
    return l2_sims

def Sparse_Normalization(sims, tau=0.1, max_nz=1):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    sn_sims = ot.smooth.smooth_ot_dual(r, c, 1-sims, tau, 
                                 reg_type='sparsity_constrained', max_nz=max_nz)
    return sn_sims

def main():
    # parser = argparse.ArgumentParser(description='PyTorch Template')
    # parser.add_argument('--sims_path', type=Path, help='path to checkpoint for evaluation')
    # args = parser.parse_args()

    class args:
        q_t_sims_path = 'runs/sims/clip4clip/msrvtt-meanP/clip4clip_sims.npy' ## test_imgs @ test_txts.T
        gq_t_sims_path = "./sims/f30k_train_txts_test_imgs_sims.npy" ## test_imgs @ train_txts.T
        gq_q_sims_path = "./sims/f30k_train_txts_test_txts_sims.npy" ## train_txts @ test_txts.T
        gq_gt_sims_path = "./sims/f30k_train_txts_train_imgs_sims.npy" ## train_imgs @ train_txts.T

    try:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True).tolist()["sims"] 
    except:q_t_sims = np.load(args.q_t_sims_path, allow_pickle=True)
    logger.info("load q_t_sims.")

    # try:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True).tolist()["sims"]
    # except:gq_t_sims = np.load(args.gq_t_sims_path, allow_pickle=True)
    # logger.info("load gq_t_sims.")

    # all sims -> n_vid x n_cap
    q_t_sims = q_t_sims.T 
    
    logger.info("Baseline: ")
    sims2rank(q_t_sims)

    ## others : t_tau = 0.02 for best
    # logger.info("Inverted_Softmax")
    # t_hubness, v_hubness = Inverted_Softmax(q_t_sims, t_tau=0.02, v_tau=0.02)
    # sims2rank(q_t_sims-v_hubness, q_t_sims-t_hubness)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(q_t_sims, tau=0.01)
    sims2rank(q_t_sims-t_hubness[:,np.newaxis])

    logger.info("Hungarian_Normalization")
    hn_sims = Hungarian_Normalization(q_t_sims)
    sims2rank(hn_sims)

    logger.info("L2_Normalization: k=1")
    l2_sims = L2_Normalization(q_t_sims)
    sims2rank(l2_sims)

    logger.info("Sparse_Normalization: k=1")
    sn_sims = Sparse_Normalization(q_t_sims, max_nz=1)
    sims2rank(sn_sims)

    logger.info("Sparse_Normalization: k=5")
    sn_sims = Sparse_Normalization(q_t_sims, max_nz=5)
    sims2rank(sn_sims)

    logger.info("Sparse_Normalization: k=10")
    sn_sims = Sparse_Normalization(q_t_sims, max_nz=10)
    sims2rank(sn_sims)

    logger.info("\n")

if __name__=="__main__":
    main()
