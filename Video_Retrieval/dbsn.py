import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
    logger.info("time profile: metrics {:.5f}s".format(toc3 - toc2))
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Mean R: {:.1f} - Median R: {:.1f} ".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MeanR'], tv_metrics['MR']))
    if v2t_sims is not None:
        vt_metrics = v2t(sims=v2t_sims)
        logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Mean R: {:.1f} - Median R: {:.1f}".format(
            vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MeanR'], vt_metrics['MR']))
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


def db_norm(
    train_cap_test_vis,
    train_vis_test_vis,
    test_cap_test_vis,
    k=1,
    beta1=1.99,
    beta2=1/3,
    dynamic_normalized=False
):
    def IS(train_test, test_test, tau=20, k=1):
        test_test_normalized = test_test
        train_test = np.exp(train_test/tau)
        test_test = np.exp(test_test/tau)

        normalizing_sum = np.sum(train_test, axis=0)
        test_test_normalized = test_test / normalizing_sum
        return test_test_normalized
    
    func = qb_norm if dynamic_normalized else IS

    test_test_query_normalized = func(train_cap_test_vis, test_cap_test_vis, k=k, tau=beta1)
    test_test_gallery_normalized = func(train_vis_test_vis, test_cap_test_vis, k=k, tau=beta2)

    sim_matrix_normalized = test_test_query_normalized * test_test_gallery_normalized

    return sim_matrix_normalized

def Sinkhorn_Normalization(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True, numItermax=10) #, numItermax=10
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

def main_sims():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--dataset', default='f30k', help='coco, f30k')
    parser.add_argument('--sims_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--qtsims_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--gqgt_sims_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--gttsims_path', type=Path, help='path to checkpoint for evaluation')
    args = parser.parse_args()

    try:sims = np.load(args.sims_path, allow_pickle=True).tolist()["sims"] 
    except:sims = np.load(args.sims_path, allow_pickle=True)
    sims = sims.T

    try:qtsims = np.load(args.qtsims_path, allow_pickle=True).tolist()["sims"]
    except:qtsims = np.load(args.qtsims_path, allow_pickle=True)
    qtsims = qtsims.T

    logger.info("\n%s"%args.sims_path)
    logger.info("Baseline: ")
    sims2rank(sims)

    logger.info("Inverted_Softmax")
    v_hubness, t_hubness = Inverted_Softmax(qtsims, v_tau=0.01, t_tau=0.01)
    sims2rank(sims-t_hubness)

    logger.info("DIS")
    sims2rank(qb_norm(qtsims.T.copy(), sims.T.copy(), k=2, tau=0.01).T)

    logger.info("DualIS")
    try:gttsims = np.load(args.gttsims_path, allow_pickle=True).tolist()["sims"]
    except:gttsims = np.load(args.gttsims_path, allow_pickle=True)
    gttsims = gttsims.T
    dual_sims = db_norm(qtsims.T.copy(),gttsims.T.copy(),sims.T.copy(),
        k=1,
        beta1=0.01,
        beta2=1,
        dynamic_normalized=False
    )
    sims2rank(dual_sims.T)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(qtsims, tau=0.01)
    sims2rank(sims-t_hubness[:,np.newaxis])

    # here qt 
    try:gqgt_sims = np.load(args.gqgt_sims_path, allow_pickle=True).tolist()["sims"]
    except:gqgt_sims = np.load(args.gqgt_sims_path, allow_pickle=True)
    gqgt_sims = gqgt_sims.T
    logger.info("Dual-Bank Sinkhorn_Normalization")
    qbsims = np.concatenate([qtsims, gqgt_sims], axis=0)
    v_hubness, t_hubness = Sinkhorn_Normalization(qbsims, tau=0.01)
    sims2rank(sims-t_hubness[:qtsims.shape[0],np.newaxis])
    logger.info("\n")

def main_embs():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--dataset', default='f30k', help='coco, f30k')
    parser.add_argument('--qemb_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--temb_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--gqemb_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--gtemb_path', type=Path, help='path to checkpoint for evaluation')
    args = parser.parse_args()

    q = np.load(args.qemb_path, allow_pickle=True)
    t = np.load(args.temb_path, allow_pickle=True)
    gq = np.load(args.gqemb_path, allow_pickle=True)
    gt = np.load(args.gtemb_path, allow_pickle=True)
    sims = t@q.T
    qtsims = t@gq.T
    gqgt_sims = gt@gq.T
    ttsims = t@gt.T

    logger.info("\n%s"%args.qemb_path)
    logger.info("Baseline: ")
    sims2rank(sims)

    logger.info("Inverted_Softmax")
    tau_gq = 0.02
    v_hubness, t_hubness = Inverted_Softmax(qtsims, v_tau=tau_gq, t_tau=tau_gq)
    sims2rank(sims-t_hubness)

    logger.info("Dual Bank Inverted_Softmax")
    tau_gg = 0.015
    v_hubness, tt_hubness = Inverted_Softmax(ttsims, v_tau=tau_gg, t_tau=tau_gg)
    beta = 0.01
    dbsims = sims-t_hubness-beta*tt_hubness
    sims2rank(dbsims)

    logger.info("Dual Bank Inverted_Softmax v2")
    dual_sims = db_norm(qtsims.T.copy(),ttsims.T.copy(),sims.T.copy(),
        k=1,
        beta1=0.02,
        beta2=0.5,
        dynamic_normalized=False
    )
    sims2rank(dual_sims.T)

    logger.info("QB norm")
    sims2rank(qb_norm(qtsims.T.copy(), sims.T.copy(), k=2, tau=0.02).T)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(qtsims, tau=0.01)
    sims2rank(sims-t_hubness[:,np.newaxis])

    # here qt )
    logger.info("Dual-Bank Sinkhorn_Normalization")
    qbsims = np.concatenate([qtsims, gqgt_sims], axis=0)
    v_hubness, t_hubness = Sinkhorn_Normalization(qbsims, tau=0.01)
    sims2rank(sims-t_hubness[:qtsims.shape[0],np.newaxis])
    return

if __name__=="__main__":
    main_sims()
    # main_embs()
