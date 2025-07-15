import ot
import time
import logging
import argparse
import numpy as np
from pathlib import Path

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

def sims2rank(sim_matrix):
    toc2 = time.time()
    logger.info("videos:{:d}, captions:{:d}.".format(*sim_matrix.shape))
    (r1, r5, r10, r50, medr, meanr) = i2t(npts=sim_matrix.shape[0], sims=sim_matrix)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2i(npts=sim_matrix.shape[0], sims=sim_matrix)
    toc3 = time.time()
    logger.info("time profile: metrics {:.5f}s".format(toc3 - toc2))
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))
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
    P, logs = ot.sinkhorn(r, c, 1-sims, reg=tau, log=True, numItermax=10)
    v_hubness = -tau * np.log(logs["v"])
    t_hubness = -tau * np.log(logs["u"])
    return v_hubness, t_hubness

def main_sims():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--dataset', default='f30k', help='coco, f30k')
    parser.add_argument('--sims_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--qtsims_path', type=Path, help='path to checkpoint for evaluation')
    parser.add_argument('--qqsims_path', type=Path, help='path to checkpoint for evaluation')
    args = parser.parse_args()

    try:sims = np.load(args.sims_path, allow_pickle=True).tolist()["sims"] 
    except:sims = np.load(args.sims_path, allow_pickle=True)

    try:qtsims = np.load(args.qtsims_path, allow_pickle=True).tolist()["sims"]
    except:qtsims = np.load(args.qtsims_path, allow_pickle=True)

    logger.info("\n%s"%args.sims_path)
    logger.info("Baseline: ")
    sims2rank(sims)

    logger.info("Inverted_Softmax")
    v_hubness, t_hubness = Inverted_Softmax(qtsims, v_tau=0.01, t_tau=0.01)
    sims2rank(sims-t_hubness)
    # sims2rank(sims-t_hubness, args.dataset)

    logger.info("Dual Bank Inverted_Softmax")
    v_hubness, t_hubness = Inverted_Softmax(qtsims, v_tau=0.01, t_tau=0.01)
    sims2rank(sims-t_hubness)
    # sims2rank(sims-t_hubness, args.dataset)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(qtsims, tau=0.01)
    sims2rank(sims-t_hubness[:,np.newaxis])
    # sims2rank(sims-t_hubness[:,np.newaxis], args.dataset)

    # here qt 
    try:qqsims = np.load(args.qqsims_path, allow_pickle=True).tolist()["sims"]
    except:qqsims = np.load(args.qqsims_path, allow_pickle=True)
    logger.info("Dual-Bank Sinkhorn_Normalization")
    qbsims = np.concatenate([qtsims, qqsims], axis=0)
    v_hubness, t_hubness = Sinkhorn_Normalization(qbsims, tau=0.01)
    sims2rank(sims-t_hubness[:qtsims.shape[0],np.newaxis])
    # sims2rank(sims-t_hubness[:,np.newaxis], args.dataset)
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
    if q.shape[0] == t.shape[0]:t=t[::5]
    gq = np.load(args.gqemb_path, allow_pickle=True)
    gt = np.load(args.gtemb_path, allow_pickle=True)
    if gq.shape[0] == gt.shape[0]:gt=gt[::5]
    sims = t@q.T
    qtsims = t@gq.T
    # qqsims = gt@gq.T
    ttsims = t@gt.T

    logger.info("\n%s"%args.qemb_path)
    logger.info("Baseline: ")
    sims2rank(sims)

    logger.info("Inverted_Softmax")
    tau_gq = 0.02
    v_hubness, t_hubness = Inverted_Softmax(qtsims, v_tau=tau_gq, t_tau=tau_gq)
    sims2rank(sims-t_hubness)
    # sims2rank(sims-t_hubness, args.dataset)


    logger.info("DualIS")
    tau_gg = 0.02
    v_hubness, tt_hubness = Inverted_Softmax(ttsims, v_tau=tau_gg, t_tau=tau_gg)
    beta = 0.01
    dbsims = sims-t_hubness-beta*tt_hubness
    sims2rank(dbsims)
    # sims2rank(sims-t_hubness, args.dataset)

    logger.info("DualDIS")
    dual_sims = db_norm(qtsims.T.copy(),ttsims.T.copy(),sims.T.copy(),
        k=1,
        beta1=0.02,
        beta2=1,
        dynamic_normalized=False
    )
    sims2rank(dual_sims.T)

    logger.info("QB norm")
    sims2rank(qb_norm(qtsims.T.copy(), sims.T.copy(), k=1, tau=0.02).T)

    logger.info("Sinkhorn_Normalization")
    v_hubness, t_hubness = Sinkhorn_Normalization(qtsims, tau=0.01)
    sims2rank(sims-t_hubness[:,np.newaxis])
    # sims2rank(sims-t_hubness[:,np.newaxis], args.dataset)
    
    # here qt )
    qqsims = gt@gq.T
    logger.info("Dual-Bank Sinkhorn_Normalization")
    qbsims = np.concatenate([qtsims, qqsims], axis=0)
    v_hubness, t_hubness = Sinkhorn_Normalization(qbsims, tau=0.01)
    sims2rank(sims-t_hubness[:qtsims.shape[0],np.newaxis])
    # sims2rank(sims-t_hubness[:,np.newaxis], args.dataset)
    logger.info("\n")

    return

if __name__=="__main__":
    # main_sims()
    main_embs()
