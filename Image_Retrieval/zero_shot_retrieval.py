import os
import ot
global logger
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import logging
import sys
import time

import torch
from transformers import CLIPProcessor, CLIPModel

from dataloader import get_dataloader

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/panzx/dataset/CrossModalRetrieval',
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k',
                        help='coco, f30k')
    parser.add_argument('--split', default='test',
                        help='train, dev, test')
    parser.add_argument('--model', default='clip',
                        help='plip, clip')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--save_path', default=None, type=str,
                        help='Path to save the similarity results.')
    return parser

def setup_logger(name, save_dir, dist_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    # don't log results for the non-master process
    if dist_rank > 0:
        return logger
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(f"[{dist_rank}]"+"[%(asctime)s %(name)s %(lineno)s %(levelname)s]: %(message)s")
    formatter = logging.Formatter("[%(asctime)s %(name)s %(lineno)s %(levelname)s]: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


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
    
def otnorm(sims, tau=0.01):
    r = np.ones(sims.shape[0])*5
    c = np.ones(sims.shape[1])
    P = ot.sinkhorn(r,c,1-sims,reg=tau)
    return P

def isnorm(x, tau, axis):
    """Compute softmax values for each sets of scores in x."""
    ## 1: na\ive formulation
    return np.exp(x/tau) / np.sum(np.exp(x/tau), axis=axis, keepdims=True)

    ## 2: Avoiding overflow
    # e_max = np.max(x,axis=axis,keepdims=True)
    # e_x = np.exp((x - e_max)/tau)
    # return e_x / e_x.sum() * x.shape[1-axis]

    ## 3: torch.softmax
    # return torch.softmax(torch.tensor(x)/tau,dim=axis).numpy()

if __name__=="__main__":
    parser = get_argument_parser()
    config = parser.parse_args()

    if config.save_path:
        if not os.path.exists(config.save_path):os.makedirs(config.save_path)
        logger = setup_logger('pitr', config.save_path, 0)
    else:
        logger = setup_logger('pitr', "./", 0)
    logger.info(config)

    ## dataloader
    dataloader = get_dataloader(config)

    ## model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # root = "/home/panzx/dataset/dependency/ckpt/transformers/{}".format(config.model)
    root = "/home/panzx/dataset/dependency/ckpt/transformers/hf_clip/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path=root).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=root)

    image_embeds = None
    cur = time.time()
    for batch in tqdm(dataloader):
        ## data path
        image_paths, captions, ids = batch
        with torch.no_grad():
            images = [Image.open(path) for path in image_paths]
            # inputs = processor(text=captions,images=images, return_tensors="pt", padding="max_length")
            inputs = processor(text=captions,images=images, return_tensors="pt", padding=True, 
                               max_length=77, truncation=True,).to(device)
            outputs = model(**inputs)
            image_embed = outputs["image_embeds"]
            text_embed = outputs["text_embeds"]

        if image_embeds is None:
            image_embeds = np.zeros((len(dataloader.dataset), image_embed.size(1)))
            text_embeds = np.zeros((len(dataloader.dataset), text_embed.size(1)))
            # all_audio_ids = np.zeros(len(dataloader.dataset))
                            
        # cache embeddings
        image_embeds[ids] = image_embed.data.cpu().numpy().copy()
        text_embeds[ids] = text_embed.data.cpu().numpy().copy()
        # all_audio_ids[ids] = audio_ids.data.cpu().numpy().copy()
    rep_time = time.time() - cur 
    logger.info(f"representation time: {rep_time:.4f}s")

    # if config.save_path:
    #     np.save("{}/{}_{}_imgs.npy".format(config.save_path, config.dataset, config.split), image_embeds)
    #     np.save("{}/{}_{}_txts.npy".format(config.save_path, config.dataset, config.split), text_embeds)
    
    sims = image_embeds @ text_embeds.T
    sims = sims[::5,:]
    npts = sims.shape[0]

    if config.save_path:np.save(config.save_path, {'npts': npts, 'sims': sims})

    ## baseline
    cur = time.time()
    (r1, r5, r10, r50, medr, meanr) = i2t(npts, sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2i(npts, sims)
    rank_time = time.time() - cur 
    logger.info(f"ranking time: {rank_time:.4f}s")
    logger.info("zero-shot results of CLIP baseline:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    ## is norm
    cur = time.time()
    sims_i2t = sims * isnorm(sims, tau=0.01, axis=0)
    sims_t2i = sims * isnorm(sims, tau=0.05, axis=1)
    is_time = time.time() - cur 
    logger.info(f"is time: {is_time:.4f}s")

    (r1, r5, r10, r50, medr, meanr) = i2t(npts, sims_i2t)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2i(npts, sims_t2i)
    logger.info("is norm CLIP:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    cur = time.time()
    sims = otnorm(sims) 
    sn_time = time.time() - cur 
    logger.info(f"sn time: {sn_time:.4f}s")
    (r1, r5, r10, r50, medr, meanr) = i2t(npts, sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2i(npts, sims)
    logger.info("ot norm CLIP:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))