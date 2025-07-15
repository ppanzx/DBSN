import os
import ot
global logger
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoProcessor, ClapModel, AutoFeatureExtractor

from dataloader import get_val_dataloader
from logger import setup_logger

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data5/panzhengxin/dataset/AudioTextRetrieval', help='path to datasets')
    parser.add_argument('--dataset', default='AudioCaps',
                        help='AudioCaps, Clotho')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    return parser

def a2t(npts, sims, return_ranks=False, mode="coco"):
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


def t2a(npts, sims, return_ranks=False, mode="coco"):
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
    
def snnorm(sims, tau=0.01):
    r = np.ones(sims.shape[0])/sims.shape[0]
    c = np.ones(sims.shape[1])/sims.shape[1]
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
    logger = setup_logger('pitr', "./", 0)
    ## dataloader
    parser = get_argument_parser()
    config = parser.parse_args()
    logger.info(config)

    ## model 
    root = "laion/clap-htsat-fused"
    model = ClapModel.from_pretrained(pretrained_model_name_or_path=root)
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=root)

    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=root)

    dataloader = get_val_dataloader(config)
    text_embeds = None
    for batch in tqdm(dataloader):
        audios, captions, ids, cap_ids = batch
        with torch.no_grad():
            text_inputs = processor(text=captions, audios=None, return_tensors="pt", padding=True)
            text_embed = model.get_text_features(**text_inputs)
            if text_embeds is None:
                image_embeds = np.zeros((len(dataloader.dataset), text_embed.size(1)))
                text_embeds = np.zeros((len(dataloader.dataset)*5, text_embed.size(1)))
            text_embeds[cap_ids] = text_embed.data.cpu().numpy().copy()

            for i,audio in enumerate(audios):
                audio_inputs = processor(audios=audio, sampling_rate=48000, return_tensors="pt")
                image_embed = model.get_audio_features(**audio_inputs)
                image_embeds[ids[i]] = image_embed.data.cpu().numpy().copy()
    sims = image_embeds @ text_embeds.T

    ## baseline
    (r1, r5, r10, r50, medr, meanr) = a2t(sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2a(sims.T)
    logger.info("zero-shot results of CLAP baseline:")
    logger.info("Audio to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to audio: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    ## is norm
    sims_i2t = isnorm(sims, tau=0.01, axis=0)
    (r1, r5, r10, r50, medr, meanr) = a2t(sims_i2t)
    sims_t2i = isnorm(sims, tau=1, axis=1)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2a(sims_t2i)
    logger.info("is norm CLAP:")
    logger.info("Audio to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to audio: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    sims = snnorm(sims) 
    (r1, r5, r10, r50, medr, meanr) = a2t(sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = t2a(sims.T)
    logger.info("ot norm CLAP:")
    logger.info("Audio to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to audio: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))