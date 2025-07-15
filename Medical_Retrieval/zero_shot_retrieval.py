import os
import ot
import csv
global logger
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

from dataloader import get_val_dataloader
from logger import setup_logger

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/panzx/dataset/clinic',
                        help='path to datasets')
    parser.add_argument('--dataset', default='books_set',
                        help='books_set, pubmed_set')
    parser.add_argument('--model', default='plip',
                        help='plip, clip')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    return parser


def ranking(sims, return_ranks=False):
    """
    image -> text retrieval
    args:
        sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
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

    dataloader = get_val_dataloader(config)

    ## model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "/home/panzx/dataset/dependency/ckpt/transformers/{}".format(config.model)
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path=root).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=root)

    image_embeds = None
    data_list=list()
    for batch in tqdm(dataloader):
        ## data path
        (image_paths, captions), ids = batch
        data_list += list(zip(image_paths, captions))
        with torch.no_grad():
            images = [Image.open(path) for path in image_paths]
            # inputs = processor(text=captions,images=images, return_tensors="pt", padding="max_length")
            inputs = processor(text=captions,images=images, return_tensors="pt", padding=True, 
                               max_length=77, truncation=True, ).to(device)
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

    sims = image_embeds @ text_embeds.T
    torch.save(sims,"visualization/%s_sims.pt"%config.dataset)

    with open('visualization/%s_anno.csv'%config.dataset, mode='w', newline='') as file:
        writer = csv.writer(file)
        for pair in data_list:
            writer.writerow(pair)

    ## baseline
    (r1, r5, r10, r50, medr, meanr) = ranking(sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = ranking(sims.T)
    logger.info("zero-shot results of PLIP baseline:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    ## is norm
    sims_i2t = sims * isnorm(sims, tau=0.01, axis=0)
    (r1, r5, r10, r50, medr, meanr) = ranking(sims_i2t)
    sims_t2i = sims * isnorm(sims, tau=1, axis=1)
    (r1a, r5a, r10a, r50a, medra, meanra) = ranking(sims_t2i)
    logger.info("is norm PLIP:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))

    sims = snnorm(sims) 
    (r1, r5, r10, r50, medr, meanr) = ranking(sims)
    (r1a, r5a, r10a, r50a, medra, meanra) = ranking(sims.T)
    logger.info("ot norm PLIP:")
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, r50, medr, meanr))
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1a, r5a, r10a, r50a, medra, meanra))