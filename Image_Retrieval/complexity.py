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
    parser.add_argument('--batch_size', default=5000, type=int,
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

def isnorm(sims, tau=0.01, dim=0):
    P = torch.softmax(sims/tau, dim=dim)
    return P

def snnorm(sims, tau=0.01):
    r = torch.ones(sims.shape[0], device=sims.device)/sims.shape[0]
    c = torch.ones(sims.shape[1], device=sims.device)/sims.shape[1]
    P = ot.sinkhorn(r, c, 1-sims, reg=tau, numItermax=10)
    return P

def chunk_sort(sim, chunk_size=1):
    N, M = sim.shape
    
    rank_per_element = torch.zeros_like(sim, dtype=torch.int32, device=sim.device)

    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        chunk = sim[i:end_idx]
        sorted_chunk = torch.argsort(torch.argsort(chunk, dim=1), dim=1)
        rank_per_element[i:end_idx] = sorted_chunk
    return rank_per_element
def compute_metrics_cuda(sim, gt, recalls=[1, 5, 10, 50]):
    """
    row-wise compute ranking results using PyTorch on CUDA
    """
    device = sim.device
    gt = gt.to(device)

    # rank_per_element = torch.argsort(torch.argsort(-sim, dim=1), dim=1)
    rank_per_element = chunk_sort(-sim)

    metrics = {}

    for recall in recalls:
        pred = rank_per_element < recall
        _is_gt_recall = (pred & (gt == 1)).sum(dim=1) > 0
        r = _is_gt_recall.sum().float() / _is_gt_recall.shape[0]
        metrics[f"R{recall}"] = r.item() 

    ### no need for overall metrics
    # rank_gt = torch.min(torch.where(gt == 0, sim.shape[1], rank_per_element), dim=1)[0]
    # metrics["MedianR"] = rank_gt.median().item() + 1
    # metrics["MeanR"] = rank_gt.float().mean().item() + 1

    return metrics

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
    sn_device="cuda" 
    image_embeds = torch.zeros((len(dataloader.dataset), 512), device=sn_device)
    text_embeds = torch.zeros((len(dataloader.dataset), 512), device=sn_device)
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

        # if image_embeds is None:
        #     image_embeds = np.zeros((len(dataloader.dataset), image_embed.size(1)))
        #     text_embeds = np.zeros((len(dataloader.dataset), text_embed.size(1)))

        # cache embeddings
        # image_embeds[ids] = image_embed.data.cpu().numpy().copy()
        # text_embeds[ids] = text_embed.data.cpu().numpy().copy()
        image_embeds[ids] = image_embed.to(sn_device)
        text_embeds[ids] = text_embed.to(sn_device)

    rep_time = time.time() - cur 
    logger.info(f"representation time: {rep_time:.4f}s")

    # if config.save_path:
    #     np.save("{}/{}_{}_imgs.npy".format(config.save_path, config.dataset, config.split), image_embeds)
    #     np.save("{}/{}_{}_txts.npy".format(config.save_path, config.dataset, config.split), text_embeds)
    
    sims = image_embeds @ text_embeds.T
    sims = sims[::5,:]
    npts = sims.shape[0]
    gt = (torch.arange(sims.shape[0]).unsqueeze(dim=1)==torch.arange(sims.shape[1]).unsqueeze(dim=0)//5).to(sims.device)

    if config.save_path:np.save(config.save_path, {'npts': npts, 'sims': sims})

    k = 1
    num=10000
    f30k_train_caps = np.load("./runs/querybank/f30k_train_txts.npy", allow_pickle=True)[::k][:num]
    f30k_train_imgs = np.load("./runs/querybank/f30k_train_imgs.npy", allow_pickle=True)[::k*5][:num*5]
    cat_imgs = torch.cat([image_embeds, torch.tensor(f30k_train_imgs, device=sims.device)], axis=0)
    cat_sims = cat_imgs @ torch.tensor(f30k_train_caps, device=sims.device).T

    ## baseline
    cur = time.time()
    bl_metric = compute_metrics_cuda(sims, gt)
    rank_time = time.time() - cur 
    logger.info(f"ranking time: {rank_time:.4f}s")
    logger.info("zero-shot results of CLIP baseline:")
    for k, v in bl_metric.items():
        logger.info(f"{k} = {100*v:.2f}")

    ## is norm
    cur = time.time()
    sims_tbq = image_embeds @ torch.tensor(f30k_train_caps, device=sims.device, dtype=torch.float32).T
    sims_i2t = isnorm( sims_tbq, tau=0.01, dim=0)
    # sims_i2t = isnorm(sims, tau=0.01, dim=0)
    # sims_t2i = isnorm(sims, tau=0.05, dim=1)
    is_time = time.time() - cur 
    logger.info(f"is time: {is_time:.4f}s")
    logger.info(torch.cuda.memory_summary())  # 输出详细分配情况

    # is_metric = compute_metrics_cuda(sims_i2t, gt)
    # is_metric = compute_metrics_cuda(sims_t2i, gt)
    # logger.info("is norm CLIP:")
    # for k, v in is_metric.items():
    #     logger.info(f"{k} = {100*v:.2f}")

    ## sn
    cur = time.time()
    sn_sims = snnorm(sims_tbq) 
    sn_time = time.time() - cur 
    logger.info(f"sn time: {sn_time:.4f}s")
    logger.info(torch.cuda.memory_summary())  # 输出详细分配情况
    # sn_metric = compute_metrics_cuda(sn_sims, gt)
    # logger.info("ot norm CLIP:")
    # for k, v in sn_metric.items():
    #     logger.info(f"{k} = {100*v:.2f}")

    ## dbsn
    cur = time.time()
    sn_sims = snnorm(cat_sims) 
    dbsn_time = time.time() - cur 
    logger.info(f"sn time: {dbsn_time:.4f}s")
    logger.info(torch.cuda.memory_summary())  # 输出详细分配情况
