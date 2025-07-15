import os
import ot
import sys
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

## arguments
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/panzx/dataset/ImageNet',
                        help='path to datasets')
    parser.add_argument('--split', default='val', help='train, val, test')
    parser.add_argument('--model', default='resnet152',
                        help='resnet variants',choices=["resnet50","resnet152"])
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--save_path', default=None, type=str,
                        help='Path to save the similarity results.')
    return parser

## logger
def setup_logger(name, save_dir, dist_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    # don't log results for the non-master process
    if dist_rank > 0:
        return logger
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
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
    
## Optimal-Transport-based normalization in Pytorch
def snnorm(sims, tau=0.01):
    r = torch.ones(sims.shape[0],device=sims.device)
    c = torch.ones(sims.shape[1],device=sims.device)*50
    P = ot.sinkhorn(r,c,1-sims,reg=tau)
    return P

## Inverted-Softmax-based normalization in Pytorch
def isnorm(sims, tau=0.01, dim=-1):
    sims = sims*torch.softmax(sims/tau, dim=dim)
    return sims

## accuracy
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

if __name__=="__main__":
    parser = get_argument_parser()
    config = parser.parse_args()

    if config.save_path:
        if not os.path.exists(config.save_path):os.makedirs(config.save_path)
        logger = setup_logger('pitr', config.save_path, 0)
    else:
        logger = setup_logger('pitr', "./", 0)
    logger.info(config)

    ## model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.model=="resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
    elif config.model=="resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=weights)
    model.eval()
    model.to(device)
    model_children = list(model.children())
    feat_model = nn.Sequential(*model_children[:-1])
    centroids = model.fc.state_dict()["weight"]
    bias = model.fc.state_dict()["bias"]

    ## dataloader
    transform = weights.transforms()
    imagenet = ImageNet(root=config.root, split="val", transform=transform)
    dataloader = DataLoader(imagenet, batch_size=config.batch_size, 
                                num_workers=config.num_workers,
                                shuffle=False, drop_last=False,)

    ## evaluation
    bs = config.batch_size
    ds = len(dataloader.dataset)
    bl_logits = torch.zeros(ds,1000).to(device)
    logits = torch.zeros(ds,1000).to(device)
    features = torch.zeros(ds, 2048)
    targets = torch.zeros(ds).to(device)
    for i, (images, target) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        target = target.to(device)

        # predict
        with torch.no_grad():
            feature = feat_model(images)
            feature = torch.flatten(feature, 1)

            bl_logit = feature @ centroids.t() + bias
            ## norm

            feature /= feature.norm(dim=-1, keepdim=True)
            centroid = centroids/centroids.norm(dim=-1, keepdim=True)

            logit = feature @ centroid.t()
        bl_logits[i*bs:(i+1)*bs] =  bl_logit
        logits[i*bs:(i+1)*bs] =  logit
        targets[i*bs:(i+1)*bs] =  target
        features[i*bs:(i+1)*bs] =  feature.cpu().detach()

    if config.save_path:
        torch.save(features,config.save_path+"/Imagenet_val_{}_features.pt".format(config.model))
        torch.save(centroids.cpu().detach(),config.save_path+"/Imagenet_val_{}_centroids.pt".format(config.model))

    # measure accuracy
    acc1, acc5, acc10 = accuracy(bl_logits, targets, topk=(1, 5, 10))
    logger.info("baseline")
    logger.info(f"Top-1 accuracy: {100*acc1/ds:.2f}")
    logger.info(f"Top-5 accuracy: {100*acc5/ds:.2f}")
    logger.info(f"Top-10 accuracy: {100*acc10/ds:.2f}")

    # measure accuracy
    acc1, acc5, acc10 = accuracy(logits, targets, topk=(1, 5, 10))
    logger.info("modified")
    logger.info(f"Top-1 accuracy: {100*acc1/ds:.2f}")
    logger.info(f"Top-5 accuracy: {100*acc5/ds:.2f}")
    logger.info(f"Top-10 accuracy: {100*acc10/ds:.2f}")

    # logits = feature @ centroids.t()
    is_logits = isnorm(logits, tau=0.02, dim=-2)
    acc1, acc5, acc10 = accuracy(is_logits, targets, topk=(1, 5, 10))
    logger.info("is-norm results")
    logger.info(f"Top-1 accuracy: {100*acc1/ds:.2f}")
    logger.info(f"Top-5 accuracy: {100*acc5/ds:.2f}")
    logger.info(f"Top-10 accuracy: {100*acc10/ds:.2f}")

    ot_logits = snnorm(logits, tau=0.01)
    acc1, acc5, acc10 = accuracy(ot_logits, targets, topk=(1, 5, 10))
    logger.info("ot-norm results")
    logger.info(f"Top-1 accuracy: {100*acc1/ds:.2f}")
    logger.info(f"Top-5 accuracy: {100*acc5/ds:.2f}")
    logger.info(f"Top-10 accuracy: {100*acc10/ds:.2f}")