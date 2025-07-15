"""
    refer to [Openai/clip](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
"""
import os
import ot
import sys
import json
import logging
import argparse
from tqdm import tqdm

import clip
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader

## arguments
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/panzx/dataset/ImageNet',
                        help='path to datasets')
    parser.add_argument('--split', default='val',
                        help='train, val, test')
    parser.add_argument('--huggingface', action='store_true', default=False)
    parser.add_argument('--model', default='ViT-B/32',
                        help='clip variants')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
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
    r = torch.ones(sims.shape[0],device=sims.device)/sims.shape[0]
    c = torch.ones(sims.shape[1],device=sims.device)/sims.shape[1]
    P = ot.sinkhorn(r,c,1-sims,reg=tau)
    return P

## Inverted-Softmax-based normalization in Pytorch
def isnorm(sims, tau=0.01, dim=-1):
    sims = sims*torch.softmax(sims/tau, dim=dim)
    return sims

## save 
def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        json.dump(lst, file)

def load_list_from_file(filename):
    with open(filename, 'r') as file:
        lst = json.load(file)
    return lst

## 
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

clips={"RN50":"rn50", "RN101":"rn101", "RN50x4":"rn50x4", "RN50x16":"rn50x16", "RN50x64":"rn50x64",
        "ViT-B/32":"vitb32", "ViT-B/16":"vitb16", "ViT-L/14":"vitl14", "ViT-L/14@336px":"vitl14@336px"}

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
    if config.huggingface:
        model = CLIPModel.from_pretrained(config.model).to(device)
        processor = CLIPProcessor.from_pretrained(config.model)
    else:
        model, processor = clip.load(config.model)
        model.to(device)

    ## dataloader
    imagenet = ImageNet(root=config.root, split="val", transform=processor)
    dataloader = DataLoader(imagenet, batch_size=config.batch_size, 
                                num_workers=config.num_workers,
                                shuffle=False, drop_last=False,)

    ## prompts
    imagenet_classes = load_list_from_file("imagenet_classes.json")
    imagenet_templates = load_list_from_file("imagenet_templates.json")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(imagenet_classes):
            texts = [template.format(classname) for template in imagenet_templates] #format with class
            if config.huggingface:
                inputs = processor(texts, padding=True, return_tensors="pt").to(device)
                class_embeddings = model.get_text_features(**inputs)
            else:
                texts = clip.tokenize(texts).to(device) #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    ## evaluation
    bs = config.batch_size
    ds = len(dataloader.dataset)
    logits = torch.zeros(ds,1000).to(device)
    targets = torch.zeros(ds).to(device)
    features = torch.zeros(ds, 512) if "336" not in config.model else torch.zeros(ds, 768)
    for i, (images, target) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        target = target.to(device)

        # predict
        with torch.no_grad():
            if config.huggingface:
                image_features = model.get_image_features(images)
            else:
                image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logit = image_features @ zeroshot_weights
        logits[i*bs:(i+1)*bs] =  logit
        targets[i*bs:(i+1)*bs] =  target
        features[i*bs:(i+1)*bs] =  image_features.cpu().detach()

    if config.save_path:
        torch.save(features,config.save_path+"/Imagenet_val_{}_features.pt".format(clips[config.model]))
        torch.save(zeroshot_weights.cpu().detach(),config.save_path+"/Imagenet_val_{}_centroids.pt".format(clips[config.model]))

    # measure accuracy
    acc1, acc5, acc10 = accuracy(logits, targets, topk=(1, 5, 10))
    logger.info("baseline")
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
