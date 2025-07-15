global logger
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

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
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--save_path', default=None, type=str,
                        help='Path to save the similarity results.')
    return parser

def mixgen(image, text, num, lam=0.5):
    # default MixGen
    for i in range(num):
        # image mixup
        image[i,:] = lam * image[i,:] + (1 - lam) * image[i+num,:]
        # text concat
        text[i] = text[i] + " " + text[i+num]
    return image, text

if __name__=="__main__":
    parser = get_argument_parser()
    config = parser.parse_args()

    print(config)

    ## dataloader
    dataloader = get_dataloader(config)

    ## model 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = "/home/panzx/dataset/dependency/ckpt/transformers/hf_clip/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(pretrained_model_name_or_path=root).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=root)

    image_embeds = None
    for batch in tqdm(dataloader):
        ## data path
        image_paths, captions, ids = batch
        # continue
        with torch.no_grad():
            images = [Image.open(path) for path in image_paths]
            # inputs = processor(text=captions, images=images, return_tensors="pt", padding="max_length")
            # 
            inputs = processor(text=captions,images=images, return_tensors="pt", padding=True, 
                               max_length=77, truncation=True,).to(device)
            
            # ## unnormlized results
            # txt_inputs = {k: inputs[k] for k in ["input_ids","attention_mask"]}
            # img_inputs = {k: inputs[k] for k in ["pixel_values"]}
            # image_embed = model.get_image_features(**img_inputs)
            # text_embed = model.get_text_features(**txt_inputs)

            # normlized results
            outputs = model(**inputs)
            image_embed = outputs["image_embeds"]
            text_embed = outputs["text_embeds"]
            image_embed = image_embed/image_embed.norm(p=2, dim=1, keepdim=True)
            text_embed = text_embed/text_embed.norm(p=2, dim=1, keepdim=True)

        if image_embeds is None:
            image_embeds = np.zeros((len(dataloader.dataset), image_embed.size(1)))
            text_embeds = np.zeros((len(dataloader.dataset), text_embed.size(1)))
                            
        # cache embeddings
        image_embeds[ids] = image_embed.data.cpu().numpy().copy()
        text_embeds[ids] = text_embed.data.cpu().numpy().copy()

    if config.save_path:
        np.save("{}/coco_{}_imgs.npy".format(config.save_path, config.split), image_embeds)
        np.save("{}/coco_{}_txts.npy".format(config.save_path, config.split), text_embeds)
    