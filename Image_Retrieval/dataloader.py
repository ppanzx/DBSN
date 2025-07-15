#!/usr/bin/env python3
# coding: utf-8
# @Author  : Zhengxin Pan, Zhejiang University
# @E-mail  : panzx@zju.edu.cn

import os
import os.path as osp
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class ImageCaptionDataset(Dataset):
    def __init__(self, root, split="test"):
        super(ImageCaptionDataset,self).__init__()
        self.root = root 
        self.split = split
        self.make_pairs()
        
    def make_pairs(self,):
        self.nt1i = 5 if self.split=="train" else 1
        loc_cap = osp.join(self.root, 'precomp')
        loc_image = osp.join(self.root, 'precomp')
        loc_mapping = osp.join(self.root, 'id_mapping.json')
        if 'coco' in self.root:
            image_base = osp.join(self.root, 'images')
            if self.split=="test":self.split="testall"
        else:
            image_base = osp.join(self.root, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            id_to_path = json.load(f_mapping)

        # Read Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % self.split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(self.split)), 'r') as f:
            lines = f.readlines()
            image_ids = [int(x.strip()) for x in lines]
            self.image_paths = [osp.join(image_base, id_to_path[str(k)]) for k in image_ids]
        return 

    def __getitem__(self, index):
        return self.image_paths[index//self.nt1i], self.captions[index], index
    
    def __len__(self,):
        return len(self.captions)

def get_dataloader(config):
    dataset = ImageCaptionDataset(osp.join(config.root, config.dataset), config.split)
    return DataLoader(dataset=dataset,
                      batch_size=config.batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=config.num_workers,
                      )
