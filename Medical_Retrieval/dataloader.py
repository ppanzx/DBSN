#!/usr/bin/env python3
# coding: utf-8
# @Author  : Zhengxin Pan, Zhejiang University
# @E-mail  : panzx@zju.edu.cn

import os.path as osp
import json
from glob import glob
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

### mydataloader

class PathologyImageCaptionDataset(Dataset):
    def __init__(self, root):
        self.root = root 
        self.make_pairs()
        
    def make_pairs(self,):
        self.pais=list()
        with open("{}/captions.json".format(self.root),"r") as f:
            data = json.load(f)
        for _,item in data.items():
            caption = item["caption"]
            # self.pais.append((item["uuid"], caption))
            searches = glob("{}/images/{}*".format(self.root, item["uuid"]))
            if len(searches)==1:
                img_path = searches[0]
                self.pais.append((img_path, caption))
            else:continue
        return 

    def __getitem__(self, index):
        return self.pais[index], index
    
    def __len__(self,):
        return len(self.pais)

def get_val_dataloader(config):
    dataset = PathologyImageCaptionDataset(osp.join(config.root, config.dataset))
    return DataLoader(dataset=dataset,
                      batch_size=config.batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=config.num_workers
                      )