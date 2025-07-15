#!/usr/bin/env python3
# coding: utf-8
# @Author  : Zhengxin Pan, Zhejiang University
# @E-mail  : panzx@zju.edu.cn
import csv
import os.path as osp
import json
from glob import glob
import librosa
import numpy as np
from contextlib import suppress

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import torchaudio
import torchvision

def load_csv_file(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def get_mel(audio_data, audio_cfg):
    # mel shape: (n_mels, T)
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_cfg['sample_rate'],
        n_fft=audio_cfg['window_size'],
        win_length=audio_cfg['window_size'],
        hop_length=audio_cfg['hop_size'],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm=None,
        onesided=True,
        n_mels=audio_cfg['mel_bins'],
        f_min=audio_cfg['fmin'],
        f_max=audio_cfg['fmax']
    ).to(audio_data.device)
    
    mel = mel_tf(audio_data)
    # Align to librosa:
    # librosa_melspec = librosa.feature.melspectrogram(
    #     waveform,
    #     sr=audio_cfg['sample_rate'],
    #     n_fft=audio_cfg['window_size'],
    #     hop_length=audio_cfg['hop_size'],
    #     win_length=audio_cfg['window_size'],
    #     center=True,
    #     pad_mode="reflect",
    #     power=2.0,
    #     n_mels=audio_cfg['mel_bins'],
    #     norm=None,
    #     htk=True,
    #     f_min=audio_cfg['fmin'],
    #     f_max=audio_cfg['fmax']
    # )
    # we use log mel spectrogram as input
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T  # (T, n_mels)

def _init_defaut_audio_config(file_path="model_cfg.json"):
    with open(file_path, "r") as json_file:
        config = json.load(json_file)
    return config

def get_audio_features(sample, audio_data, max_len, data_truncating, data_filling, audio_cfg=None, require_grad=False):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    require_grad: whether to require gradient for audio data.
        This is useful when we want to apply gradient-based classifier-guidance.
    """
    grad_fn = suppress if require_grad else torch.no_grad
    with grad_fn():
        if len(audio_data) > max_len:
            if data_truncating == "rand_trunc":
                longer = torch.tensor([True])
            elif data_truncating == "fusion":
                # fusion
                mel = get_mel(audio_data, audio_cfg)
                # split to three parts
                chunk_frames = max_len // audio_cfg['hop_size'] + 1  # the +1 related to how the spectrogram is computed
                total_frames = mel.shape[0]
                if chunk_frames == total_frames:
                    # there is a corner case where the audio length is
                    # larger than max_len but smaller than max_len+hop_size.
                    # In this case, we just use the whole audio.
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([False])
                else:
                    ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                    # print('total_frames-chunk_frames:', total_frames-chunk_frames,
                    #       'len(audio_data):', len(audio_data),
                    #       'chunk_frames:', chunk_frames,
                    #       'total_frames:', total_frames)
                    if len(ranges[1]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[1] = [0]
                    if len(ranges[2]) == 0:
                        # if the audio is too short, we just use the first chunk
                        ranges[2] = [0]
                    # randomly choose index for each part
                    idx_front = np.random.choice(ranges[0])
                    idx_middle = np.random.choice(ranges[1])
                    idx_back = np.random.choice(ranges[2])
                    # select mel
                    mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                    mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                    mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                    # shrink the mel
                    mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, audio_cfg['mel_bins']])(mel[None])[0]
                    # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                    # stack
                    mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                    sample["mel_fusion"] = mel_fusion
                    longer = torch.tensor([True])
            else:
                raise NotImplementedError(
                    f"data_truncating {data_truncating} not implemented"
                )
            # random crop to max_len (for compatibility)
            overflow = len(audio_data) - max_len
            idx = np.random.randint(0, overflow + 1)
            audio_data = audio_data[idx: idx + max_len]

        else:  # padding if too short
            if len(audio_data) < max_len:  # do nothing if equal
                if data_filling == "repeatpad":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat)
                    # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "pad":
                    audio_data = F.pad(
                        audio_data,
                        (0, max_len - len(audio_data)),
                        mode="constant",
                        value=0,
                    )
                elif data_filling == "repeat":
                    n_repeat = int(max_len / len(audio_data))
                    audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                else:
                    raise NotImplementedError(
                        f"data_filling {data_filling} not implemented"
                    )
            if data_truncating == 'fusion':
                mel = get_mel(audio_data, audio_cfg)
                mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                sample["mel_fusion"] = mel_fusion
            longer = torch.tensor([False])

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample

class AudioCaptionDataset(Dataset):
    def __init__(self, root):
        self.root = root 
        self.make_pairs()
        
    def process_audio(self, path):
        sampling_rate = 48000
        audio_waveform, _ = librosa.load(path, sr=sampling_rate)           
        audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
        audio_waveform = torch.from_numpy(audio_waveform).float()
        audio_dict = get_audio_features(
            {}, audio_waveform, sampling_rate, 
            data_truncating='fusion',
            data_filling='repeatpad',
            audio_cfg=_init_defaut_audio_config(),
            require_grad=audio_waveform.requires_grad,
        )
        audio_waveform = audio_dict["waveform"]
        return audio_waveform
        
    def make_pairs(self,):
        csv = load_csv_file("{}/csv_files/val.csv".format(self.root))
        self.at_pais=list()
        for item in csv:
            audio_path = "{}/waveforms/val/{}".format(self.root,item["file_name"])
            audio_data = self.process_audio(audio_path)
            captions = list()
            for j in range(1,6):
                captions.append(item["caption_%d"%j])
            self.at_pais.append((audio_data, captions))
    
    def __getitem__(self, index):
        return self.at_pais[index],index
    
    def __len__(self,):
        return len(self.at_pais)


def collate_fn(batch_data):
    # 初始化存储结果的列表
    audios = []
    captions = []
    ids = []
    
    # 遍历batch_data中的每个元素
    for item, idx in batch_data:
        audio, cap_list = item  # 解包每个样本的数据
        audios.append(audio)  # 添加音频路径
        captions.extend(cap_list)  # 将这个样本的所有标题添加到captions列表中
        ids.append(idx)  # 添加索引（id）

        cap_ids = [5*id+i for id in ids for i in range(5)]
    # audios = torch.stack(audios, 0)
        
    # 返回处理后的数据
    return audios, captions, ids, cap_ids


def get_val_dataloader(config):
    dataset = AudioCaptionDataset(osp.join(config.root, config.dataset))
    return DataLoader(dataset=dataset,
                      batch_size=config.batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=config.num_workers,
                      collate_fn=collate_fn)