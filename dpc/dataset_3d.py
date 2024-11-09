import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import UCF101
from torch.utils.data import Dataset

import os
import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2
sys.path.append('../utils')
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class Kinetics400_full_3d(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 unit_test=False,
                 big=False,
                 return_label=False):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.unit_test = unit_test
        self.return_label = return_label

        if big: print('Using Kinetics400 full data (256x256)')
        else: print('Using Kinetics400 full data (150x150)')

        # get action list
        self.action_dict_encode = {}
        self.action_dict_decode = {}
        action_file = os.path.join('../process_data/data/kinetics400', 'classInd.txt')
        action_df = pd.read_csv(action_file, sep=',', header=None)
        for _, row in action_df.iterrows():
            act_id, act_name = row
            act_id = int(act_id) - 1 # let id start from 0
            self.action_dict_decode[act_id] = act_name
            self.action_dict_encode[act_name] = act_id

        # splits
        if big:
            if mode == 'train':
                split = '../process_data/data/kinetics400_256/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400_256/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')
        else: # small
            if mode == 'train':
                split = '../process_data/data/kinetics400/train_split.csv'
                video_info = pd.read_csv(split, header=None)
            elif (mode == 'val') or (mode == 'test'):
                split = '../process_data/data/kinetics400/val_split.csv'
                video_info = pd.read_csv(split, header=None)
            else: raise ValueError('wrong mode')

        drop_idx = []
        print('filter out too short videos ...')
        for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
            vpath, vlen = row
            if vlen-self.num_seq*self.seq_len*self.downsample <= 0:
                drop_idx.append(idx) 
        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val': self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        if self.unit_test: self.video_info = self.video_info.sample(32, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen-self.num_seq*self.seq_len*self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen-self.num_seq*self.seq_len*self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1)*self.downsample*self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len),0)*self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath) 
        
        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq*self.seq_len)
        
        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i+1))) for i in idx_block]
        t_seq = self.transform(seq) # apply same transform
        
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1,2)

        if self.return_label:
            try:
                vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)

            label = torch.LongTensor([vid])
            return t_seq, label

        return t_seq

    def __len__(self):
        return len(self.video_info)

    def encode_action(self, action_name):
        '''give action name, return category'''
        return self.action_dict_encode[action_name]

    def decode_action(self, action_code):
        '''give action code, return action name'''
        return self.action_dict_decode[action_code]


class MovingMNIST(Dataset):
    def __init__(self, root, split='train', train_ratio=0.8, val_ratio=0.2, transform=None):
        self.root = Path(root)
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.transform = transform

        # Load the dataset
        data_path = self.root / 'moving_mnist_videos.npy'
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}.")

        self.data = np.load(data_path, mmap_mode='r')  # Use memory-mapped mode for efficiency
        total_samples = self.data.shape[0]
        train_end = int(total_samples * self.train_ratio)
        val_end = int(total_samples * (self.train_ratio + self.val_ratio))

        if self.split == 'train':
            self.data = self.data[:train_end]
        elif self.split == 'val':
            self.data = self.data[train_end:val_end]
        else:
            raise ValueError("split must be 'train' or 'val'")

        print(f"Total samples: {total_samples}")
        print(f"Train end index: {train_end}")
        print(f"Validation end index: {val_end}")
        print(f"{self.split} dataset size: {self.data.shape[0]}")

    def __getitem__(self, index):
        video = self.data[index]

        # Convert to torch tensor and add channel dimension
        video = torch.tensor(video, dtype=torch.float32).unsqueeze(1)

        # Apply transformations
        if self.transform:
            video = self.apply_transforms(video)

        # Ensure the tensor has the correct shape (N, C, SL, H, W)
        N = 10  # Number of blocks
        SL = 10  # Sequence length per block
        C, H, W = video.shape[1], video.shape[2], video.shape[3]
        video = video.view(N, SL, C, H, W).permute(0, 2, 1, 3, 4)  # Reshape and permute to (N, C, SL, H, W)

        return video

    def __len__(self):
        return self.data.shape[0]

    def apply_transforms(self, video):
        '''Apply the given transformations to the video frames'''
        transformed_video = []
        for frame in video:
            if self.transform:
                # Convert tensor to PIL image
                frame = transforms.ToPILImage()(frame)
                # Convert grayscale to RGB
                frame = frame.convert("RGB")

                # Apply the transformation
                frame = self.transform([frame])[0]  # Assuming the transform expects a list and returns a list
            transformed_video.append(frame)
        return torch.stack(transformed_video)