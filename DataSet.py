import torch
import os
import skvideo.io
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
import torch.nn  as nn
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from torchvision import models, transforms
import numpy as np
from scipy import spatial
import seaborn as sns
import os
import yt_dlp
import shutil
import matplotlib.pyplot as plt
import sys
import math
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, files,T, transform=None):
        
        self.files = files
        self.transform = transform
        self.T=T
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        random_idx=list(np.random.choice(len(self.files),6))
        paths=[self.files[i] for i in random_idx]
        paths.insert(0,self.files[idx])
        
        videos=[]
        frames=[]
        keyframe=[0]
        total_len=0
        
        for path in paths:
            video=skvideo.io.vread(path)
            keyframe.extend([keyframe[-1]+video.shape[0]])
            total_len+=video.shape[0]
            frames.append(video.shape[0])
            videos.append(torch.from_numpy(video))
        videos=torch.cat(videos)
        videos=videos.permute(3, 0, 1, 2)/255.0   
        keyframe=keyframe[:-1]
        label = torch.zeros(total_len)
        label[keyframe] =1
        
        sample = {'video': videos, 'labels': label,'paths':paths,'frames':frames}

        if self.transform:
            sample = self.transform(sample,self.T)

        return sample



