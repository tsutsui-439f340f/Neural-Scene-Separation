import torch
import torch.nn as nn
from torchvision import models, transforms

import time
import os

import random
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from DataSet import BatchDataset
import torch
import torch.nn  as nn
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy import spatial
import seaborn as sns
import shutil
import sys
import math
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

image_size1,image_size2=1080//8,1920//8
n_channels=3

def transform(sample,T):
    video=sample["video"]
    label=sample["labels"]
    paths=sample["paths"]
    frames=sample["frames"]
   
    trans_video = torch.empty(n_channels,T,image_size1,image_size2)
    
    trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size1,image_size2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])
    
    video,s,e=trim(video,T)
    video = torch.Tensor(video)
    
    for i in range(T):
        img = video[:,i]
        img = trans(img)
        img=img.reshape(n_channels,image_size1,image_size2)
        trans_video[:,i] = img
    sample = {'video': trans_video, 'labels': label[s:e],'paths':paths,'frames':frames}
    return sample

def trim(video,T):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :],start,end

def feature_extraction(vgg16,src):
    new_src=[]
    for t in range(src.shape[2]):
        new_src.extend([vgg16(src[:,:,t,:,:])])
    new_src = torch.stack(new_src)
    new_src=new_src.permute(1,0,2)
    return new_src