import torch
import torch.nn as nn
from torchvision import models, transforms

import time
import os
import skvideo.io
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

class SceneClassification_Model(nn.Module):
    def __init__(self,inp_dim,out_dim=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8)
        self.transformer_en_model= nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.out_dim=out_dim
        self.fc=torch.nn.Linear(inp_dim,out_dim)
    def forward(self,x):
        x=self.transformer_en_model(x)
        x=self.fc(x)
        return x