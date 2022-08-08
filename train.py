import torch
import torch.nn as nn
from torchvision import models, transforms
from utils import transform,feature_extraction,trim

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
from model import SceneClassification_Model
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
torch.backends.cudnn.benchmark=True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs=1
model_path = 'model.pth'
image_size1,image_size2=1080//8,1920//8
scaler = torch.cuda.amp.GradScaler()
#1シーンのフレーム数
T=60
n_channels=3

def loss_fn(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(preds, labels)

def train(dataloaders,model):
    train_loss=[]
    train_acc=[]
    val_loss=[]
    val_acc=[]

    print(time.strftime('START:%Y/%m/%d %H:%M:%S'))
    for epoch in range(epochs):

        for phase in["train","valid"]:
            if phase=="train":
                model.train()
            else:
                model.eval()
            
            epoch_loss = 0.0
            epoch_correct=0

            for _,batch_data in tqdm(enumerate(dataloaders[phase])):
                    
                optimizer.zero_grad()   
                with torch.set_grad_enabled(phase=="train"):
                    with torch.cuda.amp.autocast():
                        x=feature_extraction(vgg16,batch_data["video"].to(device)).detach()
                        batch_pred = model(x)

                        loss =loss_fn(
                                batch_pred.contiguous().view(-1,batch_pred.size(-1),),
                                batch_data["labels"].to(torch.long).to(device).contiguous().view(-1),
                                )
                        _, predicted = torch.max(batch_pred.contiguous().view(-1,batch_pred.size(-1),), 1)
                        
                    if phase=="train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                    epoch_loss+=loss.item()*batch_data["video"].size(0)
                    epoch_correct += torch.sum(predicted==batch_data["labels"].to(torch.long).to(device).contiguous().view(-1))
            
            epoch_loss=epoch_loss/len(dataloaders[phase].dataset)
            epoch_acc=epoch_correct.double()/len(dataloaders[phase].dataset)
            
            if phase=="train":
                
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            print("{} | Epoch {}/{}|{:5}  | Loss:{:.4f} Acc:{:.4f}".format(time.strftime('%Y/%m/%d %H:%M:%S'),epoch+1,epochs,phase,epoch_loss,epoch_acc))
    
    print(time.strftime('END:%Y/%m/%d %H:%M:%S'))

    return train_loss,val_loss,train_acc,val_acc

if __name__ == "__main__":
    #データパス取得
    path="../リサイズ"
    category_key=[i for i in os.listdir(path) ]
    category_id=[i for i in range(len(category_key))]
    category={}
    for i,j in zip(category_key,category_id):
        category[i]=j
    path=[os.path.join(path,i) for i in os.listdir(path) ]
    files=[os.path.join(i,j) for i in path for j in os.listdir(i)]
    data=[]
    for file in files:
        data.append([file,category[file.split("\\")[1]]])
    
    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    vgg16.half()
    vgg16.to(device)

    random.seed(100)
    random.shuffle(data)
    n_train=int(len(data)*0.8)
    n_valid=int(len(data)*0.1)
    
    d=[]
    for i in data:
        d.append(i[0])

    train_dataset = BatchDataset(
    files=d[:n_train],
    T=T,
    transform=transform
    )
    valid_dataset = BatchDataset(
        files=d[n_train:n_train+n_valid],
        T=T,
        transform=transform
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
        )
    
    model=SceneClassification_Model(1000,2)
    model.to(device)

    

    lr=1e-5
    criterion= nn.CrossEntropyLoss().to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    dataloaders=dict({"train":train_loader,"valid":valid_loader})

    train_loss,val_loss,train_acc,val_acc=train(dataloaders,model)
    #モデル保存
    torch.save(model.state_dict(), model_path)

    plt.plot(train_loss,label="train")
    plt.plot(val_loss,label="valid")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png",dpi=200, format="png")

    plt.clf()

    plt.plot(train_acc,label="train")
    plt.plot(val_acc,label="valid")
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.legend()
    plt.savefig("acc.png",dpi=200, format="png")

    


        
