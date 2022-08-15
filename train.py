from utils import transform,feature_extraction,trim
import time
import os
import random
import numpy as np
from torchvision import datasets, transforms,models
from DataSet import BatchDataset
import torch
import torch.nn  as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from model import SceneClassification_Model
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import shutil
import sys
import math

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs=10
model_path = 'model_0811.pth'
image_size1,image_size2=1080//8,1920//8
scaler = torch.cuda.amp.GradScaler()
#1シーンのフレーム数
T=60
n_channels=3

def loss_fn(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return criterion(preds, labels)
    
def metrics(true_label,pred_label):
    true_num=sum(true_label)
    both=0
    pred=0
    true=0
    p=1/(true_num+0.0001)
    for i,j in zip(true_label,pred_label):
        if i==1 and j==1:
            both+=1
        #余分な検出
        elif i==0 and j==1:
            pred+=1
        #検出ミス
        elif i==1 and j==0:
            true+=1
    if (both+pred+true)==0:
        met=1
    else:
        met=(both-(pred+true)*p)/(true_num+0.0001)
        if met<0:
            met=0
    return met,np.array([true_num,both,pred,true])

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
            epoch_correct=np.array([0,0,0,0])

            for _,batch_data in tqdm(enumerate(dataloaders[phase])):
                    
                optimizer.zero_grad()   
                with torch.set_grad_enabled(phase=="train"):
                    with torch.cuda.amp.autocast():
                        x=feature_extraction(vgg16,batch_data["video"].half().to(device)).detach()
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
                    
                    true=batch_data["labels"].to(torch.long).contiguous().view(-1).tolist()
                    preds=predicted.detach().to("cpu").tolist()
                    _,info=metrics(true,preds)
                    epoch_correct += info
                   
            epoch_loss=epoch_loss/len(dataloaders[phase].dataset)
            epoch_acc=epoch_correct
            
            if phase=="train":
                torch.save(model.state_dict(), model_path)
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
              
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
            if epoch>0:
                plt.plot(train_loss,label="train")
                plt.plot(val_loss,label="valid")
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.legend()
                plt.savefig("loss.png",dpi=200, format="png")

                plt.clf()

                plt.plot(np.array(train_acc)[:,1],label="train")
                plt.plot(np.array(val_acc)[:,1],label="valid")
                plt.xlabel("epochs")
                plt.ylabel("correct_num")
                plt.legend()
                plt.savefig("correct.png",dpi=200, format="png")

                plt.clf()
    
                plt.plot(np.array(train_acc)[:,2],label="train")
                plt.plot(np.array(val_acc)[:,2],label="valid")
                plt.xlabel("epochs")
                plt.ylabel("error_num")
                plt.legend()
                plt.savefig("error.png",dpi=200, format="png")
                plt.clf()

                plt.plot(np.array(train_acc)[:,3],label="train")
                plt.plot(np.array(val_acc)[:,3],label="valid")
                plt.xlabel("epochs")
                plt.ylabel("miss_num")
                plt.legend()
                plt.savefig("miss.png",dpi=200, format="png")
            print("{} | Epoch {}/{}|{:5}  | Loss:{:.4f} Acc:{}".format(time.strftime('%Y/%m/%d %H:%M:%S'),epoch+1,epochs,phase,epoch_loss," ".join(map(str,list(epoch_acc)))))
    
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
    
    vgg16 = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
    vgg16.avgpool = torch.nn.Identity()
    vgg16.classifier = torch.nn.Identity()
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
        batch_size=16,
        shuffle=True,
        num_workers=4
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
        )
    
    model=SceneClassification_Model(14336,1000,2)
    model.to(device)

    lr=1e-5

    weights = torch.tensor([1.0, 100.0]).cuda()
    criterion= nn.CrossEntropyLoss(weight=weights).to(device)
    
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    dataloaders=dict({"train":train_loader,"valid":valid_loader})

    train_loss,val_loss,train_acc,val_acc=train(dataloaders,model)
   
   
    

    


        
