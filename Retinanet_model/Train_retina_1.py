#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import collections
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter,Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet import csv_eval
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings 
import json
from torch.utils.data.sampler import SubsetRandomSampler
import retinanet.model as retinanet_model
warnings.filterwarnings("ignore")

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))



# In[3]:


dataset_val = CSVDataset(train_file="./myvaliddataset.csv", class_list="./class_detail.csv",
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=4,drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)


# In[9]:


retinanet = torch.load("./model_final_without_finetune2.pt").cuda()


# In[10]:


mAP = csv_eval.evaluate(dataset_val, retinanet)


# In[ ]:


epochs =100


# In[ ]:


dataset_train = CSVDataset(train_file="./mytraindataset.csv", class_list="./class_detail.csv",
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
dataset_val = CSVDataset(train_file="./myvaliddataset.csv", class_list="./class_detail.csv",
                                 transform=transforms.Compose([Normalizer(), Resizer()]))


# In[ ]:


sampler = AspectRatioBasedSampler(dataset_train, batch_size=8, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

if dataset_val is not None:
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

# Create the model

retinanet = model.resnet50(num_classes=dataset_train.num_classes())
use_gpu = True

# Initialising the checkpoint 
num_classes = 8
PATH_TO_WEIGHTS = "../pretrained_weights.pt"
retinanet = retinanet_model.resnet50(80)
checkpoint = torch.load(PATH_TO_WEIGHTS)
retinanet.load_state_dict(checkpoint)
retinanet.classificationModel.fc = nn.Linear(720, num_classes)

if use_gpu:
    retinanet = retinanet.cuda()
print("Model retinanet : ",retinanet)


# In[ ]:


retinanet = torch.nn.DataParallel(retinanet).cuda()
retinanet.training = True
optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
loss_hist = collections.deque(maxlen=500)
retinanet.train()
retinanet.module.freeze_bn()
print('Num training images: {}'.format(len(dataset_train)))

for epoch_num in range(epochs):

    retinanet.train()
    retinanet.module.freeze_bn()
    epoch_loss = []
    for iter_num, data in enumerate(dataloader_train):
        try:
            optimizer.zero_grad()
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            loss = classification_loss + regression_loss
            
            if bool(loss == 0):
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            optimizer.step()
            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue

    print("Evaluating dataset............")
    mAP = csv_eval.evaluate(dataset_val, retinanet)
    scheduler.step(np.mean(epoch_loss))
    torch.save(retinanet.module, '{}_retinanet_{}.pt'.format("dataset_coco", epoch_num))

retinanet.eval()

torch.save(retinanet, 'model_final.pt')


# In[2]:





# In[3]:


pwd


# In[1]:


import numpy as np


# In[3]:


loss = np.load("epoch_loss.npy")


# In[5]:


itr = [i for i in range(len(loss))]
plt.plot(loss,itr)
plt.


# In[6]:


loss


# In[ ]:




