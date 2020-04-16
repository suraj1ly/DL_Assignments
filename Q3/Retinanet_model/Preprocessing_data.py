#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


pwd


# In[3]:


# Reference : https://github.com/qqadssp/RetinaNet-Pytorch/blob/master/utils/utils.py
def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.
    The default box order is (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


# In[4]:


# Training
def train(trainloader,epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    counter = 0
    for batch_idx, (inputs, targets,bbox) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net([inputs,bbox])
        counter = counter + 1
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# In[5]:


# Initialisation of Parameters
epoch_train = 15
dropout = 0.2
learning_rate = 0.001 
momentum = 0.9
batch_size = 32


# In[6]:


f= open("../train_annotations.json","r")
g = f.read()
annotations_dict = json.loads(g)


# In[7]:


annotations_dict.keys()


# In[8]:


annotations = pd.DataFrame(annotations_dict['annotations'])


# In[9]:


annotations


# In[10]:


def get_bbox(annotations,address):
    return np.array(annotations)[list(np.array(annotations['image_id'])).index(address)][2]

def mapping(list):
    labels = [i for i in range(len(list))]
    dict_labels = {}
    for i in range(len(labels)):
        dict_labels[list[i]] = labels[i]
    return dict_labels,labels


# In[11]:


list_class = os.listdir('./train')
dict_labels,labels = mapping(list_class)
dataset = []
class_ = []
bbox = []
label_main = []
paths = []
class_name = []
counter_except = 0
count_label = []
for i in range(len(list_class)):
    list_classwise = os.listdir('./train/'+list_class[i])
    count_label.append(len(list_classwise))
    for j in list_classwise:
        img = cv2.imread("./train/"+list_class[i]+"/"+j,1)
        try:
            box = get_bbox(annotations,j.split('.')[0])
            bbox.append(box)
            img = cv2.resize(img,(100,100))
            img = np.swapaxes(img,0,2).flatten()
            
            paths.append("./train/"+list_class[i]+"/"+j)
            class_name.append(list_class[i])
            label_main.append(dict_labels[list_class[i]])
            dataset.append(img)
            class_.append(i)
        except:
            counter_except = counter_except +1


# In[18]:


count_label = []
for i in range(len(list_class)):
    list_classwise = os.listdir('./train/'+list_class[i])
    count_label.append(len(list_classwise))


# In[26]:


labels = [i for i in range(len(count_label))]
plt.pie(count_label, labels=labels, autopct='%1.1f%%')
plt.show()


# In[12]:


dataset, label_main,bbox,class_name,paths = shuffle(dataset,label_main,bbox,class_name,paths)


# In[ ]:


# Example format path/to/image.jpg,x1,y1,x2,y2,class_name


# In[ ]:


bbox[0]


# In[ ]:


dict_labels


# In[ ]:


f = open("class_detail.csv","w")
for i in dict_labels.keys():
    f = open("class_detail.csv","a")
    f.write(i)
    f.write(",")
    f.write(str(dict_labels[i]))
    f.write("\n")


# In[ ]:


# Making the csv file 
f = open("mytraindataset.csv","w")
for i in range(len(paths[:1500])):
    f = open("mytraindataset.csv","a")
    f.write(paths[i])
    f.write(",")
    try:
        f.write(str(int(bbox[i][0])))
    except:
        pass
    f.write(",")
    try:
        f.write(str(int(bbox[i][1])))
    except:
        pass
    f.write(",")
    try:
        f.write(str(int(bbox[i][0])+int(bbox[i][3])))
    except:
        pass
    f.write(",")
    try:
        
        f.write(str(int(bbox[i][1]) + int(bbox[i][2])))
    except:
        pass
    f.write(",")
#     for j in range(len(bbox[i])):
#         try:
#             f.write(str(int(bbox[i][j])))
#             f.write(",")
#         except:
#             f.write(",")
    f.write(class_name[i])
    f.write("\n")
    f.close()

    
# Making the csv file 
f = open("myvaliddataset.csv","w")
for i in range(len(paths[1500:])):
    f = open("myvaliddataset.csv","a")
    f.write(paths[i])
    f.write(",")
    try:
        f.write(str(int(bbox[i][0])))
    except:
        pass
    f.write(",")
    try:
        f.write(str(int(bbox[i][1])))
    except:
        pass
    f.write(",")
    try:
        f.write(str(int(bbox[i][0])+int(bbox[i][3])))
    except:
        pass
    f.write(",")
    try:
        
        f.write(str(int(bbox[i][1]) + int(bbox[i][2])))
    except:
        pass
    f.write(",")
    
    f.write(class_name[i])
    f.write("\n")
    f.close()
    


# In[ ]:





# In[ ]:


# Data sample visualization
img = dataset[3].reshape(3,256,480)
img = np.swapaxes(img,0,2)
imgplot = plt.imshow(img)


# In[ ]:


# Here training
train_size = 1500
dataset, label_main,bbox = shuffle(dataset,label_main,bbox)
tensor_X = torch.stack([torch.from_numpy(i) for i in dataset])
tensor_ = torch.stack([torch.from_numpy(np.array(i)) for i in label_main])
tensor_label =  torch.stack([torch.from_numpy(np.array(i)).double() for i in bbox])
X_train = tensor_X[:train_size]
class_train = tensor_[:train_size]
X_test = tensor_X[train_size:]
class_test = tensor_[train_size:]
class_train2 = tensor_label[:train_size]
class_test2 = tensor_label[train_size:]
train_data = torch.utils.data.TensorDataset(X_train, class_train,class_train2)
train_indices = np.arange(len(train_data))
np.random.shuffle(train_indices.tolist())
train_sample = SubsetRandomSampler(train_indices)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sample, num_workers=0)
test_dataset = torch.utils.data.TensorDataset(X_test, class_test,class_test2)
testloader = torch.utils.data.DataLoader(test_dataset)


# In[ ]:


# Initialising the checkpoint 
num_classes = 8
PATH_TO_WEIGHTS = "../pretrained_weights.pt"
net = retinanet_model.resnet50(80)
checkpoint = torch.load(PATH_TO_WEIGHTS)
net.load_state_dict(checkpoint)
net.classificationModel.fc = nn.Linear(720, 8)
net = net.cuda()


# In[ ]:





# In[ ]:





# In[ ]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# In[ ]:


for epoch in range(start_epoch, start_epoch+200):
    train(trainloader,epoch)


# In[ ]:


get_ipython().system('nvidia-smi')


# In[13]:


def tsne_plot(dataset_main,label_main,title,classes):
    #TSNE Plot for glass dataset
    tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
    tsne_results = tsne.fit_transform(dataset_main)

    df_subset = pd.DataFrame()
    df_subset['X'] = tsne_results[:,0]
    df_subset['y']=label_main
    df_subset['Y'] = tsne_results[:,1]
    plt.figure(figsize=(6,4))
    plt.title(title)
    sns.scatterplot(
        x="X", y="Y",
        hue="y",
        palette=sns.color_palette("hls", classes),
        data=df_subset,
        legend="full",
        alpha=1.0
    )


# In[14]:


from sklearn.manifold import TSNE
import seaborn as sns


# In[17]:


tsne_plot(dataset,label_main,"TSNE Plot for MS Coco",8)


# In[5]:


loss1 = np.load("./epoch_loss1.npy",allow_pickle=True)
reg_loss = np.load("./epoch_regloss.npy",allow_pickle=True)
total_loss = np.load("epoch_totalloss.npy",allow_pickle=True)
class_loss = np.load("epoch_classloss.npy",allow_pickle=True)


# In[10]:


iteration = [i for i in range(len(reg_loss))]


# In[13]:


plt.plot(iteration,reg_loss,label="Regression")
plt.plot(iteration,total_loss,label="Total Loss")
plt.plot(iteration,class_loss,label="Classification Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss ")
plt.title("Loss Vs Epoch")
plt.legend()
plt.show()


# In[ ]:




