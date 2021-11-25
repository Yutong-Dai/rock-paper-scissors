import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import os
import sys
import time
import torch.utils.data as Data
from PIL import Image
#data loader, load all the png images in the path loader and save them in a list and return the list as a numpy
def data_loader(path,label):
    data_list=[]
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img=Image.open(path+filename)
            img=img.resize((28,28))
            img=np.array(img)
            img=img.reshape(1,28,28)
            data_list.append(img)
    data_list=np.array(data_list)
    data_list=data_list.reshape(data_list.shape[0],28,28)
    data_list=torch.from_numpy(data_list)
    data_list=data_list.type(torch.FloatTensor)
    data_list=data_list/255
    data_list=data_list.view(data_list.shape[0],1,28,28)
    data_list=data_list.type(torch.FloatTensor)
    label=torch.from_numpy(label)
    label=label.type(torch.LongTensor)
    return data_list,label


    




data_paper,label= data_loader('/Users/xiewenxuan/Dropbox (LU Student)/rock-paper-scissors/db/raw/src1/paper',0)
print(data_paper.shape)