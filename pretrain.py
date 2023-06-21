from __future__ import print_function
import torch
import clip
import json
from PIL import Image
import argparse
import csv
import os
import collections
import pickle
import random
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import torch.nn.functional as F
from os import path

use_gpu = torch.cuda.is_available()


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
data_path="./filelists/dataset/novel.json"
f=open(data_path,'r',encoding='utf-8')
data=json.load(f)
class_name=data['label_names']
img_path=data['image_names']
img_id=np.array(data['image_labels'])
class_id=np.arange(0,len(class_name),1)

image=Image.open(img_path[0])
images_input=preprocess(image).unsqueeze(0).to(device)

for i in range(1,len(img_path)):
    image=Image.open(img_path[i])
    image_input=preprocess(image).unsqueeze(0).to(device)
    images_input=torch.cat((images_input,image_input),0)

text_inputs=torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)

with torch.no_grad():
    image_features=model.encode_image(images_input)
    text_features=model.encode_text(text_inputs)

#图像特征
output_dict=collections.defaultdict(list)
for out,label in zip(image_features,img_id):
    output_dict[label.item()].append(out)
save_pickle("output.plk",output_dict)
#
# #文本特征
# output_dict=collections.defaultdict(list)
# for out,label in zip(image_features,img_id):
#     output_dict[label.item()].append(out)
# save_pickle("feature/GTSRB/output_txt_10.plk",output_dict)
#
