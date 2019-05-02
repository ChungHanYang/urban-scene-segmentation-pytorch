import torch
import numpy as np
import torchvision.models
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
import glob
import pickle
path = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/val/frankfurt'


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                L.append(os.path.join(root, file))
    return L


z = file_name(path)
i = 0
cnt = 0
while i < len(z):
    #x = Image.open(z[i])
    #transform = transforms.Compose([transforms.Resize((512, 1024))])
    #x = transform(x)
    # outputname = z[i][0:73] + 'resize_val' + z[i][78:] #train
    outputname = z[i][0:73] + 'resize_val' + z[i][76:]  # val
    print (outputname)
    # x.save(outputname)
    #print('save sucess')
    i += 2
