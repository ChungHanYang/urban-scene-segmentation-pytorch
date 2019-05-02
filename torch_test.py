import torch
import numpy as np
import torchvision.models
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
import glob
import pickle
for k in xrange(174):
    if k < 10:
        fileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/val/munster/munster_00000' + str(k) + '_000019_gtFine_color' + '.png'
        outputfileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/resize_val/munster/munster_00000' + str(k) + '_000019_gtFine_color' + '.png'
    elif k >= 10 and k <= 99:
        fileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/val/munster/munster_0000' + str(k) + '_000019_gtFine_color' + '.png'
        outputfileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/resize_val/munster/munster_0000' + str(k) + '_000019_gtFine_color' + '.png'
    else:
        fileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/val/munster/munster_000' + str(k) + '_000019_gtFine_color' + '.png'
        outputfileName = '/Volumes/Transcend/EEE598_image_understanding/gtFine_trainvaltest/gtFine/resize_val/munster/munster_000' + str(k) + '_000019_gtFine_color' + '.png'
    x = Image.open(fileName)
    transform = transforms.Compose([transforms.Resize((512, 1024))])
    x = transform(x)
    x.save(outputfileName)
    print('save sucess')
