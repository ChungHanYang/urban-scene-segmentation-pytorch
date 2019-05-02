import torch
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.nn.init import constant
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as spio
import pickle
import alexnet
from sklearn.utils import shuffle

torch.manual_seed(11)

def read_img(path, batch_size):
    out_img = np.zeros((batch_size, 3, 1024, 2048))
    count = 0
    for item in path:
        img = np.array(Image.open(item))
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = np.array(transform(img)).reshape(1, 3, 1024, 2048)
        out_img[count,:,:,:] = img
        count += 1
    return out_img

def read_label(path, batch_size):
    out_label = np.zeros((batch_size, 1024, 2048))
    count = 0
    for item in path:
        data = spio.loadmat(item)['y'].reshape(1, 1024, 2048)
        out_label[count,:,:] = data
        count += 1
    return out_label


class BaseNet(torch.nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self):
        raise NotImplementedError

    def fit(self):
        self.train()
        weight = Variable(torch.FloatTensor(np.array([0.94,0.99,0.2,0.1]))).cuda()
        criterion = torch.nn.NLLLoss2d(weight = weight).cuda()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-7)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        for epoch in range(self.epochs):
            scheduler.step()
            img_path_group = glob.glob(self.train_img_path + "/*/*.png")
            img_path_group = shuffle(img_path_group, random_state=epoch)
            label_path_group = []
            for i in img_path_group:
                lb_fn = os.path.splitext(i.split('/')[-1])[0][0:-12] + "_gtFine_color.mat"
                lab = self.train_label_path + "/" + lb_fn.split('_')[0] + '/' + lb_fn
                label_path_group.append(lab)

            num_batch = (len(label_path_group) // self.batch_size) + 1
            running_loss = 0.0
            for iter in range(num_batch):

                img = img_path_group[iter * self.batch_size:(iter + 1) * self.batch_size]
                label = label_path_group[iter * self.batch_size:(iter + 1) * self.batch_size]
                image = read_img(img, self.batch_size)
                target = read_label(label, self.batch_size)
                image = Variable(torch.FloatTensor(image)).cuda()
                target = Variable(torch.LongTensor(target)).cuda()
                optimizer.zero_grad()
                image = self.model(image)
                out = F.log_softmax(self.forward(image), dim = 1)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                if iter % 3 == 2:
                    print('Iteration ' + str(iter)+'/'+str(num_batch)+'----------------------------------Average Loss of 10 iteration of epoch '
                        + str(epoch) + ' is %.8f' %(running_loss/(3)))
                    running_loss = 0.0
                # if iter%200 == 199:
                #     acc = self.validation()
                #     print('Validation of epoch ' + str(epoch) + ' is  %.5f' %acc)

    def validation(self):
        self.eval()
        img_path_group = glob.glob(self.val_img_path + "/*/*.png")
        label_path_group = []
        total = 0
        correct = 0
        for i in img_path_group:
            lb_fn = os.path.splitext(i.split('/')[-1])[0][0:-12] + "_gtFine_color.mat"
            lab = self.val_label_path + "/" + lb_fn.split('_')[0] + '/' + lb_fn
            label_path_group.append(lab)
        num_batch = (len(label_path_group) // self.batch_size) + 1
        for iter in range(num_batch):
            img = img_path_group[iter * self.batch_size:(iter + 1) * self.batch_size]
            label = label_path_group[iter * self.batch_size:(iter + 1) * self.batch_size]
            image = read_img(img, self.batch_size)
            target = read_label(label, self.batch_size)
            image = Variable(torch.FloatTensor(image)).cuda()
            image = self.model(image)
            out = F.log_softmax(self.forward(image), dim=1)
            out = out.max(1)[1].squeeze().cpu().data.numpy()
            target = target.reshape(target.shape[0]*target.shape[1]*target.shape[2])
            out = out.reshape(out.shape[0]*out.shape[1]*out.shape[2])
            total += len(out)
            correct += (out == target).sum().item()
            if iter%10 == 9:
                print('Validation process is %.3f%%' %(100*((float(iter+1))/num_batch)))
        acc = (float(correct)/ float(total))
        return acc


    def predict_example(self, path = ['../segmentation/image/train/aachen/aachen_000014_000019_leftImg8bit.png']):
        self.eval()
        batch = 1
        image = read_img(path, batch)
        image = Variable(torch.FloatTensor(image)).cuda()
        image = self.model(image)
        out = F.log_softmax(self.forward(image), dim=1)
        out = out.max(1)[1].squeeze().cpu().data.numpy()
        return out

# class FCN(BaseNet):
#     def __init__(self, lr, drop, epochs, num_class, batch_size, train_img_path = '../segmentation/image/train',train_label_path = '../segmentation/label/train',
#                  val_img_path ='../segmentation/image/val', val_label_path = '../segmentation/label/val', test_img_path = '../segmentation/image/test'):
#         super(FCN, self).__init__()
#         self.lr = lr
#         self.epochs = epochs
#         self.drop = drop
#         self.num_class = num_class
#         self.batch_size = batch_size
#         self.train_img_path = train_img_path
#         self.train_label_path = train_label_path
#         self.val_img_path = val_img_path
#         self.val_label_path = val_label_path
#         self.test_img_path = test_img_path
#
#         self.conv1 = torch.nn.Conv2d(3,96, kernel_size=(11,11), stride=(4,4), padding=1).cuda()
#         xavier_normal(self.conv1.weight)
#         constant(self.conv1.bias, 0)
#
#         self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding = 1).cuda()
#
#         self.conv2 = torch.nn.Conv2d(96,256, kernel_size=(5,5), stride=(1,1), padding=0).cuda()
#         xavier_normal(self.conv2.weight)
#         constant(self.conv2.bias, 0)
#
#         self.pool2 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding = 0).cuda()
#
#         self.conv3 = torch.nn.Conv2d(256,384, kernel_size=(3,3), stride=(1,1), padding=0).cuda()
#         xavier_normal(self.conv3.weight)
#         constant(self.conv3.bias, 0)
#
#         self.conv4 = torch.nn.Conv2d(384,384, kernel_size=(3,3), stride=(1,1), padding = 0).cuda()
#         xavier_normal(self.conv4.weight)
#         constant(self.conv4.bias, 0)
#
#         self.conv5 = torch.nn.Conv2d(384,256, kernel_size=(3,3), stride=(1,1), padding = 0).cuda()
#         xavier_normal(self.conv5.weight)
#         constant(self.conv5.bias, 0)
#
#         self.pool3 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding = 0).cuda()
#
#         self.conv6 = torch.nn.Conv2d(256,2048, kernel_size=(1,1), stride=(1,1), padding = 0).cuda()
#         xavier_normal(self.conv6.weight)
#         constant(self.conv6.bias, 0)
#
#         self.conv7 = torch.nn.Conv2d(2048,2048, kernel_size=(1,1), stride=(1,1), padding = 0).cuda()
#         xavier_normal(self.conv7.weight)
#         constant(self.conv7.bias, 0)
#
#         self.conv8 = torch.nn.Conv2d(2048,self.num_class, kernel_size=(1,1), stride=(1,1), padding = 0).cuda()
#
#         self.deconv1 = torch.nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=(13,40), stride = (9,8), padding= 0).cuda()
#         xavier_normal(self.deconv1.weight)
#         constant(self.deconv1.bias, 0)
#
#         self.deconv2 = torch.nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=(2,2), stride = (2,2), padding= 0).cuda()
#         xavier_normal(self.deconv2.weight)
#         constant(self.deconv2.bias, 0)
#
#         self.deconv3 = torch.nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=(2,2), stride = (2,2), padding= 0).cuda()
#         xavier_normal(self.deconv3.weight)
#         constant(self.deconv3.bias, 0)
#
#         self.dropout1 = torch.nn.Dropout(p=self.drop)
#     def forward(self,x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         x = self.pool3(x)
#
#         x = F.relu(self.conv6(x))
#         x = F.relu(self.conv7(x))
#         x = F.relu(self.conv8(x))
#
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#
#         return x

class FCN(BaseNet):
    def __init__(self, lr, drop, epochs, num_class, batch_size, model, train_img_path='../segmentation/image/train',
                 train_label_path='../segmentation/label/train',
                 val_img_path='../segmentation/image/val', val_label_path='../segmentation/label/val',
                 test_img_path='../segmentation/image/test'):
        super(FCN, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.drop = drop
        self.num_class = num_class
        self.batch_size = batch_size
        self.train_img_path = train_img_path
        self.train_label_path = train_label_path
        self.val_img_path = val_img_path
        self.val_label_path = val_label_path
        self.test_img_path = test_img_path
        self.model = model


        self.deconv1 = torch.nn.ConvTranspose2d(256, self.num_class, kernel_size=(16, 16), stride=(8, 8),padding=0).cuda()
        xavier_normal(self.deconv1.weight)
        constant(self.deconv1.bias, 0)

        self.deconv2 = torch.nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=(2, 2), stride=(2, 2),
                                                padding=0).cuda()
        xavier_normal(self.deconv2.weight)
        constant(self.deconv2.bias, 0)

        self.deconv3 = torch.nn.ConvTranspose2d(self.num_class, self.num_class, kernel_size=(2, 2), stride=(2, 2),
                                                padding=0).cuda()
        xavier_normal(self.deconv3.weight)
        constant(self.deconv3.bias, 0)

        self.dropout1 = torch.nn.Dropout(p=self.drop)

    def forward(self, x):
        x = self.dropout1(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


if __name__ == '__main__':
    model = alexnet.alexnet(pretrained= True).cuda()
    # config = {'lr': 0.001,
    #           'epochs': 2,
    #           'num_class': 4,
    #           'batch_size': 6,
    #           'drop': 0.5
    #           }
    # seg = FCN(lr = config['lr'],
    #           epochs = config['epochs'],
    #           num_class = config['num_class'],
    #           batch_size = config['batch_size'],
    #           drop = config['drop'],
    #           model = model)

    # seg.fit()
    #
    # torch.save(seg, '../segmentation/model/FCN_2.pt')
    seg = torch.load('../segmentation/model/FCN_2.pt')

    out = seg.predict_example()
    colormap = [[0,0,142],[220,20,60],[128,64,128],[0,0,0]]
    cm = np.array(colormap).astype('uint8')
    pred = cm[out]
    plt.imshow(pred)
    plt.show()

