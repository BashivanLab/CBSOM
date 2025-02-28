import torch
import torch.nn as nn
import sys
import argparse
import os
import numpy as np
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import math
from statistics import mean
import json
import time
from torch.utils.tensorboard import SummaryWriter


from SOCONV_Class import SOCONV
from trunks import ResNet,BasicBlock

epochs=90

batch_size=256

decay_rate=-3

weight_decay=0

alpha_low = 0.008

alpha_high = 0.01

learning_rates = []

mode=f'sub_decay_rate_{decay_rate}'

layers_to_monitor = ["soconv1", "layer1.0.soconv1", "layer3.0.soconv1", "layer4.0.soconv1", "layer4.1.soconv1" ]

TRAINED_MODEL_PATH = os.path.join(f"rn_18_mode_5_{mode}_decay_rate_{decay_rate}_alpha_low={alpha_low}_alpha_high={alpha_high}_batchsize_{batch_size}_step_size ={60,80}_weight_dc={weight_decay}")
MODEL_CHECKPOINT_PATH= os.path.join(f'run1')
writer = SummaryWriter(TRAINED_MODEL_PATH)
#os.makedirs(TRAINED_MODEL_PATH)
postfix = 1
safe_path = TRAINED_MODEL_PATH
while os.path.exists(safe_path):
  safe_path = TRAINED_MODEL_PATH + f'_{postfix}'
  postfix += 1
TRAINED_MODEL_PATH = safe_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(state, filename):
    torch.save(state, filename) 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
args = parser.parse_args()


num_workers =  8
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


net = ResNet(BasicBlock, 18).to(device)    

print(net)


optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay) 

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,80], gamma=0.1) 

cross_el=nn.CrossEntropyLoss()


def flatten_weights(w):
    num_channels = w.shape[0]
    return w.reshape(num_channels, -1)

def cosine_similarity(w1, w2):
    cos = nn.CosineSimilarity(dim=1)
    return cos(w1, w2)
batch_times = [] 
for epoch in range(epochs):   
    print('epoch: ', {epoch})

    
    net.train()

    for index, data in enumerate(train_loader, 0):
        start_time = time.time()
        
        #weight_values = {}
        #difference_weight = {}    

        inputs, labels = data
        inputs=inputs.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
    
        out_puts = net(inputs)

        loss=cross_el(out_puts,labels)
    

        loss.backward()

        print(f'loss: {loss.item()} ')

        
        optimizer.step()
    
      
        net.self_organize((epoch * len(train_loader)) + index, epochs * len(train_loader))
        
        if ((epoch * len(train_loader)) + index) % 500 == 0:
            with torch.no_grad():
                net.eval()
                correct = 0
                total = 0
                for data in test_loader:
                    images, labels = data
                    images=images.to(device)
                    labels=labels.to(device)
                    outputs = net(images)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))

            writer.add_scalar("accuracy", 100 * correct / total, (epoch * len(train_loader)) + index)  

    scheduler.step()

    writer.add_scalar("Loss/train", loss, epoch)

    learning_rates.append(optimizer.state_dict()["param_groups"][0]["lr"])

    print(epoch,"epoch") 
    
    if (epoch%20==0 or epoch ==89):
        chkpt_name = os.path.join(MODEL_CHECKPOINT_PATH, f'Large_{epoch}_epochs_batchsize of {batch_size}_step_size ={60,80}_SOResnet_alpha_{alpha_high}_{alpha_low}_weight_dc={weight_decay}_min_0.003.pt')
        save_checkpoint({
             'epoch': epoch,
                'state_dict': net.state_dict(),
             'optimizer' : optimizer.state_dict(),
            }, filename=chkpt_name)
    
   


    w=net.soconv1.conv1.weight.detach().cpu()
    img=torchvision.utils.make_grid(w,nrow=8,padding=2,normalize=True)
    img_numpy=img.permute(1,2,0).numpy()
    writer.add_image(f'Layer 1 of epoch {epoch}',img,0)   


writer.flush()
writer.close()
