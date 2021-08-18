import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
import sys

from Model import vgg
import dataset
import utils

# Setting runable GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', default=2048, type=int, help='Training batch size')
    parser.add_argument('--num_workers', '-n', default=2, type=int, help='Number of workers')
    parser.add_argument('--epoch', '-e', default=500, type=int, help='Number of Epoch')
    parser.add_argument('--lr', '-l', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--output_dir', '-o', default='./result/train/', type=str, help='Override the output directory')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Choose a specific GPU by ID')

    return parser.parse_args()

def evaluate():

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            test_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

args = parse_args()

# Setting the GPU by ID
device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

n_gpus = torch.cuda.device_count()

print('Count of using GPUs : ', n_gpus) 

# Global variables
best_acc = 0

batch_size = int(args.batch_size / n_gpus)
num_workers = int(args.num_workers / n_gpus)

# Preparing Cifar10 dataset
print('==> Preparing Dataset..')
testloader = dataset.test_dataset(batch_size=batch_size, num_workers=num_workers)

print('==> Building Model..')

# for using multi GPUs
# model = vgg.VGG('VGG16')

PATH = './checkpoint/'

model = torch.load(PATH + 'model.pt')

if device == 'cuda':
    model = torch.nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

evaluate()

print('Finished Evaluating')