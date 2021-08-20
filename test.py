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

def evaluate(name, step, device, batch_size, num_workers):

    PATH = './checkpoint/'
    model = torch.load(PATH + str(name) + '_' + str(step) + '_' + 'model.pt').to(device)

    model.eval()

    testloader = dataset.test_dataset(batch_size=batch_size, num_workers=num_workers)

    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            test_loss += criterion(outputs, targets).item()
            _, predicted = torch.topk(outputs, 1)
            predicted = predicted.view(-1)
            _, predicted_5 = torch.topk(outputs, 5)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # transform the targets layer to compare with top_5 predicted layer
            targets_trans = targets.view(-1, 1)
            correct_5 += torch.eq(targets_trans, predicted_5).sum().item()
            
            top_1_err = 100.0 - 100. * correct/total
            top_5_err = 100.0 - 100. * correct_5/total

            acc = 100.*correct/total

            utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% | top_1_err: %.3f%% | top_5_err: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), acc, top_1_err, top_5_err, correct, total))
        
    return acc, test_loss, top_1_err, top_5_err

if __name__ == '__main__':
 
    # Setting the GPU by ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_gpus = torch.cuda.device_count()

    print('Count of using GPUs : ', n_gpus) 

    # Global variables
    best_acc = 0

    # Preparing Cifar10 dataset
    print('==> Preparing Dataset..')
    testloader = dataset.test_dataset(batch_size=64, num_workers=2)

    print('==> Building Model..')

    # for using multi GPUs
    # model = vgg.VGG('VGG16')

    criterion = nn.CrossEntropyLoss()

    evaluate()

    print('Finished Evaluating')