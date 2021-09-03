import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import torchsummary 

from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import time
import sys

from Model import vgg
from Model import resnet
from Model import mobilenet
from Model import resnext
import dataset
import utils
import test

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default='VGG19', type=str, help='Specify Model name')
    parser.add_argument('--flag', '-f', default=0, type=int, help='To Switch Model = 0 : VGG, 1 : RES, 2 : MobileV2, 3 : ResNeXt')
    parser.add_argument('--step', '-s', default=1, type=int, help='Model Step')
    parser.add_argument('--batch_size', '-b', default=2048, type=int, help='Training batch size')
    parser.add_argument('--num_workers', '-n', default=2, type=int, help='Number of workers')
    parser.add_argument('--epoch', '-e', default=200, type=int, help='Number of Epoch')
    parser.add_argument('--lr', '-l', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--op_flag', '-op', default=0, type=int, help='To Switch LR Optimizer = 0 : SGD, 1 : RMSprop, 2 : Adam')
    parser.add_argument('--sh_flag', '-sh', default=0, type=int, help='To Switch LR Scheduler = 0 : MultiStep, 1 : Cosine, 2 : Exponential')
    parser.add_argument('--cardinality', '-card', default=16, type=int, help='When using ResNeXt input Cardinality')
    parser.add_argument('--bottleneck_width', '-bw', default=64, type=int, help='When using ResNeXt input bottleneck_width')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--output_dir', '-o', default='./result/train/', type=str, help='Override the output directory')
    parser.add_argument('--gpu', '-g', default=5, type=int, help='Choose a specific GPU by ID')

    return parser.parse_args()

def train():

    model.train()
    
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), acc, correct, total))

    return acc, train_loss

def validate():

    model.eval()

    test_loss = 0
    correct = 0
    correct_5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):

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

            utils.progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% | top_1_err: %.3f%% | top_5_err: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), acc, top_1_err, top_5_err, correct, total))
        
    return acc, test_loss, top_1_err, top_5_err

def save_ckpt(accuracy, top_1_err, top_5_err):
    global best_acc
    global best_top_1_err
    global best_top_5_err

    if accuracy > best_acc:
        print('Saving Checkpoint..')

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        PATH = './checkpoint/'
        best_acc = accuracy
        best_top_1_err = top_1_err
        best_top_5_err = top_5_err

        torch.save(model, PATH + str(args.model) + '_' + str(args.step) + '_' + 'model.pt')

if __name__ == '__main__':
    # Parsing arguments
    args = parse_args()

    # Setting runable GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Setting the GPU by ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_gpus = torch.cuda.device_count()
    print('Count of using GPUs : ', n_gpus) 

    # Global variables
    best_acc = 0
    best_top_1_err = 0
    best_top_5_err = 0

    # Preparing Cifar10 dataset
    print('==> Preparing Dataset..')
    trainloader, validloader = dataset.train_val_dataset(batch_size=args.batch_size, num_workers=args.num_workers, valid_size=0.1)

    # Building Model
    print('==> Building Model..')

    # for using multi GPUs
    if args.flag == 0:
        model = vgg.VGG(args.model)
    elif args.flag == 1:
        model = resnet.ResNet(args.model)
    elif args.flag == 2:
        model = mobilenet.MobileNetV2()
    elif args.flag == 3:
        model = resnext.ResNeXt(args.model, args.cardinality, args.bottleneck_width)

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    torchsummary.summary(model, input_size=(3, 32 ,32))

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.op_flag == 0:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.op_flag == 1:
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.op_flag == 2:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    if args.sh_flag == 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    elif args.sh_flag == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    elif args.sh_flag == 2:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Using Tensorboard for visualization
    writer = SummaryWriter('runs/'+ str(args.model) + '_' + str(args.step) + '_' + str(args.epoch) + '_' + str(args.lr) + '/')

    start_time = time.time()
    for epoch in range(1, args.epoch + 1):
        epoch_time = time.time()
        print('Epoch : %d' % epoch)

        train_acc, train_los = train()
        val_acc, val_los, top_1_err, top_5_err = validate()

        log_lr = scheduler.optimizer.param_groups[0]['lr']

        save_ckpt(val_acc, top_1_err, top_5_err)

        end_time = time.time()

        curr_time = end_time - epoch_time

        print(f'Current Epoch Training Time : {curr_time}s')
        scheduler.step()

        writer.add_scalar('Loss/train', train_los, epoch)
        writer.add_scalar('Loss/val', val_los, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning rate', log_lr, epoch)
        writer.add_scalar('Top_1_error', top_1_err, epoch)
        writer.add_scalar('Top_5_error', top_5_err, epoch)

    writer.close()

    total_train_time = end_time-start_time

    print(f'Finish Training.. Total Training Time : {total_train_time}s, Best Val Accuracy : {best_acc}%')
    print(f'top-k error of best weight of model, top-1 err : {best_top_1_err}%, top-5 err : {best_top_5_err}%')

    # Evaluate the test dataset from the best accuracy of validation model.
    test_acc, test_los, test_top_1_err, test_top_5_err = test.evaluate(name=args.model,
                                                            step=args.step, 
                                                            device=device, 
                                                            batch_size=args.batch_size, 
                                                            num_workers=args.num_workers)

    print(f'Finish Evaluating.. Test Accuracy : {test_acc}% | Test Loss : {test_los}% | top_1_err : {test_top_1_err}% | top_5_err : {test_top_5_err}%')

    # write the result of testing dataset
    utils.result_note(args.model, args.step, test_acc, test_los, test_top_1_err, test_top_5_err, total_train_time)