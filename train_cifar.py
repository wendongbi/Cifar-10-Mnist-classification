from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import argparse

from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from models import *
from torch.nn.utils import clip_grad_norm_
from utils import progress_bar
import numpy as np
import matplotlib.pyplot as plt

train_id = 4
resume_id = 3


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--epoch', default=50, type=int, metavar='N')
parser.add_argument('--weight-decay','--wd',default=1e-4,type=float, metavar='W')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--logspace', default=5, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc_val = 0  # best test accuracy
best_acc_test = 0  # best test accuracy

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
save_path = 'checkpoint/' + str(train_id) + '/'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.Grayscale(),
    # transforms.ColorJitter(1, 1, 1, 0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    # transforms.CenterCrop(42),
    # transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=False)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=12)



# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + str(resume_id) + '/test_ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if args.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
logspace_lr = torch.logspace(np.log10(args.lr), np.log10(args.lr) - args.logspace, args.epoch)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_res = 0
    if args.logspace != 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = logspace_lr[epoch-start_epoch]
    else:
        adjust_learning_rate(optimizer, epoch)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        prob = F.softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        # clip_grad_norm_(parameters=net.parameters(), max_norm=0.1)

        optimizer.step()

        train_loss += loss.item() 
        loss_res = train_loss/(batch_idx + 1)
        

        max_num, predicted = prob.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        train_acc = 100.*correct/total,
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return loss_res, train_acc
        
        

    
def validation(epoch):
    global best_acc_val
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        flag = True
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            prob = outputs
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            # normalization - softmax
            for i in range(len(prob)):
                prob[i] = prob.exp()[i] / prob.exp()[i].sum()
            max_num, predicted = prob.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Val Accu: ' + str(100. * correct / total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_val:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + 'val_ckpt.t7')
        best_acc_val = acc
    return acc
def test(epoch):
    global best_acc_test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        flag = True
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            prob = outputs
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            # normalization - softmax
            for i in range(len(prob)):
                prob[i] = prob.exp()[i] / prob.exp()[i].sum()
            max_num, predicted = prob.max(1)
            


            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        print('Test Accu: ' + str(100. * correct / total))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc_test:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + 'test_ckpt.t7')
        best_acc_test = acc
    return acc


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


test_acc = []
train_loss = []
train_acc = []
if __name__ == "__main__":
    for epoch in range(start_epoch, start_epoch+args.epoch):
        loss, _ = train(epoch)
        acc = test(epoch)
        train_loss.append(loss)
        test_acc.append(acc)

        # plot and save
        # x = np.load('results/val_acc_{}.npy'.format(train_id))
        x = [i for i in range(len(test_acc))]
        y1 = train_loss
        y2 = test_acc
        plt.figure(figsize=(10, 10)).tight_layout(pad=0.4, w_pad=3., h_pad=5.)
        plt.title('ResNet50 Accuracy')
        plt.subplot(211)
        plt.plot(x, y1, color='blue')
        plt.title('train loss')

        plt.subplot(212)
        plt.plot(x, y2, color='red')
        plt.title('train and test accu(best accu:' + str(best_acc_test) + ')')
        plt.savefig('results/result_{}.jpg'.format(train_id))
        # plt.show()
    # save results
    np.save('results/test_acc_{}.npy'.format(train_id), np.array(test_acc))
    np.save('results/train_loss_{}.npy'.format(train_id), np.array(train_loss))

    

    