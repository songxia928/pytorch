from __future__ import print_function

import os
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


from models.ResNet3D_VAE import ResNet3dVAE

from models.capsnet import CapsNet, ReconstructionNet, CapsNetWithReconstruction
from models.capsnet_3d import CapsNet_3D
from models.capsnet_3d_resnet import CapsNet_3D_ResNet
from models.capsnet import MarginLoss
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnet_3d import generate_model

from data_loader import get_train_test_loader 

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--net', type=str, default='resnet_3d', # resnet, capsnet, resnet_3d
                        help='network for train')
    parser.add_argument('--dataset', type=str, default='onePerson_3d', # mnist, cifar10, slice, onePerson_3d
                        help='dataset for train')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',  # 128
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',   #100, 250
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',  # 0.001
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no_cuda', action='store_true', default=False,  # False
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    #parser.add_argument('--with_reconstruction', action='store_true', default=False)  # False
    parser.add_argument('--with_reconstruction', type=int, default=0)  
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader, test_loader, data_size, n_classes = get_train_test_loader(dataset = args.dataset, 
                                                                            batch_size = args.batch_size,
                                                                            **kwargs)

    if args.net == 'resnet':
        model = resnet18(data_size, n_classes)

    elif args.net == 'capsnet': 
        model = CapsNet(data_size, n_classes, args.routing_iterations)
        if args.with_reconstruction:
            reconstruction_model = ReconstructionNet(data_size, n_dim, n_classes=n_classes)
            reconstruction_alpha = 0.0005
            model = CapsNetWithReconstruction(model, reconstruction_model)

    elif args.net == 'resnet_3d':
        model = generate_model(18, data_size, n_classes=n_classes)

    elif args.net == 'capsnet_3d': 
        model = CapsNet_3D(data_size, n_classes, args.routing_iterations)
        if args.with_reconstruction:
            reconstruction_model = ReconstructionNet(data_size, n_dim, n_classes=n_classes)
            reconstruction_alpha = 0.0005
            model = CapsNetWithReconstruction(model, reconstruction_model)

    elif args.net == 'capsnet_3d_resnet': 
        model = CapsNet_3D_ResNet(data_size, n_classes, args.routing_iterations)
        if args.with_reconstruction:
            reconstruction_model = ReconstructionNet(data_size, n_dim, n_classes=n_classes)
            reconstruction_alpha = 0.0005
            model = CapsNetWithReconstruction(model, reconstruction_model)

    elif args.net == 'resnet3d_vae': 
        model = ResNet3dVAE(data_size, n_classes)
        reconstruction_alpha = 0.0005

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)
    loss_fn = MarginLoss(0.9, 0.1, 0.5)

    def train(epoch, best_train):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target, requires_grad=False)
            optimizer.zero_grad()
            if args.with_reconstruction:
                #output, probs = model(data, target)
                output, probs = model(data)
                reconstruction_loss = F.mse_loss(output, data)
                print(reconstruction_loss)
                margin_loss = loss_fn(probs, target)
                loss = reconstruction_alpha * reconstruction_loss + margin_loss
            else:
                output, probs = model(data)
                loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        train_loss /= len(train_loader.dataset)
        train_acc = 100. * float(correct) / len(train_loader.dataset)
        print('|train_loss: {:.6f}, train_acc: {}/{} ({:.2f}%) | train_acc:{:.2f}%, best_epoch:{}'.format(
            train_loss, correct, len(train_loader.dataset),
            train_acc, best_train['acc'], best_train['epoch']))
        return train_loss, train_acc

    def test(best_test):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data, target = Variable(data, volatile=True), Variable(target)
            data, target = Variable(data), Variable(target)

            if args.with_reconstruction:
                #output, probs = model(data, target)
                output, probs = model(data)
                reconstruction_loss = F.mse_loss(output, data, size_average=False).item()
                print(reconstruction_loss)
                margin_loss = loss_fn(probs, target)
                test_loss += loss_fn(probs, target, size_average=False).item()
                test_loss += reconstruction_alpha * reconstruction_loss
            else:
                output, probs = model(data)
                test_loss += loss_fn(probs, target, size_average=False).item()

            pred = probs.data.max(1, keepdim=True)[1]  # get the index of the max probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        test_acc = 100. * float(correct) / len(test_loader.dataset)
        print('|test_loss : {:.6f}, test_acc : {}/{} ({:.2f}%) | best_acc :{:.2f}%, best_epoch:{}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            test_acc, best_test['acc'], best_test['epoch']))
        return test_loss, test_acc


    best_test = {'epoch':-1, 'acc':-1, 'path_model':None}  # epoch, acc 
    best_train = {'epoch':-1, 'acc':-1, }  # epoch, acc 
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch, best_train)
        test_loss, test_acc = test(best_test)
        scheduler.step(test_loss)

        if best_train['acc'] < train_acc:
            best_train = {'epoch': epoch, 
                         'acc': train_acc,
                        }

        if best_test['acc'] < test_acc:
            path_model = './model_saved/{}_{}_acc{:.2f}_epoch{}.pth'.format(args.dataset, args.net, test_acc, epoch) 
            torch.save(model.state_dict(), path_model)
            if best_test['path_model'] is not None and os.path.exists(best_test['path_model']):
                os.remove(best_test['path_model'])

            best_test = {'epoch': epoch, 
                         'acc': test_acc,
                         'path_model': path_model,
                        }

