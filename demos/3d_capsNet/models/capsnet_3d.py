from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.optim import lr_scheduler
from torch.autograd import Variable


def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()

        #self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        conv1_t_size=7
        conv1_t_stride=2
        self.conv = nn.Conv3d( input_channels,
                               output_caps*output_dim,
                               kernel_size=(conv1_t_size, 7, 7),
                               #kernel_size=(4, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               #padding=0,
                               bias=False)

        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        #print('input2:', input.shape)
        out = self.conv(input)
        #print('out:', out.shape)
        N, C, D, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)
        #print('out_view1:', out.shape)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        #print('out_premute:', out.shape)
        out = out.view(out.size(0), -1, out.size(4))
        #print('out_view2:', out.shape)
        out = squash(out)
        return out


class CapsNet_3D(nn.Module):
    def __init__(self, data_size, n_classes, routing_iterations):
        super(CapsNet_3D, self).__init__()

        #self.conv1 = nn.Conv2d(data_size['c'], 256, kernel_size=9, stride=1)
        n_input_channels=data_size['n_slice']
        self.in_planes = 64  # 256
        conv1_t_size=7
        conv1_t_stride=1
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               #padding=(0, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        conv1_t_size=5
        conv1_t_stride=2
        self.conv2 = nn.Conv3d(self.in_planes,
                               self.in_planes*2,
                               kernel_size=(conv1_t_size, 5, 5),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 2, 2),
                               #padding=(0, 0, 0),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(self.in_planes*2)

        self.num_caps = 32
        self.caps_size = 8

        if data_size['dataset'] == 'mnist': size = 6
        elif data_size['dataset'] == 'cifar10': size = 8
        elif '_3d' in data_size['dataset']: size = 20
        self.num_primaryCaps = 32 * size * size

        self.primaryCaps = PrimaryCapsLayer(self.in_planes*2, self.num_caps, self.caps_size, kernel_size=9, stride=2)  # outputs 6*6

        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, self.caps_size, n_classes, 16, routing_module)

    def forward(self, input):
        #print('input:', input.shape)
        #print(input[0])
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        #print('x:', x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        #x = F.relu(x)
        #print('x:', x.shape)
        x = self.primaryCaps(x)
        #print('x:', x.shape, x[0])
        x = self.digitCaps(x)
        probs = x.pow(2).sum(dim=2).sqrt()
        #print('x:', x.shape, x[0])
        #print('probs:', probs.shape, probs[0])
        return x, probs


class ReconstructionNet(nn.Module):
    def __init__(self, data_size, n_dim, n_classes=10):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim * n_classes, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, data_size['h']*data_size['w']*data_size['c'])
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x, target):
        mask = Variable(torch.zeros((x.size()[0], self.n_classes)), requires_grad=False)
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        mask.scatter_(1, target.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        x = x * mask
        x = x.view(-1, self.n_dim * self.n_classes)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net

    def forward(self, x, target):
        x, probs = self.capsnet(x)
        reconstruction = self.reconstruction_net(x, target)
        return reconstruction, probs


class MarginLoss(nn.Module):
    def __init__(self, m_pos, m_neg, lambda_):
        super(MarginLoss, self).__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(self, lengths, targets, size_average=True):
        t = torch.zeros(lengths.size()).long()
        if targets.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_pos - lengths).pow(2) + \
                 self.lambda_ * (1. - targets.float()) * F.relu(lengths - self.m_neg).pow(2)
        return losses.mean() if size_average else losses.sum()

