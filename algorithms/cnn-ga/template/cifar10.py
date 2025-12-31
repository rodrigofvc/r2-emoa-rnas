"""
from __future__ import print_function

import lzma
import pickle

import numpy as np
import torch
from thop import profile
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import data_loader
import os
from datetime import datetime
import multiprocessing

from adversarial import get_attack_function
from utils import StatusUpdateTool

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #self.stem = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        #generated_init


    def forward(self, x):
        #x = self.stem(x)
        #generate_forward

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TrainModel(object):
    def __init__(self):
        data_dir = os.path.expanduser('../../../data')
        trainloader, validate_loader = data_loader.get_train_valid_loader(data_dir, random_seed=18906049, augment=False,batch_size=96, num_workers=0, pin_memory=False)
        net = EvoCNNModel()
        if torch.cuda.is_available():
            cudnn.benchmark = True
            net = net.cuda()
        criterion = nn.CrossEntropyLoss()
        self.criterion = criterion
        # objectives (std_loss, adv_loss, flops, params)
        self.F = np.zeros(4,)
        self.trainloader = trainloader
        self.validate_loader = validate_loader
        self.file_id = os.path.basename(__file__).split('.')[0]
        attack_params = {
            'name': 'FGSM',
            'params': {
                'eps': '8/255',
            }
        }
        self.attack_f = get_attack_function(attack_params)
        self.lambda_1 = 0.5
        self.lambda_2 = 0.5
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        net = net.to(self.device)
        self.net = net

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch):
        self.net.train()
        if epoch ==0: lr = 0.01
        if epoch > 0: lr = 0.1;
        if epoch > 148: lr = 0.01
        if epoch > 248: lr = 0.001
        optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum = 0.9, weight_decay=5e-4)
        total = 0
        std_correct = 0
        adv_correct = 0
        std_loss_mean = 0
        adv_loss_mean = 0
        total_loss_mean = 0
        attack = self.attack_f(self.net)
        for _, (inputs, labels) in enumerate(self.trainloader, 0):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            adv_inputs, std_logits = attack(inputs, labels)
            adv_logits = self.net(adv_inputs)
            std_loss = self.criterion(std_logits, labels)
            adv_loss = self.criterion(adv_logits, labels)
            total_loss = self.lambda_1 * std_loss + self.lambda_2 * adv_loss
            total_loss.backward()
            optimizer.step()

            std_predicts = std_logits.argmax(dim=1)
            adv_predicts = adv_logits.argmax(dim=1)
            std_correct += (std_predicts == labels).sum().item()
            adv_correct += (adv_predicts == labels).sum().item()
            total += labels.size(0)
            std_loss_mean += std_loss.item()
            adv_loss_mean += adv_loss.item()
            total_loss_mean += total_loss.item()
            #print('Training Epoch:%d, Batch:%d/%d'% (epoch+1, _+1, len(self.trainloader)), end='\r')
        self.log_record('Train-Epoch:%3d,  Std_Acc: %.3f, Adv_Acc: %.3f, Std_Loss: %.3f, Adv_Loss: %.3f, Total_Loss: %.3f'% (epoch+1, std_correct/total, adv_correct/total, std_loss_mean/total, adv_loss_mean/total, total_loss_mean/total))

    def test(self, epoch):
        self.net.eval()
        test_loss = 0.0
        correct = 0
        std_correct = 0
        adv_correct = 0
        std_loss_mean = 0
        adv_loss_mean = 0
        total_loss_mean = 0
        total = 0
        attack = self.attack_f(self.net)
        for _, (inputs, labels) in enumerate(self.validate_loader, 0):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            adv_inputs, std_logits = attack(inputs, labels)
            adv_logits = self.net(adv_inputs)
            std_loss = self.criterion(std_logits, labels)
            adv_loss = self.criterion(adv_logits, labels)
            total_loss = self.lambda_1 * std_loss + self.lambda_2 * adv_loss

            std_predicts = std_logits.argmax(dim=1)
            adv_predicts = adv_logits.argmax(dim=1)
            std_correct += (std_predicts == labels).sum().item()
            adv_correct += (adv_predicts == labels).sum().item()
            total += labels.size(0)
            std_loss_mean += std_loss.item()
            adv_loss_mean += adv_loss.item()
            total_loss_mean += total_loss.item()

        self.F[0] = std_loss_mean / total
        self.F[1] = adv_loss_mean / total

        x = torch.randn(1, 3, 32, 32).to(self.device)
        macs, params = profile(self.net, inputs=(x,), verbose=False)
        flops = (2 * macs) / 1e6
        params = params / 1e6

        self.F[2] = round(flops, 4)
        self.F[3] = round(params, 4)
        self.log_record('Validate-Epoch:%3d,  Std_Acc: %.3f, Adv_Acc: %.3f, Std_Loss: %.3f, Adv_Loss: %.3f, Total_Loss: %.3f, Flops: %.4f, Params: %.4f'% (epoch+1, std_correct/total, adv_correct/total, std_loss_mean/total, adv_loss_mean/total, total_loss_mean/total, round(flops, 4), round(params, 4)))


    def process(self):
        total_epoch = StatusUpdateTool.get_epoch_size()
        for p in range(total_epoch):
            self.train(p)
            self.test(p)
        return self.F

class StoreModel(object):
    def save_architect(self, i, objectives, save_dir):
        model = EvoCNNModel().to('cpu')
        architect_path = save_dir + os.sep + 'architectures' + os.sep
        if not os.path.exists(architect_path):
            os.makedirs(architect_path)
        architect_path += f'arch_{i}.xz'
        with lzma.open(architect_path, 'wb') as f:
            pickle.dump((model, objectives), f)


class RunModel(object):
    def do_work(self, gpu_id, file_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        F = np.zeros(4)
        m = TrainModel()
        process_failed = False
        try:
            m.log_record('Used GPU#%s, worker name:%s[%d]'%(gpu_id, multiprocessing.current_process().name, os.getpid()), first_time=True)
            F = m.process()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
            process_failed = True
        finally:
            # only store the fitness when the process is successful
            if not process_failed:
                m.log_record('Objectives: Std_Loss: %.5f, Adv_Loss: %.5f, Flops: %.4f, Params: %.4f'% (F[0], F[1], F[2], F[3]))
                f = open('./populations/after_%s.txt'%(file_id[4:6]), 'a+')
                f.write('%s=%s\n'%(file_id, np.array_str(F)))
                f.flush()
                f.close()
"""
