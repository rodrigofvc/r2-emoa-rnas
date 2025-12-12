import sys

import torchvision
from thop import profile

# update your projecty root path before running
sys.path.insert(0, '/path/to/nsga-net')

import numpy as np
import torch

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.macro_models import EvoNetwork
from models.micro_models import NetworkCIFAR as Network

from adversarial import get_attack_function

import time
from misc import utils
from search import micro_encoding
from search import macro_encoding


if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'


def main(genome, epochs, search_space='micro',
         save='Design_1', expr_root='search', seed=0, gpu=0, init_channels=16,
         layers=11, auxiliary=False, cutout=False, drop_path_prob=0.0):

    # ---- train logger ----------------- #
    #save_pth = os.path.join(expr_root, '{}'.format(save))
    #utils.create_exp_dir(save_pth)
    #log_format = '%(asctime)s %(message)s'
    #logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    #                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    # fh = logging.FileHandler(os.path.join(save_pth, 'log.txt'))
    # fh.setFormatter(logging.Formatter(log_format))
    # logging.getLogger().addHandler(fh)

    # ---- parameter values setting ----- #
    CIFAR_CLASSES = 10
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    data_root = '../../../data'
    batch_size = 96
    cutout_length = 16
    auxiliary_weight = 0.4
    grad_clip = 5
    report_freq = 50
    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
        'lambda_1': 0.5,
        'lambda_2': 0.5,
    }

    if search_space == 'micro':
        genotype = micro_encoding.decode(genome)
        model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
    elif search_space == 'macro':
        genotype = macro_encoding.decode(genome)
        channels = [(3, init_channels),
                    (init_channels, 2*init_channels),
                    (2*init_channels, 4*init_channels)]
        model = EvoNetwork(genotype, channels, CIFAR_CLASSES, (32, 32), decoder='residual')
    else:
        raise NameError('Unknown search space type')

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if cutout:
        train_transform.transforms.append(utils.Cutout(cutout_length))


    train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)

    train_portion = 0.5
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96
    #print(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True)
    #print(f"Train batches: {len(train_queue)}")

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))

    attack_params = {
        'name': 'FGSM',
        'params': {
            'eps': '8/255',
        }
    }

    attack_f = get_attack_function(attack_params)

    start = time.time()
    for epoch in range(epochs):
        train(train_queue, model, criterion, optimizer, attack_f, train_params)
        scheduler.step()
    print('Training time: {} seconds'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start))))

    start = time.time()
    valid_acc, valid_std_loss, valid_adv_loss = infer(valid_queue, model, criterion, attack_f, train_params)
    print('Inference time: {} seconds'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start))))
    #logging.info('valid_acc %f', valid_acc)

    # calculate for flops
    model.eval()
    x = torch.randn(1, 3, 32, 32).to(device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = (2 * macs) / 1e6
    params = params / 1e6
    model_flops = round(flops, 4)
    model_parameters = round(params, 4)
    return {
        'valid_acc': valid_acc,
        'std_error': valid_std_loss,
        'adv_error': valid_adv_loss,
        'params': model_parameters,
        'flops': model_flops,
    }

# Training
def train(train_queue, net, criterion, optimizer, attack_f, params):
    net.train()
    std_loss = 0
    adv_loss = 0
    correct = 0
    total = 0
    attack = attack_f(net)
    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        adv_X, std_logits = attack(inputs, targets)
        adv_logits = net(adv_X)[0]
        std_loss = criterion(std_logits, targets)
        adv_loss = criterion(adv_logits, targets)
        total_loss = params['lambda_1'] * std_loss + params['lambda_2'] * adv_loss

        total_loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        std_loss += std_loss.item()
        adv_loss += adv_loss.item()
        _, predicted = std_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100*correct/total, std_loss/total, adv_loss/total

def infer(valid_queue, net, criterion, attack_f, params):
    net.eval()
    std_loss = 0
    adv_loss = 0
    correct = 0
    total = 0
    attack = attack_f(net)
    for step, (inputs, targets) in enumerate(valid_queue):
        inputs, targets = inputs.to(device), targets.to(device)

        adv_inputs, std_logits = attack(inputs, targets)
        outputs_adv, _ = net(adv_inputs)
        std_loss = criterion(std_logits, targets)
        adv_loss = criterion(outputs_adv, targets)

        std_loss += std_loss.item()
        adv_loss += adv_loss.item()
        _, predicted = std_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total

    return acc, std_loss/total, adv_loss/total


if __name__ == "__main__":
    DARTS_V2 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]], [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
               [[[0, 0], [0, 1]], [[2, 2], [0, 1]], [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    start = time.time()
    print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_16', seed=1, init_channels=16,
               auxiliary=False, cutout=False, drop_path_prob=0.0))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
    # start = time.time()
    # print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_32', seed=1, init_channels=32))
    # print('Time elapsed = {} mins'.format((time.time() - start) / 60))

