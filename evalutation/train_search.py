import argparse
import ssl
import time

import torch
import torch.nn.functional as F
import torchattacks
import torchvision
from torch import nn

import utils
from architect import Architect
from model_search import Network


def get_attack_function(attack_params):
    if attack_params['name'] == 'FGSM':
        attack_function = lambda model: torchattacks.FGSM(model, **attack_params['params'])
    elif attack_params['name'] == 'PGD':
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, attack_f, device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  std_correct = 0
  adv_correct = 0
  total = 0
  model.train()
  valid_iterator = iter(valid_queue)
  for step, (input, target) in enumerate(train_queue):
      timesstamp = time.time()
      n = input.size(0)
      prepare_time = time.time()
      input = input.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)


      input_search, target_search = next(valid_iterator)
      input_search = input_search.to(device, non_blocking=True)
      target_search = target_search.to(device, non_blocking=True)

      print('data prep DONE in %.3f seconds' % (time.time() - prepare_time))
      architect_time = time.time()
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
      print('architect step DONE in %.3f seconds' % (time.time() - architect_time))

      weights_time = time.time()
      optimizer.zero_grad()

      attack = attack_f(model)
      adv_X = attack(input, target)
      logits_adv = model(adv_X)
      adv_loss = criterion(logits_adv, target)

      logits = model(input)
      natural_loss = criterion(logits, target)

      total_loss = architect.lambda_1 * natural_loss + architect.lambda_2 * adv_loss

      total_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      print('weights step DONE in %.3f seconds' % (time.time() - weights_time))

      std_predicts = logits.argmax(dim=1)
      adv_predicts = logits_adv.argmax(dim=1)
      std_correct += (std_predicts == target).sum().item()
      adv_correct += (adv_predicts == target).sum().item()
      total += target.size(0)
      std_accuracy = std_correct / total
      adv_accuracy = adv_correct / total

      # if step % args.report_freq == 0:
      print ('-- train step %d/%d loss_ws %.5f std_acc %.3f adv_acc %.3f %f segs' % (step+1, len(train_queue), objs.avg, std_accuracy, adv_accuracy, time.time() - timesstamp))

  return top1.avg, objs.avg

def infer(valid_queue, model, attack_f, device):
    model.eval()
    meter_loss, meter_top1 = 0.0, 0.0
    n = 0
    std_correct = 0
    adv_correct = 0
    total = 0
    for step, (input, target) in enumerate(valid_queue):
        input  = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        std_logits = model(input)
        std_loss = F.cross_entropy(std_logits, target)
        adv_input = attack_f(model)(input, target).to(device, non_blocking=True)
        adv_logits = model(adv_input)
        adv_loss = F.cross_entropy(adv_logits, target)
        total_loss = 0.5 * (std_loss + adv_loss)

        n += input.size(0)
        meter_loss += total_loss.item() * input.size(0)
        meter_top1 += (std_logits.argmax(1) == target).float().sum().item()

        std_predicts = std_logits.argmax(dim=1)
        adv_predicts = adv_logits.argmax(dim=1)
        std_correct += (std_predicts == target).sum().item()
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)
        std_accuracy = std_correct / total
        adv_accuracy = adv_correct / total
        print('infer step %d loss_ws %.5f std_acc %.3f adv_acc %.3f' % (step, total_loss.item(), std_accuracy, adv_accuracy))
    return meter_top1 / n * 100.0, meter_loss / n


if __name__ == '__main__':
    torch.manual_seed(0)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print("Using device:", DEVICE)

    parser = argparse.ArgumentParser("CIFAR")
    parser.add_argument('--data', type=str, default='../data', help='location of the data')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='weight for std loss')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='weight for adv loss')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of cells')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args()

    attack_params = {'name': 'FGSM', 'params': {'eps': 0.007}}

    # Puedes pasar listas por celda (coincide con tu Network actual)
    steps_cells = [3, 4, 4, 4, 4, 4, 4, 4]
    multiplier_cells = [3, 4, 4, 4, 4, 4, 4, 4]

    # Las celdas donde se aplican las celdas de reducción
    reduction_layers = {args.layers // 3, (2 * args.layers) // 3}

    # Grafos por celda (0=s0, 1=s1, 2.. internos).
    g0 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 4],  # x0 -> x1, x2
        3: [4],  # x1 -> x2
    }

    g1 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g2 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g3 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g4 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g5 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g6 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    g7 = {
        0: [2, 3],  # s0 -> x0, x1
        1: [2, 4],  # s1 -> x0, x2
        2: [3, 5],  # x0 -> x1, x3
        3: [4, 5],  # x1 -> x2, x3
        4: [5],  # x2 -> x3
    }

    cells_edges = [g0, g1, g2, g3, g4, g5, g6, g7]
    CIFAR_CLASSES = 10
    criterion = nn.CrossEntropyLoss()

    model = Network(
        C=args.init_channels,
        num_classes=CIFAR_CLASSES,
        layers=args.layers,
        criterion=criterion,
        steps_cells=steps_cells,
        multiplier_cells=multiplier_cells,
        cells_edges=cells_edges,
        reduction_layers=reduction_layers,
        stem_multiplier=3,
    ).to(DEVICE)


    optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    #split = int(np.floor(args.train_portion * num_train))
    split = 100
    #split = 10000 // 100

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=2, pin_memory=True)


    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=2, pin_memory=True)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    attack_f = get_attack_function(attack_params)
    architect = Architect(model, args, lambda_1=args.lambda_1, lambda_2=args.lambda_2,
                          criterion=criterion, attack_f=attack_f, device=DEVICE)
    start = time.time()
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]

        print(f">>>> Epoch {epoch+1}/{args.epochs}")

        time_train = time.time()
        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, attack_f=attack_f, device=DEVICE)
        print ('train_acc %f', train_acc)
        print(f"Tiempo de entrenamiento epoca {epoch}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_train))}")

        time_valid = time.time()
        # validation
        valid_acc, valid_obj = infer(valid_queue, model, attack_f=attack_f, device=DEVICE)
        print(f"Tiempo de validacion epoca {epoch}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_valid))}")
        scheduler.step()
        #utils.save(model, os.path.join(args.save, 'weights.pt'))
    print(f"Tiempo total de entrenamiento/validacion: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} horas")

