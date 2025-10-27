import torch

from rnas import get_attack_function
from train_search import run_batch_epoch, infer, discretize
from model_search import Network
import torch.nn as nn
import torchvision
import argparse
import ssl
import time
import utils


if __name__ == '__main__':

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print("Using device:", DEVICE)

    nsga_net_setup = True

    parser = argparse.ArgumentParser("CIFAR")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
    parser.add_argument('--classes', type=int, default=-1, help='number of classes')
    parser.add_argument('--device', type=str, default=DEVICE, help='device to use for training')
    parser.add_argument('--objectives', type=int, default=4, help='number of objectives to optimize')
    parser.add_argument('--std_loss_index', type=int, default=0, help='index of the standard loss in the objectives')
    parser.add_argument('--adv_loss_index', type=int, default=1, help='index of the adversarial loss in the objectives')
    parser.add_argument('--flops_index', type=int, default=2, help='index of the FLOPs in the objectives')
    parser.add_argument('--params_index', type=int, default=3, help='index of the number of parameters in the objectives')
    parser.add_argument('--data', type=str, default='../data', help='location of the data')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='weight for std loss')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='weight for adv loss')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--n_population', type=int, default=25, help='number of individuals in the population')
    parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')
    if nsga_net_setup:
        parser.add_argument('--reduction', default=False, help='set reduction layers at 1/3 and 2/3 of the network')
        # NSGA-Net (6 min x indv)
        parser.add_argument('--layers', type=int, default=3, help='total number of cells')
        parser.add_argument('--steps', type=int, default=5, help='total number of intern nodes per cell')
        parser.add_argument('--multiplier', type=int, default=5, help='total number of concat nodes per cell')
    else:
        # NEvoNAS (10 min x indv)
        parser.add_argument('--reduction', default=True, help='set reduction layers at 1/3 and 2/3 of the network')
        parser.add_argument('--layers', type=int, default=8, help='total number of cells')
        parser.add_argument('--steps', type=int, default=4, help='total number of intern nodes per cell')
        parser.add_argument('--multiplier', type=int, default=4, help='total number of concat nodes per cell')

    parser.add_argument('--attack', type=dict, default=None, help='adversarial attack parameters')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    args = parser.parse_args([])

    torch.manual_seed(args.seed)

    args.attack = {'name': 'FGSM', 'params': {'eps': 0.007}}

    attack_params = {'name': 'FGSM', 'params': {'eps': 0.007}}

    if args.reduction:
        reduction_layers = [args.layers//3, 2*args.layers//3]
    else:
        reduction_layers = None

    criterion = nn.CrossEntropyLoss()

    model = Network(
        C=args.init_channels,
        num_classes=args.classes,
        layers=args.layers,
        criterion=criterion,
        steps=args.steps,
        multiplier_cells=args.multiplier,
        reduction_layers=reduction_layers,
        stem_multiplier=3,
        fairdarts_eval=False,
        device=args.device,
    ).to(args.device)


    optimizer = torch.optim.SGD(
      model.weight_parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    #split = int(np.floor(args.train_portion * num_train))
    split = 100

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=2, pin_memory=True)

    valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=2, pin_memory=True)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    attack_f = get_attack_function(attack_params)

    start = time.time()

    individuals = []
    for i in range(args.n_population):
        normal_arch = torch.rand_like(model.arch_parameters()[0]).to(args.device).requires_grad_(False)
        reduction_arch = torch.rand_like(model.arch_parameters()[1]).to(args.device).requires_grad_(False)
        individuals.append([normal_arch, reduction_arch])

    for epoch in range(args.epochs):
        model.train()
        time_stamp_epoch = time.time()
        for n_batch, (input, target) in enumerate(train_queue):
            individual = individuals[epoch % args.n_population]
            time_stamp = time.time()
            std_acc, adv_acc, loss = run_batch_epoch(model, individual, input, target, criterion, optimizer, attack_f, args)
            print(f">>>> Epoch {epoch+1}/{args.epochs} Batch {n_batch+1}/{len(train_queue)} ({(time.time() - time_stamp):.4f}) seg : std_acc {std_acc/args.batch_size*100:.2f}%, adv_acc {adv_acc/args.batch_size*100:.2f}%, loss {loss:.4f}")
        scheduler.step()
        print(f">>>> Epoch {epoch+1} training DONE in {(time.time() - time_stamp_epoch):.4f} seg")

        model.eval()
        for i, individual in enumerate(individuals):
            model.update_arch_parameters(individual)
            discrete = discretize(individual, model.genotype(), DEVICE)
            model.update_arch_parameters(discrete)
            time_stamp = time.time()
            std_acc, adv_acc, loss = infer(valid_queue, model, criterion, attack_f, args)
            print(f"Evaluation {i+1}/{len(individuals)}: std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, loss {loss:.4f} ({(time.time() - time_stamp):.4f} seg)")

    print(f"Tiempo total de entrenamiento/validacion {args.epochs}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} horas")