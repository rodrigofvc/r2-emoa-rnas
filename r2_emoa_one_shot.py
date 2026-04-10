import numpy as np
import time
import random
import ssl
import logging

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy
from micro_space.model_search import discretize, Network
from individual import Individual
from worker_process import evaluate_population_multiprocessing
from rnas_train import run_batch_epoch
from evolutionary import tournament_selection, binary_crossover, polynomial_mutation, update_population_r2
import torch
from torch import nn
import torchvision

from indicators import update_ref_points

"""
 python3 rnas_search.py --seed 18906049 --algorithm r2-emoa-one-shot --dataset cifar10 --batch_size 96  \
 --n_population 10 --generations 2 --epochs_warmup 10 --epochs_train_supernet 2 \
 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 \
 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
 --report_freq 50 --gpu 0 --init_channels 16 --reduction --layers 5 --steps 4 --multiplier 4 \
 --attack FGSM --cutout --cutout_length 16 --drop_path_prob 0.3 \
 --grad_clip 0.5 --train_portion 0.5
"""
def prepare_args_supernet(args):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    args.device = device

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True

    ssl._create_default_https_context = ssl._create_unverified_context

    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = Network(
        C=args.init_channels,
        num_classes=n_classes,
        layers=args.layers,
        criterion=criterion,
        steps=args.steps,
        multiplier=args.multiplier,
        stem_multiplier=3,
        device=args.device,
    ).to(args.device)

    if args.pretrained_supernet is not None:
        logging.info(f"Loading pretrained supernet from {args.pretrained_supernet}")
        model = utils.load_supernet(args.pretrained_supernet)
        model = model.to(args.device)

    optimizer = torch.optim.Adam(
      model.parameters(),
      args.learning_rate,
      weight_decay=args.weight_decay)

    ssl._create_default_https_context = ssl._create_unverified_context
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96
    logging.info(f"Training samples: {split}, Validation samples: {num_train - split}")

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (args.generations + 1) * args.epochs_train_supernet + args.epochs_warmup, eta_min=args.learning_rate_min)

    weights_r2 = utils.get_weights_r2(args.n_population)

    return model, criterion, optimizer, scheduler, train_queue, valid_queue, weights_r2

def initial_population(n_population, alphas_dim, k, args):
    individuals = []
    for i in range(n_population):
        flattened = np.random.rand(alphas_dim[0] * alphas_dim[1] * 2)
        individuals.append(Individual(X=flattened.copy(), k=k, search_space=args.search_space))
    return individuals

def unpack_alphas(vec, shape_alphas, args):
    n_norm = shape_alphas[0] * shape_alphas[1]
    assert type(vec) == np.ndarray

    a_norm_np = vec[:n_norm].reshape(shape_alphas).copy()
    a_norm = torch.tensor(a_norm_np, dtype=torch.float32, device=args.device).requires_grad_(False)

    a_reduction_np = vec[n_norm:].reshape(shape_alphas).copy()
    a_reduction = torch.tensor(a_reduction_np, dtype=torch.float32, device=args.device).requires_grad_(False)
    return [a_norm, a_reduction]

def train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, gen, args, r2_weights,
                   nadir_point_, ideal_point_, warmup=False):
    model.train()
    if warmup:
        epochs = args.epochs_warmup
    else:
        epochs = args.epochs_train_supernet
    r2_weights_pop = r2_weights[len(pop)]
    r2_weights_pop = torch.tensor(r2_weights_pop, device=args.device, dtype=torch.float32)
    assert r2_weights_pop.shape[0] == len(pop)
    model_flops, model_parameters = utils.get_model_metrics(model)
    model_flops, model_parameters = torch.tensor(float(model_flops), device=args.device), torch.tensor(
        float(model_parameters), device=args.device)
    z_ref_stch = torch.zeros(4, device=args.device, dtype=torch.float32)
    nadir_point = torch.tensor(nadir_point_, device=args.device, dtype=torch.float32)
    ideal_point = torch.tensor(ideal_point_, device=args.device, dtype=torch.float32)
    for epoch in range(epochs):
        for n_batch, (input, target) in enumerate(train_queue):
            individual = pop[n_batch % args.n_population]
            individual_r2_weights = r2_weights_pop[n_batch % args.n_population]
            individual_architect = unpack_alphas(individual.X, model.alphas_dim, args)
            model.update_arch_parameters(individual_architect)
            discrete = discretize(individual_architect, model.genotype(), args.device)
            model.update_arch_parameters(discrete)
            time_stamp = time.time()
            std_acc, adv_acc, loss = run_batch_epoch(model, input, target, criterion, optimizer, args,
                                                     model_flops, model_parameters, individual_r2_weights, z_ref_stch,
                                                     nadir_point, ideal_point)
            if n_batch % args.report_freq == 0:
                logging.info(
                    f'>>>> Gen {gen}/{args.generations} | Epoch {epoch}/{epochs} | Batch {n_batch}/{len(train_queue)} | Loss {loss:.4f} | Std Acc {std_acc:.2f}% | Adv Acc {adv_acc:.2f}% | Time {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_stamp))} (HH:MM:SS)')
        scheduler.step()
    update_ref_points(pop, nadir_point_, ideal_point_)
    utils.save_model(model, args.save_path_final_model, "super-net.pt")

def r2_emoa_oneshot_nas(args):
    model, criterion, optimizer, scheduler, train_queue, valid_queue, weights_r2 = prepare_args_supernet(args)
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    time_search = time.time()
    pop = initial_population(args.n_population, model.alphas_dim, args.objectives, args)
    logging.info(f">>>> Initial population of size {len(pop)} created.")
    nadir_point = np.ones(4, )
    ideal_point = np.zeros(4, )
    if args.epochs_warmup > 0:
        logging.info(">>>> Warmup training of the supernet...")
        train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, 0, args, weights_r2,
                       nadir_point, ideal_point, warmup=True)
        logging.info(">>>> Warmup training DONE.")
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': [],
                  'lr_log': []}
    evaluate_population_multiprocessing(args.n_population,0, pop, weights_r2, nadir_point, ideal_point, args)
    archive = archive_update_pq(archive, pop)
    archive_losses = archive_update_pq(archive_losses, pop, k=2)
    hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                         weights_r2, statistics)
    logging.info(f"Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    for generation in range(args.generations):
        start = time.time()
        time_stamp_epoch = time.time()
        train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, generation + 1, args,
                       weights_r2, nadir_point, ideal_point)
        logging.info(
            f">>>> Gen {generation + 1} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_epoch))} (HH:MM:SS)")

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        offsprings = binary_crossover(parents, n_childs=args.n_population*2, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        # Evaluate offspring
        evaluate_population_multiprocessing(args.n_population, generation, mutation, weights_r2, nadir_point, ideal_point, args)
        update_ref_points(mutation, nadir_point, ideal_point)
        logging.info(
            f"Tiempo total de entrenamiento/validacion {args.generations}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} (HH:MM:SS)")

        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(args.n_population, pop, mutation, weights_r2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                             weights_r2, statistics)
        utils.save_model(model, args.save_path_final_model, f"super-net.pt")
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        statistics['lr_log'].append(optimizer.param_groups[0]['lr'])
        utils.store_statisctics(statistics, np.array([p.F for p in mutation if p.feasible]))
        logging.info(f"Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    logging.info(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return model, archive, archive_accuracy, archive_losses, statistics
