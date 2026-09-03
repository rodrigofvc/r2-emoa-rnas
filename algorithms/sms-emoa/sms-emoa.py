import argparse
import logging
import os
import shutil
import time
from pathlib import Path
import random

import numpy as np
import torch
import torchvision
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from archivers import archive_update_pq
from individual import Individual
from micro_space.micro_encoding import PRIMITIVES, convert, decode
from micro_space.model import NetworkCIFAR
from micro_space.model_search import alphas_to_genotype
from rnas_train import train_individual, infer
from utils import create_experiment_dir, save_architecture, save_archive, save_archive_losses, plot_archive_losses, \
    plot_hypervolume, plot_hypervolume2, plot_r2, save_statistics_to_csv, data_transforms_cifar10, get_model_metrics, \
    get_weights_r2_file, store_metrics, store_population_data, save_params


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class NAS(Problem):
    def __init__(self, dataset, n_var=20, n_obj=4, n_constr=0, lb=None, ub=None,
                 init_channels=16, layers=5, epochs=25, args_problem=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)
        self.xl = lb
        self.xu = ub
        self.dataset = dataset
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.statistics = {'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
        self.archive = []
        self.archive_2 = []
        self.args_problem = args_problem
        self.vtype = int

    def _get_model_from_individual(self, id, individual_X, args):

        if args.dataset == 'cifar10':
            n_classes = 10
        elif args.dataset == 'cifar100':
            n_classes = 100
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        if args.search_space == 'continuous':
            k = sum(2 + i for i in range(args.steps))
            alphas_dim = (k, len(PRIMITIVES))
            genotype = alphas_to_genotype(individual_X, alphas_dim, args)
        else:
            genome = convert(individual_X)
            genotype = decode(genome, args.steps, args.multiplier)
        set_seeds(args.seed + id)
        model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype).to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs_train_individual, eta_min=args.learning_rate_min)
        flops, params = get_model_metrics(model)
        set_seeds(args.seed + id)

        train_transform, valid_transform = data_transforms_cifar10(args)
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
        if torch.backends.mps.is_available():
            # testing
            split = 96
            num_train = split + 96

        split = int(np.floor(args.train_portion * num_train))

        if args.proxy_data_dir is None:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split],
                generator=torch.Generator().manual_seed(args.seed),
            )
            train_queue = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=0, pin_memory=True,
                drop_last=True, generator=torch.Generator().manual_seed(args.seed))
        else:
            logging.info(f"Using proxy data from {args.proxy_data_dir}")
            proxy_indices = np.load(args.proxy_data_dir)
            train_data_proxy = torch.utils.data.Subset(
                train_data,
                proxy_indices.tolist(),
            )
            train_queue = torch.utils.data.DataLoader(
                train_data_proxy, batch_size=args.batch_size,
                num_workers=0, pin_memory=True, drop_last=True,
                generator=torch.Generator().manual_seed(args.seed)
            )

        if args.proxy_eval_dir is None:
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train],
                generator=torch.Generator().manual_seed(args.seed),
            )
            valid_queue = torch.utils.data.DataLoader(
                valid_data, batch_size=args.batch_size,
                sampler=valid_sampler,
                num_workers=0, pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed))
        else:
            logging.info(f"Using proxy evaluation data from {args.proxy_eval_dir}")
            proxy_eval_indices = np.load(args.proxy_eval_dir)
            valid_data_proxy = torch.utils.data.Subset(
                valid_data,
                proxy_eval_indices.tolist(),
            )
            valid_queue = torch.utils.data.DataLoader(
                valid_data_proxy, batch_size=args.batch_size,
                num_workers=0, pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed)
            )
        criterion = torch.nn.CrossEntropyLoss()

        return model, optimizer, scheduler, flops, params, train_queue, valid_queue, criterion

    def _train_eval_monas(self, id, genome, args):
        model, optimizer, scheduler, flops, params, train_queue, valid_queue, criterion = self._get_model_from_individual(id, genome, args)
        set_seeds(args.seed + id)
        feasible = train_individual(model, train_queue, criterion, optimizer, scheduler, args)
        if feasible:
            std_accuracy, adv_accuracy, std_loss, adv_loss = infer(valid_queue, model, criterion, args)
        else:
            std_accuracy, adv_accuracy, std_loss, adv_loss = 0.0, 0.0, 1000, 1000
        if args.search_space == 'continuous':
            k = sum(2 + i for i in range(args.steps))
            alphas_dim = (k, len(PRIMITIVES))
            genotype = alphas_to_genotype(genome, alphas_dim, args)
        else:
            genome = convert(genome)
            genotype = decode(genome, args.steps, args.multiplier)
        performance = {
            'std_loss': std_loss,
            'adv_loss': adv_loss,
            'flops': flops,
            'params': params,
            'std_acc': std_accuracy,
            'adv_acc': adv_accuracy,
            'genotype': genotype
        }
        return performance

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        population = []
        for i in range(x.shape[0]):
            performance = self._train_eval_monas(i, x[i, :], self.args_problem)
            objs[i, 0] = performance['std_loss']
            objs[i, 1] = performance['adv_loss']
            objs[i, 2] = performance['flops']
            objs[i, 3] = performance['params']
            individual = Individual(X=x[i, :].copy(), k=self.n_obj, search_space=self.args_problem.search_space)
            individual.F = objs[i, :].copy()
            individual.genotype = performance['genotype']
            individual.std_acc = performance['std_acc']
            individual.adv_acc = performance['adv_acc']
            if individual.genotype is not None:
                individual.feasible = True
                population.append(individual)
            logging.info(
                f"Individual {self._n_evaluated}: std_acc {performance['std_acc']:.2f}, adv_acc {performance['adv_acc']:.2f} std_loss {performance['std_loss']:.3f}, adv_loss {performance['adv_loss']:.3f}, flops {performance['flops']:.2f}, params {performance['params']:.2f}")
            self._n_evaluated += 1
        self.archive = archive_update_pq(self.archive, population)
        self.archive_2 = archive_update_pq(self.archive_2, population, k=2)
        out["F"] = objs

def sms_emoa_rnas(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')
    print("Running with config:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    n_var = int(4 * args.steps * 2)
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    h = 1
    n_ops = len(PRIMITIVES)
    for b in range(0, n_var // 2, 4):
        ub[b] = n_ops - 1
        ub[b + 1] = h
        ub[b + 2] = n_ops - 1
        ub[b + 3] = h
        h += 1
    ub[n_var // 2:] = ub[:n_var // 2]
    problem = NAS(dataset=args.dataset, n_var=n_var,
                  n_obj=4, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs_train_individual, args_problem=args)
    X = np.column_stack([
        np.random.randint(
            int(lb[j]),
            int(ub[j]) + 1,
            size=args.n_population
        )
        for j in range(n_var)
    ]).astype(np.int32)

    algorithm = SMSEMOA(
        pop_size=args.n_population,
        n_offsprings=args.n_population,
        sampling=X,
        crossover=PointCrossover(n_points=2, prob=args.prob_cross),
        mutation=PolynomialMutation(
            eta=args.eta_mut,
            prob=args.prob_mut,
            vtype=float
        ),
        normalize=True
    )
    algorithm.setup(problem, seed=args.seed, termination=NoTermination(), verbose=False)
    start = time.time()
    r2_weights = get_weights_r2_file(args.r2_weights_dir)
    target_evaluations = args.generations * args.n_population
    next_log = args.n_population
    while algorithm.problem._n_evaluated < target_evaluations:
        current_gen = algorithm.problem._n_evaluated // args.n_population
        if (args.increase_epochs and algorithm.problem._n_evaluated > 0 and
                algorithm.problem._n_evaluated % args.n_population == 0):
            generation_to_start = algorithm.problem._n_evaluated // args.n_population
            if generation_to_start % 10 == 0:
                algorithm.problem.args_problem.epochs_train_individual += 5

        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        algorithm.tell(infills=pop)
        if algorithm.problem._n_evaluated >= next_log:
            next_log += args.n_population
            elapsed_time = time.time() - start
            pop_obj = algorithm.pop.get("F")
            pop_X = algorithm.pop.get("X")
            hyp, hyp_2, r2 = store_metrics(algorithm.problem._n_evaluated,
                                           algorithm.problem.archive, algorithm.problem.archive_2,
                                           args, r2_weights, algorithm.problem.statistics)
            store_population_data(current_gen, pop_X, pop_obj, algorithm.problem.archive,
                                  algorithm.problem.archive_2,
                                  algorithm.problem.statistics, elapsed_time, args.save_path_final_architect)
            plot_hypervolume(algorithm.problem.statistics, args.save_path_final_architect)
            plot_hypervolume2(algorithm.problem.statistics, args.save_path_final_architect)
            plot_r2(algorithm.problem.statistics, args.save_path_final_architect)
            save_archive_losses(problem.archive_2, args.save_path_final_architect)
            plot_archive_losses(problem.archive_2, args.save_path_final_architect)
            algorithm.problem.elapsed_time = elapsed_time

            # report generation info to files
            logging.info(">>>>>> generation = {}".format(current_gen))
            logging.info("       hyp_4 = {}, hyp_2 = {} r2 = {}".format(hyp, hyp_2, r2))
            logging.info('       evaluated so far {} architectures'.format(algorithm.problem._n_evaluated))



    res = algorithm.result()
    args.time_taken = time.time() - start
    print(f">>>> Total search time: ({(time.time() - start) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - start))} (DD:HH:MM:SS)")
    # store non-dominated solutions
    n_classes = 10 if args.dataset == 'cifar10' else 100
    for i, arch in enumerate(res.X):
        genome = convert(arch)
        genotype = decode(genome, args.steps, args.multiplier)
        # check architecture
        model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype).to(args.device)
        save_architecture(i, genotype, args.save_path_final_architect)
    save_archive(problem.archive, args.save_path_final_architect)
    save_archive_losses(problem.archive_2, args.save_path_final_architect)
    plot_archive_losses(problem.archive_2, args.save_path_final_architect)
    plot_hypervolume(problem.statistics, args.save_path_final_architect)
    plot_hypervolume2(problem.statistics, args.save_path_final_architect)
    plot_r2(problem.statistics, args.save_path_final_architect)
    save_statistics_to_csv(problem.statistics, args.save_path_final_architect)
    save_params(args, args.save_path_final_architect)
    print('Results stored in {}'.format(args.save_path_final_architect))
    return problem.archive, problem.archive_2, problem.statistics

"""
# python3 sms-emoa.py --seed 18906049 --dataset cifar10 --batch_size 32 --n_population 10 \
--generations 2 --epochs_train_individual 1 \
--data ../../data --num_workers 0 \
--prob_cross 0.9 --prob_mut 0.1 --eta_mut 20 --loss_type ws --mu 0.1 --lambda_1 0.5 \
--lambda_2 0.5 --learning_rate 0.025 --learning_rate_min 0.001 \
--momentum 0.9 --weight_decay 3e-4 --report_freq 45 --gpu 0 --init_channels 8 \
--reduction --layers 5 --steps 4 --multiplier 4 --attack FGSM \
--cutout_length 16 --drop_path_prob 0.3 --grad_clip 5.0 --increase_epochs
"""
# python sms-emoa.py --seed 18906049 --dataset cifar10 --batch_size 192 --n_population 40 --generations 30 --epochs_train_individual 10 --data ../../data --num_workers 0 --prob_cross 0.9 --prob_mut 0.1 --eta_mut 3 --loss_type ws --lambda_1 0.5 --lambda_2 0.5 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --report_freq 45 --gpu 0 --init_channels 8 --reduction --layers 5 --steps 4 --multiplier 4 --attack FGSM --grad_clip 5.0 --increase_epochs --proxy_data_dir proxy-data/cifar10_train_25000.npy --proxy_eval_dir proxy-data/cifar10_eval_25000.npy --initial_population initial/initial_population_40.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running SMS-EMOA for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--search_space', type=str, default="discrete", choices=['continuous', 'discrete'], help='search space to use')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_population', type=int, default=40, help='population size')
    parser.add_argument('--generations', type=int, default=30, help='number of generations to search')
    parser.add_argument('--epochs_train_individual', type=int, default=1, help='number of epochs to train individual per generation')
    parser.add_argument('--objectives', type=int, default=4, help='number of objectives')
    parser.add_argument('--std_loss_index', type=int, default=0, help='index of standard loss in objectives')
    parser.add_argument('--adv_loss_index', type=int, default=1, help='index of adversarial loss in objectives')
    parser.add_argument('--flops_index', type=int, default=2, help='index of flops in objectives')
    parser.add_argument('--params_index', type=int, default=3, help='index of params in objectives')
    parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--prob_cross', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--prob_mut', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--eta_mut', type=int, default=20, help='mutation eta')
    parser.add_argument('--loss_type', type=str, default='ws', choices=['tchebycheff', 'ws'], help='type of loss function to use for backpropagation')
    parser.add_argument('--mu', type=float, default=0.1, help='mu for thchebycheff function')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='weight for standard loss in ws scalarization')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='weight for adversarial loss in ws scalarization')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='init channels')
    parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers (cells)')
    parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell (intern nodes except input and output)')
    parser.add_argument('--multiplier', type=int, default=6, help='number of multiplier for number of channels (intern nodes to concat)')
    parser.add_argument('--attack', type=str, default='FGSM', help='adversarial attack to use')
    parser.add_argument('--attack_eps', type=float, default=8 / 255, help='attack epsilon')
    parser.add_argument('--attack_alpha', type=float, default=10 / 255, help='attack alpha for PGD or FGSM with random start')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--r2_weights_dir', type=str, default='r2_weights/weights_40.json', help='directory to store r2 weights')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--timestamp_individual', type=int, default=7, help='timestamp in minutes for training/eval each architecture')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='Enable CUDA_LAUNCH_BLOCKING for debugging')
    parser.add_argument('--increase_epochs', action='store_true', default=False, help='Increase the number of epochs to train the supernet and individuals as generations progress')
    parser.add_argument('--losses_objs', action='store_true', default=False, help='Use the standard and adversarial losses as objectives instead of using accuracies as objectives')
    parser.add_argument('--reload_dir', type=str, default=None, help='Directory to reload the experiment from if --reload is set')
    parser.add_argument('--proxy_data_dir', type=str, default=None, help='Directory to load the proxy data indices (if provided)')
    parser.add_argument('--proxy_eval_dir', type=str, default=None, help='Directory to load the proxy evaluation data indices (if provided)')
    parser.add_argument('--initial_population', type=str, default=None, help='Path to the initial population file (if provided)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    if args.reload_dir is None:
        results_dir = create_experiment_dir('sms-emoa', args.dataset, args.seed)
    elif args.reload_dir == 'auto-last':
        # reload the last experiment in the results directory for the given algorithm and dataset
        base_dir = Path("results") / args.algorithm / args.dataset
        # base_dir = Path(".")

        if not base_dir.exists():
            raise ValueError("No experiments found for the given algorithm and dataset")

        dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        if not dirs:
            raise ValueError("No experiments found for the given algorithm and dataset")

        latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)

        results_dir = str(latest_dir) + os.sep + "search"
        args.reload_dir = results_dir
    else:
        results_dir = args.reload_dir
    print(f'Results dir: {results_dir}')
    args.save_path_final_model = results_dir
    args.save_path_final_architect = results_dir

    archive, archive_losses, statistics = sms_emoa_rnas(args)
    for i, individual in enumerate(archive):
        save_architecture(i, individual, args.save_path_final_architect)
    save_archive(archive, args.save_path_final_architect)
    save_archive_losses(archive_losses, args.save_path_final_architect)
    plot_archive_losses(archive_losses, args.save_path_final_architect)
    plot_hypervolume(statistics, args.save_path_final_architect)
    plot_hypervolume2(statistics, args.save_path_final_architect)
    plot_r2(statistics, args.save_path_final_architect)
    save_statistics_to_csv(statistics, args.save_path_final_architect)
    logging.info(f"Experiment completed and results saved in {results_dir}")