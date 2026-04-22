import copy
import json
import os
import random
import shutil
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pymoo.core.termination import NoTermination

from archivers import archive_update_pq
from utils_search import store_metrics, save_architecture, save_archive, plot_hypervolume, plot_hypervolume2, plot_r2, \
    save_statistics_to_csv, save_params, save_archive_losses, plot_archive_losses, store_population_data, load_execution
from worker_process import worker_evaluate_individual

# update your projecty root path before running
if os.path.exists('/Users/rodrigofvc/Documents/doctorado/r2-emoa-rnas/algorithms/nsga-net'):
    sys.path.insert(0, '/Users/rodrigofvc/Documents/doctorado/r2-emoa-rnas/algorithms/nsga-net')
elif os.path.exists('/home/rvelazquez/r2-emoa-rnas/algorithms/nsga-net'):
    sys.path.insert(0, '/home/rvelazquez/r2-emoa-rnas/algorithms/nsga-net')
elif os.path.exists("C:" + os.sep + "Users" + os.sep + "rodri" + os.sep + "Documents" + os.sep + "r2-emoa-rnas" + os.sep + "algorithms" + os.sep + "nsga-net"):
    sys.path.insert(0, "C:" + os.sep + "Users" + os.sep + "rodri" + os.sep + "Documents" + os.sep + "r2-emoa-rnas" + os.sep + "algorithms" + os.sep + "nsga-net")
else:
    raise FileNotFoundError('Project path not found, please update the path in the script before running')

import time
import logging
import argparse
from misc import utils

import numpy as np
import micro_encoding
from search import macro_encoding
from search import nsganet as engine
from pymoo.core.problem import Problem

"""
python -X dev search/evolution_search.py --seed 18906049 --search_space micro --dataset cifar10 --n_classes 10 --init_channels 16 --layers 5 --n_gens 15 --epochs 10 --pop_size 40  --batch_size 192 --n_offspring 40 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --layers 5 --steps 4 --multiplier 4 --cutout_length 16 --drop_path_prob 0.3 --debug_cuda --timestamp 9  --reload_dir search-NSGA-Net-micro-20260419-070500
"""

"""
python3 search/evolution_search.py --seed 18906049 --search_space micro \
--dataset cifar10 --n_classes 10 --init_channels 16 --layers 5 --n_gens 2 --epochs 1 \
--pop_size 10  --batch_size 32 --n_offspring 10 \
--learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
--layers 5 --steps 4 --multiplier 4 --cutout_length 16 --drop_path_prob 0.3 --train_portion 0.5 
"""
parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='NSGA-Net', help='experiment name')
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset name')
parser.add_argument('--n_classes', type=int, choices=[10, 100], help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=6, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=6, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=5, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=16, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=4, help='equivalent with N = 3')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell (intern nodes except input and output)')
parser.add_argument('--multiplier', type=int, default=6, help='number of multiplier for number of channels (intern nodes to concat)')
parser.add_argument('--fgsm_eps', type=float, default=8 / 255, help='attack epsilon')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--timestamp', type=int, default=6, help='timestamp in minutes for training/eval each architecture')
parser.add_argument('--debug_cuda', action='store_true', default=False,
                    help='Enable CUDA_LAUNCH_BLOCKING for debugging')
parser.add_argument('--increase_epochs', action='store_true', default=False,
                    help='Increase the number of epochs to train the supernet and individuals as generations progress')
parser.add_argument('--reload_dir', type=str, default=None,
                    help='Directory to reload the experiment from if --reload is set')
args = parser.parse_args()

if args.reload_dir is not None:
    with open(args.reload_dir + os.sep + 'params.json', 'r') as f:
        args_dict = json.load(f)
        args_dict['reload_dir'] = args.reload_dir
        args = argparse.Namespace(**args_dict)
else:
    args.save = 'search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save)
    save_params(args, args.save)


if os.path.exists("logs"):
    shutil.rmtree("logs")
os.makedirs("logs", exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

pop_hist = []  # keep track of every evaluated architecture


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, dataset, n_classes, search_space='micro', n_var=20, n_obj=4, n_constr=0, lb=None, ub=None,
                 init_channels=16, layers=5, epochs=25, args_problem=None, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr)
        self.xl = lb
        self.xu = ub
        self.dataset = dataset
        self.n_classes = n_classes
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self.save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.statistics = {'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
        self.archive = []
        self.archive_2 = []
        self.args_problem = args_problem

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1

            args_individual = copy.copy(self.args_problem)
            args_individual.seed = args_individual.seed + arch_id
            args_individual.epochs_train_individual = self.args_problem.epochs
            args_individual.gen = -1 # not used in individual worker
            gen = len(self.statistics['hyp_log'])
            performance = worker_evaluate_individual(gen, i, x[i, :].copy(), args_individual)
            objs[i, 0] = performance['std_loss']
            objs[i, 1] = performance['adv_loss']
            objs[i, 2] = performance['flops']
            objs[i, 3] = performance['params']

            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints

# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get("F")
    #store_non_dominated_solutions
    algorithm.problem.archive = archive_update_pq(algorithm.problem.archive, pop_obj)
    algorithm.problem.archive_2 = archive_update_pq(algorithm.problem.archive_2, pop_obj[:, :2])
    hyp, hyp_2, r2 = store_metrics(algorithm.problem.dataset, algorithm.evaluator.n_eval, np.array(algorithm.problem.archive), np.array(algorithm.problem.archive_2), algorithm.problem.save_dir, algorithm.problem.statistics)

    plot_hypervolume(algorithm.problem.statistics, algorithm.problem.save_dir)
    plot_hypervolume2(algorithm.problem.statistics, algorithm.problem.save_dir)
    plot_r2(algorithm.problem.statistics, algorithm.problem.save_dir)

    elapsed_time = time.time() - algorithm.problem.start_time
    algorithm.problem.elapsed_time = elapsed_time
    utils.store_population_data(gen, pop_obj, algorithm.problem.archive, algorithm.problem.archive_2, algorithm.problem.statistics, elapsed_time, algorithm.problem.save_dir)

    # report generation info to files
    logging.info(">>>>>> generation = {}".format(gen))
    logging.info("       hyp_4 = {}, hyp_2 = {} r2 = {}".format(hyp, hyp_2, r2))
    logging.info('       evaluated so far {} architectures'.format(algorithm.evaluator.n_eval))

def main():
    np.random.seed(args.seed)

    logging.info("args = %s", args)

    # setup NAS search problem
    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    else:
        raise NameError('Unknown search space type')
    start = time.time()
    problem = NAS(dataset=args.dataset, n_classes=args.n_classes, n_var=n_var, search_space=args.search_space,
                  n_obj=4, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, args_problem=copy.copy(args), save_dir=args.save)


    # configure the nsga-net method
    algorithm = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    algorithm.setup(problem, seed=args.seed, termination=NoTermination(), verbose=False)

    elapsed_time = 0
    init_generation = 0
    if args.reload_dir is not None:
        logging.info(">>>>>> loading checkpoint from {}".format(args.reload_dir))
        args_execution, statistics, init_generation, n_evaluated, pop_obj, pop_X, archive, archive_2, elapsed_time = load_execution(args.reload_dir)
        algorithm.problem.statistics = statistics
        algorithm.problem._n_evaluated = n_evaluated
        algorithm.problem.archive = archive
        algorithm.problem.archive_2 = archive_2
        algorithm.problem.args_problem = args_execution
        # initialize the population with the loaded one
        pop_initialized = algorithm.ask()
        pop_initialized.set("F", pop_obj)
        pop_initialized.set("X", pop_X)
        algorithm.tell(infills=pop_initialized)

    for gen in range(init_generation, args.n_gens):
        start_time_gen = time.time()
        np.random.seed(args.seed + gen)
        random.seed(args.seed + gen)

        pop = algorithm.ask()
        algorithm.evaluator.eval(problem, pop)
        pop_obj = pop.get("F")
        pop_X = pop.get("X")

        algorithm.problem.archive = archive_update_pq(algorithm.problem.archive, pop_obj)
        algorithm.problem.archive_2 = archive_update_pq(algorithm.problem.archive_2, pop_obj[:, :2])
        hyp, hyp_2, r2 = store_metrics(algorithm.problem.dataset, algorithm.problem._n_evaluated,
                                       np.array(algorithm.problem.archive), np.array(algorithm.problem.archive_2),
                                       algorithm.problem.save_dir, algorithm.problem.statistics)
        plot_hypervolume(algorithm.problem.statistics, algorithm.problem.save_dir)
        plot_hypervolume2(algorithm.problem.statistics, algorithm.problem.save_dir)
        plot_r2(algorithm.problem.statistics, algorithm.problem.save_dir)

        elapsed_time += time.time() - start_time_gen
        algorithm.problem.elapsed_time = elapsed_time
        store_population_data(gen, algorithm.problem._n_evaluated, pop_obj, pop_X, algorithm.problem.archive, algorithm.problem.archive_2,
                                    algorithm.problem.statistics, elapsed_time, algorithm.problem.save_dir)

        # report generation info to files
        logging.info(">>>>>> generation = {}".format(gen))
        logging.info("       hyp_4 = {}, hyp_2 = {} r2 = {}".format(hyp, hyp_2, r2))
        logging.info('       evaluated so far {} architectures'.format(algorithm.problem._n_evaluated))

        algorithm.tell(infills=pop)

    res = algorithm.result()

    args.time_taken = time.time() - start
    print(f">>>> Total search time: ({(time.time() - start) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - start))} (DD:HH:MM:SS)")
    # store non-dominated solutions
    for i, arch in enumerate(res.X):
        genome = micro_encoding.convert(arch) if args.search_space == 'micro' else macro_encoding.convert(arch)
        genotype = micro_encoding.decode(genome) if args.search_space == 'micro' else macro_encoding.decode(genome)
        save_architecture(i, genotype, res.F[i], args.save)
    save_archive(np.array(problem.archive), args.save)
    save_archive_losses(np.array(problem.archive_2), args.save)
    plot_archive_losses(np.array(problem.archive_2), args.save)
    plot_hypervolume(problem.statistics, args.save)
    plot_hypervolume2(problem.statistics, args.save)
    plot_r2(problem.statistics, args.save)
    save_statistics_to_csv(problem.statistics, args.save)
    print('Results stored in {}'.format(args.save))

if __name__ == "__main__":
    main()