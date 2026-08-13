import os
import shutil
import sys
from pathlib import Path

from pymoo.operators.sampling.rnd import FloatRandomSampling

from archivers import archive_update_pq
from individual import Individual
from micro_space.micro_encoding import PRIMITIVES
from utils_search import save_archive, save_archive_2, plot_archive_losses, plot_hypervolume, plot_hypervolume2, \
  plot_r2, save_statistics_to_csv, save_params, save_architecture

sys.path.insert(0, './s1')

import argparse
import logging
import random
import numpy as np
import time
import ut as utils

import utils_search
import copy
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2            import NSGA2
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx         import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm           import PolynomialMutation
from worker_process import train_supernet, worker_evaluate_individual

"""
python3 -X dev arch_search.py --seed 18906049 --dataset cifar10 --batch_size 32 --gpu 0 \
--init_channels 16 --generations 4 --n_population 10 \
--learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
--mutate_rate 0.1 --report_freq 50 --layers 5 --steps 4 --multiplier 4 --reduction \
--grad_clip 5 --train_portion 0.5 --epochs_warmup 2 --epochs_train_supernet 1 \
--timestamp_supernet 45 --timestamp_individual 5 
"""

parser = argparse.ArgumentParser("S1")
parser.add_argument('--batch_size',        type=int, default = 96, help = 'batch size')
parser.add_argument('--cutout',            action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length',     type = int, default = 16, help = 'cutout length')
parser.add_argument('--data',          type = str, default = '../../../data', help = 'location of the data corpus')
parser.add_argument('--dataset',           type = str, default = '', help = '["cifar10", "cifar100"]')
parser.add_argument('--generations',            type = int, default = 30, help = 'num of generations')
parser.add_argument('--gpu',               type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip',         type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels',     type = int, default = 16, help = 'num of init channels')
parser.add_argument('--knn',               type = int, default = 5, help = 'k-nearest neighbors')
parser.add_argument('--layers',            type = int, default = 5, help = 'total number of layers')
parser.add_argument('--lambda_1',         type = float, default = 0.5, help = 'weight for std loss')
parser.add_argument('--lambda_2',         type = float, default = 0.5, help = 'weight for adv loss')
parser.add_argument('--steps',             type = int, default = 6, help = 'number of steps in one cell')
parser.add_argument('--multiplier',        type = int, default = 6, help = 'multiplier for number of channels')
parser.add_argument('--learning_rate',     type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--momentum',          type = float, default = 0.9, help = 'momentum')
parser.add_argument('--mutate_rate',       type = float, default = 0.1, help = 'mutation rate')
parser.add_argument('--attack_eps', type=float, default=8 / 255, help='attack epsilon')
#parser.add_argument('--output_dir',        type = str, default = None, help = 'location of trials')
parser.add_argument('--n_population',          type = int, default = 40, help = 'population size')
parser.add_argument('--report_freq',       type = float, default = 50, help = 'report frequency')
parser.add_argument('--seed',              type = int, default = 18906049, help = 'random seed')
parser.add_argument('--train_portion',      type = float, default = 0.5, help = 'split option for CIFAR100')
#parser.add_argument('--train_discrete',    default=False, action='store_true')
#parser.add_argument('--valid_batch_size',  type = int, default = 64, help = 'validation batch size')
parser.add_argument('--weight_decay',      type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--epochs_warmup',     type = int, default = 0, help = 'number of epochs to warmup the supernet before starting the search process')
parser.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
#parser.add_argument('--workers',           type=int, default=0, help='number of data loading workers (default: 2)')
parser.add_argument('--pretrained_supernet', type=str, default=None, help='path to pretrained supernet to load before training')
parser.add_argument('--save_path_final_model', type=str, default=None, help='path to save the final supernet model after search')
parser.add_argument('--timestamp_supernet', type=int, default=45, help='timestamp in minutes for training the supernet')
parser.add_argument('--timestamp_individual', type=int, default=5, help='timestamp in minutes for evaluating each architecture')
parser.add_argument('--search_space', type=str, default="continuous", choices=['continuous', 'discrete'], help='search space to use')
parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
parser.add_argument('--debug_cuda', action='store_true', default=False,
                    help='Enable CUDA_LAUNCH_BLOCKING for debugging')
parser.add_argument('--increase_epochs', action='store_true', default=False,
                    help='Increase the number of epochs to train the supernet and individuals as generations progress')
parser.add_argument('--reload_dir', type=str, default=None,
                    help='Directory to reload the experiment from if --reload is set')

args = parser.parse_args()


def initial_population(n_population, alphas_dim):
  individuals = []
  for i in range(n_population):
    flattened = np.random.rand(alphas_dim[0] * alphas_dim[1] * 2)
    individuals.append(flattened)
  individuals = np.array(individuals)
  return individuals

def prepare_args_supernet(args_):
  if args_.reload_dir is not None:
    # the execution is a reload and we need to load all the variables from the previous execution
    args, statistics, initial_generation, n_evaluated, pop_obj, pop_X, archive, archive_losses, time_search = utils_search.load_execution(
      args_.reload_dir)
    architectures_evaluated = len(statistics['hyp_log']) * args.n_population
    logging.info(
      f">>>> Reloading execution from {args_.reload_dir} at generation {initial_generation} with {architectures_evaluated} architectures already evaluated.")
  else:
    # the execution is new and we need to initialize all the variables
    args = args_
    initial_generation = 0
    archive = []
    archive_losses = []
    architectures_evaluated = 0
    np.random.seed(args.seed)
    random.seed(args.seed)
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'),
                  'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
    time_search = time.time()
    k = sum(2 + i for i in range(args.steps))
    num_ops = len(PRIMITIVES)
    alphas_dim = (k, num_ops)
    pop_obj = initial_population(args.n_population, alphas_dim)
    pop_X = np.full_like(pop_obj, 100)  # bad values, will be replaced after evaluation
    logging.info(f">>>> Initial population of size {len(pop_obj)} created.")
  print("Running with config:")
  for key, value in vars(args).items():
    print(f"{key}: {value}")
  k = sum(2 + i for i in range(args.steps))
  num_ops = len(PRIMITIVES)
  alphas_dim = (k, num_ops)
  n_var = alphas_dim[0] * alphas_dim[1] * 2

  weights_r2 = utils_search.get_weights_r2(args.n_population)

  if args.pretrained_supernet is not None and initial_generation == 0:
    shutil.copy(args.pretrained_supernet, str(args.save_path_final_model) + os.sep + "super-net.pt")
    logging.info(f">>>> Pretrained supernet loaded from {args.pretrained_supernet} and saved to {str(args.save_path_final_model) + os.sep + 'super-net.pt'} for future reference.")
  return args, weights_r2, statistics, initial_generation, architectures_evaluated, pop_obj, pop_X, archive, archive_losses, n_var, time_search, alphas_dim

class NAS(Problem):
  def __init__(self, n_var, n_obj, xl, xu, args):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
    self.archive = []
    self.archive_2 = []
    self.args_problem = args
    self._n_evaluated = 0
    self.generations = 0

  def _evaluate(self, x, out, *args, **kwargs):

    objs = np.full((x.shape[0], self.n_obj), np.nan)
    population = []
    arch_id = self._n_evaluated + 1
    for i in range(x.shape[0]):
      args_individual = copy.copy(self.args_problem)
      args_individual.seed = args_individual.seed + arch_id
      performance = worker_evaluate_individual(self.generations, i, x[i,:].copy(), args_individual)
      objs[i, 0] = performance['std_loss']
      objs[i, 1] = performance['adv_loss']
      objs[i, 2] = performance['flops']
      objs[i, 3] = performance['params']
      self._n_evaluated += 1
      individual = Individual(X=x[i,:].copy(), k=self.n_obj, search_space='continuous')
      individual.F = objs[i, :].copy()
      individual.std_acc = performance['std_acc']
      individual.adv_acc = performance['adv_acc']
      if performance['genotype'] is not None:
        individual.genotype = performance['genotype']
        individual.feasible = True
      population.append(individual)
    self.archive = archive_update_pq(self.archive, population)
    self.archive_2 = archive_update_pq(self.archive_2, population, k=2)
    out["F"] = objs

if args.seed is None or args.seed < 0:
  args.seed = random.randint(1, 100000)
if args.reload_dir is None:
  DIR = "search-S1-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
  args.save_dir = DIR
  utils.create_exp_dir(DIR)
  save_params(args, DIR)
elif args.reload_dir == 'auto-last':
  # reload the last experiment in the results directory for the given algorithm and dataset
  base_dir = Path(".")

  if not base_dir.exists():
    raise ValueError("No experiments found for the given algorithm and dataset")

  dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("search-")]

  if not dirs:
    raise ValueError("No experiments found for the given algorithm and dataset")

  latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)

  DIR = str(latest_dir)
  args.reload_dir = DIR
  args.save_dir = DIR

else:
  DIR = args.reload_dir


#if os.path.exists("logs"):
#  shutil.rmtree("logs")
#os.makedirs("logs", exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

np.random.seed(args.seed)
random.seed(args.seed)

if args.dataset == 'cifar10':    num_classes = 10
elif args.dataset == 'cifar100': num_classes = 100

args, weights_r2, statistics, initial_generation, architectures_evaluated, pop_obj, pop_X, archive, archive_losses, n_var, elapsed_time, alphas_dim = prepare_args_supernet(args)

args.save_path_final_model = DIR

# create the algorithm object
algorithm = NSGA2(pop_size=args.n_population,
                  crossover=SimulatedBinaryCrossover(eta=15, prob=0.7),
                  mutation=PolynomialMutation(prob=args.mutate_rate, eta=20),
                  sampling=FloatRandomSampling()
                  )

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

lb = np.zeros(n_var)
ub = np.ones(n_var)

nas = NAS(n_var=n_var, n_obj=4, xl=lb, xu=ub, args=args)

# create an algorithm object that never terminates
algorithm.setup(problem=nas, termination=termination, seed=args.seed)

execution_time = time.time()

# STAGE 1
if initial_generation == 0 and args.pretrained_supernet is None and args.epochs_warmup > 0:
  logging.info(">>>> Starting warm-up phase for the supernet for {} epochs before starting the search process.".format(args.epochs_warmup))
  start_warm_up = time.time()
  train_supernet(pop_X, 0, args, warmup=True)
  elapsed_time = time.time() - start_warm_up

if initial_generation != 0:
  pop_initialized = algorithm.ask()
  pop_initialized.set("F", pop_obj)
  pop_initialized.set("X", pop_X)
  algorithm.tell(infills=pop_initialized)

for n_gen in range(initial_generation, args.generations):
  nas.generations = n_gen
  start_time_gen = time.time()
  np.random.seed(args.seed + n_gen)
  random.seed(args.seed + n_gen)
  # ask the algorithm for the next solution to be evaluated
  pop = algorithm.ask()

  train_supernet(pop.get("X"), n_gen, args, warmup=False)
  algorithm.evaluator.eval(nas, pop)
  pop_obj = pop.get("F")
  pop_X = pop.get("X")

  architectures_evaluated += len(pop)

  #archive = archive_update_pq(archive, pop_obj)
  #archive_losses = archive_update_pq(archive_losses, pop_obj[:, :2], k=2)
  hyp, hyp2, r2 = utils_search.store_metrics(architectures_evaluated, algorithm.problem.archive, algorithm.problem.archive_2, args, weights_r2, statistics)
  plot_hypervolume(statistics, args.save_path_final_model)
  plot_hypervolume2(statistics, args.save_path_final_model)
  plot_r2(statistics, args.save_path_final_model)

  elapsed_time += time.time() - start_time_gen
  utils_search.store_population_data(n_gen, architectures_evaluated, pop_obj, pop_X, algorithm.problem.archive,
                        algorithm.problem.archive_2, statistics, elapsed_time, args.save_path_final_model, alphas_dim, args)

  logging.info(f'>>>>>>> Generation {n_gen}')
  logging.info(f'        hyp: {hyp}, hyp_2: {hyp2}, R2: {r2}')
  logging.info(f'        Architectures evaluated so far: {architectures_evaluated}')
  logging.info(f"        Time elapsed for generation {n_gen} (HH:MM:SS): {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time_gen))}")

  algorithm.tell(infills=pop)


logging.info('Total search time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - execution_time))))

# obtain the result objective from the algorithm
res = algorithm.result()

for i, ind in enumerate(algorithm.problem.archive):
    save_architecture(i, ind, args.save_path_final_model)

save_archive(algorithm.problem.archive, args.save_path_final_model)
save_archive_2(algorithm.problem.archive_2, args.save_path_final_model)
plot_archive_losses(algorithm.problem.archive_2, args.save_path_final_model)
plot_hypervolume(statistics, args.save_path_final_model)
plot_hypervolume2(statistics, args.save_path_final_model)
plot_r2(statistics, args.save_path_final_model)
save_statistics_to_csv(statistics, args.save_path_final_model)
print('Results stored in {}'.format(args.save_path_final_model))
