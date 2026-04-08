import json
import os
import ssl
import sys
import subprocess
import random

import numpy as np
import time

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy, dominates
from micro_space.micro_encoding import PRIMITIVES, Genotype
from individual import Individual
from evolutionary import tournament_selection, binary_crossover, polynomial_mutation, point_crossover
from indicators import contribution_r2, update_ref_points


"""
 python3 rnas_search.py --seed 18906049 --algorithm r2-emoa --dataset cifar10 --batch_size 32  \
 --n_population 10 --epochs_train_individual 2 --generations 2 \
 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 \
 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 \
 --attack FGSM --fgsm_eps 8/255 --cutout False --cutout_length 16 --drop_path_prob 0.3 \
 --grad_clip 0.5 --train_portion 0.5
"""
def prepare_args_standard(args):

    np.random.seed(args.seed)
    random.seed(args.seed)

    ssl._create_default_https_context = ssl._create_unverified_context

    weights_r2 = utils.get_weights_r2(args.n_population)

    k = sum(2 + i for i in range(args.steps))
    num_ops = len(PRIMITIVES)
    alphas_dim = (k, num_ops)

    return alphas_dim, weights_r2

def initial_population(n_population, alphas_dim, k, args):
    individuals = []
    for i in range(n_population):
        if args.search_space == 'discrete':
            # Each group of 4 integers represents two operations for a given node (op1, from_node1, op2, from_node2)
            n_var = 4 * args.steps * 2
            n_ops = len(PRIMITIVES)
            flattened = np.zeros(n_var, dtype=np.int32)
            h = 1
            for b in range(0, n_var // 2, 4):
                flattened[b] = np.random.randint(0, n_ops)
                flattened[b+1] = np.random.randint(0, h + 1)
                flattened[b+2] = np.random.randint(0, n_ops)
                flattened[b+3] = np.random.randint(0, h + 1)
                h += 1
            flattened[n_var // 2:] = flattened[:n_var // 2]
        else:
            flattened = np.random.rand(alphas_dim[0]*alphas_dim[1]*2)
        individuals.append(Individual(X=flattened.copy(), k=k, search_space=args.search_space))
    return individuals

def worker_evaluate_individual(gen, i, individual_X, weight_individual, nadir_point, ideal_point, args, return_dict):
    log_file = "logs" + os.sep + f"worker_{gen}_{i}.log"
    result_file = "logs" + os.sep + f"result_gen{gen}_ind{i}.json"

    process_args = [
        sys.executable, "-X dev -u", "individual_worker.py",
        '--gen', str(gen),
        '--i', str(i),
        '--seed', str(args.seed + i),  # Different seed for each process to avoid identical weight initialization
        '--individual_x', np.array2string(individual_X, separator=',', max_line_width=np.inf),
        '--search_space', str(args.search_space),
        '--dataset', str(args.dataset),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--epochs_train_individual', str(args.epochs_train_individual),
        '--data', str(args.data),
        '--mu', str(args.mu),
        '--learning_rate', str(args.learning_rate),
        '--learning_rate_min', str(args.learning_rate_min),
        '--momentum', str(args.momentum),
        '--weight_decay', str(args.weight_decay),
        '--init_channels', str(args.init_channels),
        '--layers', str(args.layers),
        '--steps', str(args.steps),
        '--multiplier', str(args.multiplier),
        '--fgsm_eps', str(args.fgsm_eps),
        '--cutout_length', str(args.cutout_length),
        '--drop_path_prob', str(args.drop_path_prob),
        '--grad_clip', str(args.grad_clip),
        '--train_portion', str(args.train_portion),
        '--weight_individual', np.array2string(weight_individual, separator=',', max_line_width=np.inf),
        '--nadir_point', np.array2string(nadir_point, separator=',', max_line_width=np.inf),
        '--ideal_point', np.array2string(ideal_point, separator=',', max_line_width=np.inf)
    ]

    if args.reduction:
        process_args.append('--reduction')
    if args.cutout:
        process_args.append('--cutout')
    env_worker = os.environ.copy()
    env_worker['CUDA_VISIBLE_DEVICES'] = str(args.gpu) # Set the GPU device for the subprocess
    if args.debug_cuda:
        env_worker['CUDA_LAUNCH_BLOCKING'] = '1'
        env_worker['TORCH_USE_CUDA_DSA'] = '1'

    with open(log_file, 'w') as f_log:
        process = subprocess.Popen(process_args,
                                   stdout=f_log,
                                   stderr=f_log,
                                   text=True,
                                   env=env_worker)

        try:
            process.communicate(timeout=args.timestamp * 60)

            if process.returncode == 0 and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    res_dict = json.load(f)
                    res_dict['genotype'] = Genotype(**res_dict['genotype'])
                    return_dict[i] = res_dict
                os.remove(result_file)
                print(f"Gen {gen} Individual {i}: std_acc {return_dict[i]['std_acc']:.2f}, adv_acc {return_dict[i]['adv_acc']:.2f} std_loss {return_dict[i]['std_loss']:.3f}, adv_loss {return_dict[i]['adv_loss']:.3f}, flops {return_dict[i]['flops']:.2f}, params {return_dict[i]['params']:.2f}")
            else:
                print(f"Gen {gen} Individual {i} failed with return code {process.returncode}")

        except subprocess.TimeoutExpired:
            print(f"Individual {i} exceed timestamp, it will be removed from the population.")
            process.kill()
            process.communicate(timeout=10)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt received. Terminating individual {i} process.")
            process.terminate()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)    
            sys.exit('Search interrupted by user.')
        except Exception as e:
            print(f"Failed {i}: {e}")
        finally:
            # wait a bit to ensure the process has terminated and released resources before starting the next one
            time.sleep(5)
            # set default values for failed individuals
            if i not in return_dict:
                return_dict[i] = {
                    "std_acc": 0.0,
                    "adv_acc": 0.0,
                    "std_loss": 1000.0,
                    "adv_loss": 1000.0,
                    "flops": 1000.0,
                    "params": 1000.0,
                    "genotype": None
                }




def evaluate_population_multiprocessing(gen, pop, weights_r2, nadir_point, ideal_point, args):
    return_dict = {}
    for i, individual in enumerate(pop):
        weight_individual = weights_r2[len(pop)][i].copy()
        worker_evaluate_individual(gen, i, individual.X.copy(),
                                   weight_individual, nadir_point.copy(),
                                   ideal_point.copy(), args, return_dict)
        individual.std_acc = float(return_dict[i]["std_acc"])
        individual.adv_acc = float(return_dict[i]["adv_acc"])
        individual.F = np.array([
            float(return_dict[i]["std_loss"]),
            float(return_dict[i]["adv_loss"]),
            float(return_dict[i]["flops"]),
            float(return_dict[i]["params"])
        ], dtype=np.float64)
        individual.genotype = return_dict[i]["genotype"]
        individual.feasible = True if individual.genotype is not None else False


# R2 version where each architecture has its own weights (no supernet training). This is a baseline to compare with the supernet version.
def r2_emoa_rnas(args):
    alphas_dim, weights_r2 = prepare_args_standard(args)
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    nadir_point = np.ones(4,)
    ideal_point = np.zeros(4,)
    time_search = time.time()
    pop = initial_population(args.n_population, alphas_dim, args.objectives, args)
    print(f">>>> Initial population of size {len(pop)} created.")
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
    evaluate_population_multiprocessing(0, pop, weights_r2, nadir_point, ideal_point, args)
    update_ref_points(pop, nadir_point, ideal_point)

    archive = archive_update_pq(archive, pop)
    archive_losses = archive_update_pq(archive_losses, pop, k=2)
    hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                         weights_r2, statistics)
    print(f">>>> Gen 0 | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    for generation in range(args.generations):
        time_stamp_gen = time.time()

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        if args.search_space == 'discrete':
            offsprings = point_crossover(parents, n_childs=args.n_population, prob_cross=args.prob_cross)
        else:
            offsprings = binary_crossover(parents, n_childs=args.n_population, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        evaluate_population_multiprocessing(generation+1, mutation, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += args.n_population
        update_ref_points(mutation, nadir_point, ideal_point)

        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(args.n_population, pop, mutation, weights_r2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                             weights_r2, statistics)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        print(f">>>> Gen {generation + 1} | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        print(
            f">>>> Gen {generation + 1} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
    print(f">>>> Total architectures evaluated: {architectures_evaluated}")
    print(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return archive, archive_accuracy, archive_losses, statistics


def non_dominated_sort(population):
    N = len(population)
    S = [[] for _ in range(N)] # solutions dominated by i
    n = [0] * N # number of solutions dominating i
    fronts = [[]]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(population[i], population[j], k=4):
                S[i].append(j)
            elif dominates(population[j], population[i], k=4):
                n[i] += 1

    # First front
    for i in range(N):
        if n[i] == 0:
            fronts[0].append(i)

    # Build other fronts
    f = 0
    while len(fronts[f]) > 0:
        next_front = []
        for i in fronts[f]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break
        f += 1

    return [[population[i] for i in front] for front in fronts]

def update_population_r2(n, pop, offspring, weights_r2):
    c = pop + offspring
    # Remove unfeasible solutions before sorting and calculating contributions
    c = [p for p in c if p.feasible]
    fronts = non_dominated_sort(c)
    last_front = len(fronts) - 1
    while len(c) > n:
        weights = weights_r2[len(c)]
        front_k = fronts[last_front]
        if last_front < 0:
            break
        if len(front_k) == 0:
            last_front -= 1
            continue
        if len(front_k) == 1:
            worst = front_k[0]
            c.remove(worst)
            front_k.remove(worst)
            last_front -= 1
            continue
        z_ref = np.min([ind.F for ind in front_k], axis=0)
        nadir_point = np.max([ind.F for ind in front_k], axis=0)
        for ind in front_k:
            ind.c_r2 = contribution_r2(front_k, ind, weights, nadir_point, z_ref)
            #print(f"Individual {ind.F} R2 contribution {ind.c_r2}")
        worst = sorted(front_k, key=lambda x: x.c_r2)[0]
        c.remove(worst)
        front_k.remove(worst)
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return c
