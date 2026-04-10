import logging
import ssl
import random

import numpy as np
import time

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy, dominates
from micro_space.micro_encoding import PRIMITIVES, Genotype
from individual import Individual
from evolutionary import tournament_selection, binary_crossover, polynomial_mutation, point_crossover, \
    update_population_r2
from indicators import update_ref_points
from worker_process import evaluate_population_multiprocessing

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
    logging.info(f">>>> Initial population of size {len(pop)} created.")
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
    evaluate_population_multiprocessing(args.n_population, 0, pop, weights_r2, nadir_point, ideal_point, args)
    update_ref_points(pop, nadir_point, ideal_point)

    archive = archive_update_pq(archive, pop)
    archive_losses = archive_update_pq(archive_losses, pop, k=2)
    hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                         weights_r2, statistics)
    logging.info(f">>>> Gen 0 | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    for generation in range(args.generations):
        time_stamp_gen = time.time()

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        if args.search_space == 'discrete':
            offsprings = point_crossover(parents, n_childs=args.n_population*2, prob_cross=args.prob_cross)
        else:
            offsprings = binary_crossover(parents, n_childs=args.n_population*2, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        evaluate_population_multiprocessing(args.n_population, generation+1, mutation, weights_r2, nadir_point, ideal_point, args)
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
        utils.store_statisctics(statistics, np.array([p.F for p in mutation if p.feasible]))
        logging.info(f">>>> Gen {generation + 1} | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        logging.info(
            f">>>> Gen {generation + 1} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
    logging.info(f">>>> Total architectures evaluated: {architectures_evaluated}")
    logging.info(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return archive, archive_accuracy, archive_losses, statistics
