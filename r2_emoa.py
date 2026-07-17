import logging
import random

import numpy as np
import time

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy
from micro_space.micro_encoding import PRIMITIVES
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
def prepare_args_standard(args_):
    if args_.reload_dir is not None:
        # the execution is a reload and we need to load all the variables from the previous execution
        args, statistics, initial_generation, pop, archive, archive_acc_4objs, archive_accuracy, archive_losses, nadir_point, ideal_point, time_search = utils.load_execution(args_.reload_dir)
        architectures_evaluated = len(statistics['hyp_log']) * args.n_population
        logging.info(f">>>> Reloading execution from {args_.reload_dir} at generation {initial_generation} with {architectures_evaluated} architectures already evaluated.")
    else:
        # the execution is new and we need to initialize all the variables
        args = args_
        initial_generation = 0
        archive = []
        archive_acc_4objs = []
        archive_accuracy = []
        archive_losses = []
        architectures_evaluated = 0
        nadir_point = np.ones(4, )
        ideal_point = np.zeros(4, )
        np.random.seed(args.seed)
        random.seed(args.seed)
        statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'),
                      'min_f2': float('inf'),
                      'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': [], 'hyp_acc_log': [], 'hyp2_acc_log': []}
        k = sum(2 + i for i in range(args.steps))
        num_ops = len(PRIMITIVES)
        alphas_dim = (k, num_ops)
        time_search = time.time()
        pop = initial_population(args.n_population, alphas_dim, args.objectives, args)
        logging.info(f">>>> Initial population of size {len(pop)} created.")
    print("Running with config:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    #weights_r2 = utils.get_weights_r2(args.n_population)
    weights_r2 = utils.get_weights_r2_file(args.r2_weights_dir)

    return args, weights_r2, archive, archive_acc_4objs, archive_accuracy, archive_losses, nadir_point, ideal_point, architectures_evaluated, initial_generation, pop, statistics, time_search

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

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
def r2_emoa_rnas(args_):
    args, weights_r2, archive, archive_acc_4objs, archive_accuracy, archive_losses, nadir_point, ideal_point, architectures_evaluated, initial_generation, pop, statistics, time_search = prepare_args_standard(args_)
    if initial_generation == 0:
        evaluate_population_multiprocessing(0, pop, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += args.n_population
        update_ref_points(pop, nadir_point, ideal_point, losses=args.losses_objs)
        archive = archive_update_pq(archive, pop)
        archive_acc_4objs = archive_update_pq(archive_acc_4objs, pop, k=4, losses=False)
        archive_losses = archive_update_pq(archive_losses, pop, k=2)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop)
        hyp_archive, hyp_2, hyp4_acc, hyp2_acc, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, archive_accuracy, archive_acc_4objs, args,
                                                         weights_r2, statistics)
        utils.store_population_data(0, pop, archive, archive_acc_4objs, archive_accuracy, archive_losses, statistics, nadir_point, ideal_point, time_search, args.save_path_final_architect)
        logging.info(f">>>> Gen 0 | Hypervolume (4 objs, loss): {hyp_archive}, Hypervolume (2 objs, loss): {hyp_2}, Hypervolume (4 objs, acc): {hyp4_acc}, Hypervolume (2 objs, acc): {hyp2_acc}, R2: {r2_archive}")
        initial_generation = 1
    for generation in range(initial_generation, args.generations):
        set_random_seed(args.seed + generation)  # Ensure reproducibility across generations
        if args.increase_epochs and generation % 10 == 0 and generation != initial_generation:
            args.epochs_train_individual += 5
        time_stamp_gen = time.time()

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        if args.search_space == 'discrete':
            offsprings = point_crossover(parents, n_childs=args.n_population, prob_cross=args.prob_cross)
        else:
            offsprings = binary_crossover(parents, n_childs=args.n_population, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        evaluate_population_multiprocessing(generation, mutation, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += args.n_population
        update_ref_points(mutation, nadir_point, ideal_point, losses=args.losses_objs)

        archive = archive_update_pq(archive, pop + mutation)
        archive_acc_4objs = archive_update_pq(archive_acc_4objs, pop + mutation, losses=False)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(args.n_population, pop, mutation, weights_r2, losses=args.losses_objs)
        hyp_archive, hyp_2, hyp4_acc, hyp2_acc, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, archive_accuracy, archive_acc_4objs, args, weights_r2, statistics)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect, losses=False)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2_acc(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        utils.store_statisctics(statistics, np.array([p.F for p in mutation if p.feasible]))
        utils.store_population_data(generation, pop, archive, archive_acc_4objs, archive_accuracy, archive_losses, statistics, nadir_point, ideal_point, time_search, args.save_path_final_architect)
        logging.info(f">>>> Gen {generation} | Hypervolume (4 objs, loss): {hyp_archive}, Hypervolume (2 objs, loss): {hyp_2}, Hypervolume (4 objs, acc): {hyp4_acc}, Hypervolume (2 objs, acc): {hyp2_acc}, R2: {r2_archive}")
        logging.info(
            f">>>> Gen {generation} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
    logging.info(f">>>> Total architectures evaluated: {architectures_evaluated}")
    logging.info(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    statistics['total_time_search'] = time.time() - time_search
    return archive, archive_accuracy, archive_losses, statistics
