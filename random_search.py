import utils
import logging
import time
import random
import numpy as np

from archivers import archive_update_pq, archive_update_pq_accuracy
from individual import Individual
from micro_space.micro_encoding import PRIMITIVES
from worker_process import evaluate_population_multiprocessing


def prepare_args_standard(args_):
    if args_.reload_dir is not None:
        # the execution is a reload and we need to load all the variables from the previous execution
        args, statistics, initial_generation, pop, archive, archive_accuracy, archive_losses, nadir_point, ideal_point, time_search = utils.load_execution(args_.reload_dir)
        architectures_evaluated = len(statistics['hyp_log']) * args.n_population
        logging.info(f">>>> Reloading execution from {args_.reload_dir} at generation {initial_generation} with {architectures_evaluated} architectures already evaluated.")
    else:
        # the execution is new and we need to initialize all the variables
        args = args_
        initial_generation = 0
        archive = []
        archive_accuracy = []
        archive_losses = []
        architectures_evaluated = 0
        nadir_point = np.ones(4, )
        ideal_point = np.zeros(4, )
        np.random.seed(args.seed)
        random.seed(args.seed)
        statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'),
                      'min_f2': float('inf'),
                      'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'hyp2_acc_log': [], 'r2_log': []}
        k = sum(2 + i for i in range(args.steps))
        num_ops = len(PRIMITIVES)
        alphas_dim = (k, num_ops)
        time_search = time.time()
        pop = initial_population(args.n_population, alphas_dim, args.objectives, args)
        logging.info(f">>>> Initial population of size {len(pop)} created.")
    k = sum(2 + i for i in range(args.steps))
    num_ops = len(PRIMITIVES)
    alphas_dim = (k, num_ops)

    print("Running with config:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    weights_r2 = utils.get_weights_r2_file(args.r2_weights_dir)

    return args, alphas_dim, weights_r2, archive, archive_accuracy, archive_losses, nadir_point, ideal_point, architectures_evaluated, initial_generation, pop, statistics, time_search

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_discrete_bounds(args):
    n_var = 4 * args.steps * 2
    n_ops = len(PRIMITIVES)

    lb = np.zeros(n_var, dtype=np.int32)
    ub = np.ones(n_var, dtype=np.int32)

    h = 1

    for b in range(0, n_var // 2, 4):
        ub[b] = n_ops - 1
        ub[b + 1] = h
        ub[b + 2] = n_ops - 1
        ub[b + 3] = h
        h += 1

    ub[n_var // 2:] = ub[:n_var // 2]

    return lb, ub

def initial_population(n_population, alphas_dim, k, args, random_population=False):
    individuals = []
    if args.initial_population is not None and not random_population:
        # Load initial population from file
        X = np.load(args.initial_population)
        if X.shape[0] != n_population:
            raise ValueError(f"Initial population file contains only {X.shape[0]} individuals, but n_population is set to {n_population}.")
    else:
        if args.search_space == "discrete":
            lb, ub = get_discrete_bounds(args)
            X = np.column_stack([np.random.randint(int(lb[j]), int(ub[j]) + 1, size=n_population) for j in range(len(lb))]).astype(np.int32)
        else:
            n_var = alphas_dim[0] * alphas_dim[1] * 2
            X = np.random.rand(n_population, n_var)

    for i in range(n_population):
        individuals.append(Individual(X=X[i].copy(), k=k, search_space=args.search_space))

    return individuals

def random_search_rnas(args_):
    args, alphas_dim, weights_r2, archive, archive_accuracy, archive_losses, nadir_point, ideal_point, architectures_evaluated, initial_generation, pop, statistics, time_search = prepare_args_standard(args_)
    if initial_generation == 0:
        evaluate_population_multiprocessing(0, pop, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += args.n_population
        archive = archive_update_pq(archive, pop)
        archive_losses = archive_update_pq(archive_losses, pop, k=2)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop)
        hyp_archive, hyp_2, hyp2_acc, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, archive_accuracy, args,
                                                         weights_r2, statistics)
        utils.store_population_data(0, pop, archive, archive_accuracy, archive_losses, statistics, nadir_point, ideal_point, time_search, args.save_path_final_architect)
        logging.info(f">>>> Gen 0 | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        initial_generation = 1
    for generation in range(initial_generation, args.generations):
        set_random_seed(args.seed + generation)  # Ensure reproducibility across generations
        if args.increase_epochs and generation % 10 == 0 and generation != initial_generation:
            args.epochs_train_individual += 5
        time_stamp_gen = time.time()

        new_population = initial_population(args.n_population, alphas_dim, 4, args, random_population=True)
        evaluate_population_multiprocessing(generation, new_population, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += len(new_population)
        archive = archive_update_pq(archive, new_population)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, new_population)
        archive_losses = archive_update_pq(archive_losses, new_population, k=2)

        hyp_archive, hyp_2, hyp2_acc, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, archive_accuracy, args,
                                                             weights_r2, statistics)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        utils.store_statisctics(statistics, np.array([p.F for p in new_population if p.feasible]))
        utils.store_population_data(generation, pop, archive, archive_accuracy, archive_losses, statistics, nadir_point, ideal_point, time_search, args.save_path_final_architect)
        logging.info(f">>>> Gen {generation} | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        logging.info(
            f">>>> Gen {generation} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
    logging.info(f">>>> Total architectures evaluated: {architectures_evaluated}")
    logging.info(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    statistics['total_time_search'] = time.time() - time_search
    return archive, archive_accuracy, archive_losses, statistics