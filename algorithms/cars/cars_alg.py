import logging
import os
import shutil
import time
import random

import numpy as np

import utils
from nsga3 import CARS_NSGA
from archivers import archive_update_pq, archive_update_pq_accuracy
from evolutionary import tournament_selection, binary_crossover, polynomial_mutation
from individual import Individual
from micro_space.micro_encoding import PRIMITIVES
from worker_process import evaluate_population_multiprocessing, train_supernet


def prepare_args_supernet(args_):
    if args_.reload_dir is not None:
        # the execution is a reload and we need to load all the variables from the previous execution
        args, statistics, initial_generation, pop, archive, archive_accuracy, archive_losses, time_search = utils.load_execution(args_.reload_dir)
        print("Running with config:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")
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
        np.random.seed(args.seed)
        random.seed(args.seed)
        statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'),
                      'min_f2': float('inf'),
                      'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
        k = sum(2 + i for i in range(args.steps))
        num_ops = len(PRIMITIVES)
        alphas_dim = (k, num_ops)
        time_search = time.time()
        pop = initial_population(args.n_population, alphas_dim, args.objectives, args.search_space)
        logging.info(f">>>> Initial population of size {len(pop)} created.")

    weights_r2 = utils.get_weights_r2(args.n_population)

    if args.pretrained_supernet is not None:
        shutil.copy(args.pretrained_supernet, str(args.save_path_final_model) + os.sep + "super-net.pt")

    return args, weights_r2, archive, archive_accuracy, archive_losses, architectures_evaluated, initial_generation, pop, statistics, time_search

def initial_population(n_population, alphas_dim, k, search_space):
    individuals = []
    for i in range(n_population):
        flattened = np.random.rand(alphas_dim[0]*alphas_dim[1]*2)
        individuals.append(Individual(X=flattened.copy(), k=k, search_space=search_space))
    return individuals

def cars_algorithm(args_):
    args, weights_r2, archive, archive_accuracy, archive_losses, architectures_evaluated, initial_generation, pop, statistics, time_search = prepare_args_supernet(args_)

    if initial_generation == 0:
        if args.epochs_warmup > 0:
            logging.info(">>>> Warmup training of the supernet...")
            train_supernet(pop, 0, args, warmup=True)
            logging.info(">>>> Warmup training DONE.")
        statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': [],
                  'lr_log': []}
        evaluate_population_multiprocessing(0, pop, args)
        archive = archive_update_pq(archive, pop)
        archive_losses = archive_update_pq(archive_losses, pop, k=2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                         weights_r2, statistics)
        utils.store_population_data(0, pop, archive, archive_accuracy, archive_losses, statistics, time_search, args.save_path_final_architect)
        logging.info(f">>>> Gen 0 | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        initial_generation += 1


    for generation in range(initial_generation, args.generations):
        np.random.seed(args.seed + generation)
        random.seed(args.seed + generation)
        if args.increase_epochs and generation % 10 == 0 and generation != initial_generation:
            args.epochs_train_supernet += 5

        start = time.time()
        time_stamp_epoch = time.time()
        train_supernet(pop, generation, args, warmup=False)
        logging.info(
            f">>>> Gen {generation} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_epoch))} (HH:MM:SS)")

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        offsprings = binary_crossover(parents, n_childs=args.n_population, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        # Evaluate offspring
        evaluate_population_multiprocessing(generation, mutation, args)

        logging.info(
            f"Tiempo total de entrenamiento/validacion {args.generations}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} (HH:MM:SS)")
        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)

        total_population = pop + mutation

        first_objective = np.array([ind.F[1] for ind in total_population]).copy()
        other_objectives = np.array([[ind.F[0], ind.F[2], ind.F[3]] for ind in total_population]).copy()

        keep = CARS_NSGA(target=first_objective, objs=other_objectives, N=len(pop))

        next_pop = []
        for i in keep:
            next_pop.append(total_population[i])
        pop = next_pop

        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args, weights_r2, statistics)
        utils.store_population_data(generation, pop, archive, archive_accuracy, archive_losses, statistics, time_search, args.save_path_final_architect)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        utils.store_statisctics(statistics, np.array([p.F for p in mutation if p.feasible]))
        logging.info(f"Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    logging.info(f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return archive, archive_accuracy, archive_losses, statistics