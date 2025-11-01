import time

import numpy as np

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy
from evaluation.model_search import discretize
from evaluation.train_search import infer, run_batch_epoch
from evolutionary import AlphaProblem, unpack_alphas, flatten_alphas, tournament_r2
import torch
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection

from indicators import normalize_objectives, get_dynamic_r2_reference, contribution_r2


def initial_population(n_population, alphas_dim, k):
    individuals = []
    for i in range(n_population):
        normal_arch = torch.rand((alphas_dim[0], alphas_dim[1]))
        reduction_arch = torch.rand((alphas_dim[0], alphas_dim[1]))
        individuals.append(Individual(X=flatten_alphas((normal_arch, reduction_arch)))
                            .set("k", k)
                            .set("c_r2", 0.0)
                            .set("adv_acc", 0.0)
                            .set("std_acc", 0.0)
                            .set("F", np.zeros(k))
                            .set("F_norm", np.zeros(k))
                            .set("genotype", None))
    return Population(individuals=individuals)

def eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, device, statisctics):
    model.eval()
    objective_space = np.empty((pop.size, args.objectives))
    for i, individual in enumerate(pop):
        individual_architect = unpack_alphas(individual.X, model.alphas_dim)
        model.update_arch_parameters(individual_architect)
        discrete = discretize(individual_architect, model.genotype(), device)
        model.update_arch_parameters(discrete)
        individual.set("genotype", model.genotype())
        time_stamp = time.time()
        std_acc, adv_acc, std_loss, adv_loss, ws_loss = infer(valid_queue, model, criterion, attack_f, args)
        individual.std_acc = std_acc
        individual.adv_acc = adv_acc
        individual.F[args.std_loss_index] = std_loss
        individual.F[args.adv_loss_index] = adv_loss
        model_flops, model_parameters = utils.get_model_metrics(model.genotype(), model)
        individual.F[args.flops_index] = model_flops
        individual.F[args.params_index] = model_parameters
        individual.F_norm = np.zeros(args.objectives)
        objective_space[i, :] = individual.F
        print(f"Evaluation {i + 1}/{pop.size}: std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, loss {ws_loss:.4f} ({time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))}) (HH:MM:SS)")
    normalize_objectives(pop)
    z_ref = get_dynamic_r2_reference(pop)
    for ind in pop:
        ind.c_r2 = contribution_r2(pop, ind, weights_r2, z_ref)
    utils.store_statisctics(statisctics, objective_space)

def train_supernet(pop, train_queue, model, criterion, optimizer, attack_f, epoch, scheduler, args):
    model.train()
    for n_batch, (input, target) in enumerate(train_queue):
        individual = pop[n_batch % args.n_population]
        individual_architect = unpack_alphas(individual.X, model.alphas_dim)
        time_stamp = time.time()
        std_acc, adv_acc, loss = run_batch_epoch(model, individual_architect, input, target, criterion, optimizer,
                                                 attack_f, args)
        if n_batch % args.report_freq == 0:
            print(
                f">>>> Epoch {epoch}/{args.epochs} Batch {n_batch + 1}/{len(train_queue)} ({time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))}) (HH:MM:SS): std_acc {std_acc / args.batch_size * 100:.2f}%, adv_acc {adv_acc / args.batch_size * 100:.2f}%, loss {loss:.4f}")
    scheduler.step()

def r2_emoa_rnas(args, train_queue, valid_queue, model, criterion, optimizer, scheduler, attack_f, weights_r2):
    problem = AlphaProblem(model.alphas_dim)
    archive = []
    archive_accuracy = []
    pop = initial_population(args.n_population, model.alphas_dim, args.objectives)
    print(f">>>> Initial population of size {pop.size} created.")
    train_supernet(pop, train_queue, model, criterion, optimizer, attack_f, 0, scheduler, args)
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'), 'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'r2_log': []}
    eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, args.device, statistics)
    archive = archive_update_pq(archive, pop)
    utils.store_metrics(0, archive, args, weights_r2, statistics)
    time_search = time.time()
    for epoch in range(args.epochs):
        start = time.time()
        time_stamp_epoch = time.time()
        train_supernet(pop, train_queue, model, criterion, optimizer, attack_f, epoch + 1, scheduler, args)
        print(f">>>> Epoch {epoch + 1} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_epoch))} (HH:MM:SS)")

        selection = TournamentSelection(func_comp=tournament_r2)
        parents = selection.do(problem=problem, pop=pop, n_parents=2, n_select=pop.size // 2, to_pop=False)
        sbx = SBX(prob=args.prob_cross, eta=args.eta_cross, n_offsprings=2)
        pm = PM(prob=args.prob_mut, eta=args.eta_mut)

        offsprings = sbx.do(problem, pop, parents=parents)
        mutation = pm.do(problem, offsprings)
        for p in mutation:
            p.set("k", args.objectives)
            p.set("c_r2", 0.0)
            p.set("adv_acc", 0.0)
            p.set("std_acc", 0.0)
            p.set("F", np.zeros(args.objectives))
            p.set("F_norm", np.zeros(args.objectives))
            p.set("genotype", None)

        print(f'>>>>> size parents: {len(parents)}, size offsprings: {len(offsprings)}')
        # Evaluate offspring
        eval_population(model, mutation, valid_queue, args, criterion, attack_f, weights_r2, args.device, statistics)
        print(f"Tiempo total de entrenamiento/validacion {args.epochs}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} (HH:MM:SS)")

        archive = archive_update_pq(archive, Population.merge(pop, mutation))
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, Population.merge(pop, mutation))
        pop = update_population_r2(pop, mutation, weights_r2)
        utils.store_metrics(epoch, archive, args, weights_r2, statistics)
        utils.save_supernet(model, args.save_path_final_model)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
    print(f">>>> Total search time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_search))} (HH:MM:SS)")
    return model, archive, archive_accuracy, statistics

def update_population_r2(pop, offspring, weights_r2):
    c = Population.merge(pop, offspring)
    n = pop.size
    assert len(c) >= 2*n
    for i in range(n):
        z = get_dynamic_r2_reference(pop)
        for ind in c:
            ind.c_r2 = contribution_r2(c, ind, weights_r2, z)
        worst = sorted(c, key=lambda x: x.c_r2, reverse=True)[0]
        c = np.delete(c, np.where(c == worst)[0][0])
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return Population(individuals=c)
