import time

import numpy as np

import utils
from archivers import archive_update_pq
from evaluation.model_search import discretize
from evaluation.train_search import infer, run_batch_epoch
from evolutionary import AlphaProblem, unpack_alphas, flatten_alphas, tournament_r2
import torch
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection

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
                            .set("F", np.zeros(k)))
    return Population(individuals=individuals)

def r2(population, weights, z_ref):
    acc = 0.0
    for w in weights:
        min_diff = float('inf')
        for p in population:
            max_diff = max([w[i] * abs(p.F[i] - z_ref[i]) for i in range(len(p.F))])
            min_diff = min(min_diff, max_diff)
        acc += min_diff
    return acc / weights.shape[0]

def contribution_r2(population, individual, weights, z_ref):
    n = population.size
    assert weights[n].shape[0] == n, f"weights shape != population size {n}"
    full = r2(population, weights[n], z_ref)
    population_exclude = [p for p in population if p != individual]
    excl = r2(Population(individuals=population_exclude), weights[n], z_ref)
    return abs(full - excl)

def get_dynamic_r2_reference(population):
    n_obj = len(population[0].F)
    z_ref = [0] * n_obj
    max_f = 0
    for i in range(n_obj):
        max_f_i = max([ind.F[i] for ind in population])
        min_f_i = min([ind.F[i] for ind in population])
        max_f = max(max_f, max_f_i - min_f_i)
    for i in range(n_obj):
        min_f_i = min([ind.F[i] for ind in population])
        z_ref[i] = min_f_i - max_f
    return z_ref

def eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, device):
    objective_space = np.empty((pop.size, args.objectives))
    for i, individual in enumerate(pop):
        individual_architect = unpack_alphas(individual.X, model.alphas_dim)
        model.update_arch_parameters(individual_architect)
        discrete = discretize(individual_architect, model.genotype(), device)
        model.update_arch_parameters(discrete)
        time_stamp = time.time()
        std_acc, adv_acc, std_loss, adv_loss, ws_loss = infer(valid_queue, model, criterion, attack_f, args)
        individual.std_acc = std_acc
        individual.adv_acc = adv_acc
        individual.F[args.std_loss_index] = std_loss
        individual.F[args.adv_loss_index] = adv_loss
        model_flops, model_parameters = utils.get_model_metrics(model.genotype(), model)
        individual.F[args.flops_index] = model_flops
        individual.F[args.params_index] = model_parameters
        objective_space[i, :] = individual.F
        print(f"Evaluation {i + 1}/{pop.size}: std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, loss {ws_loss:.4f} ({(time.time() - time_stamp):.4f} seg)")
    z_ref = get_dynamic_r2_reference(pop)
    for ind in pop:
        ind.c_r2 = contribution_r2(pop, ind, weights_r2, z_ref)
    utils.store_metrics(objective_space, args.algorithm)

def r2_emoa_rnas(args, train_queue, valid_queue, model, criterion, optimizer, scheduler, attack_f, weights_r2):
    problem = AlphaProblem(model.alphas_dim)
    pop = initial_population(args.n_population, model.alphas_dim, args.objectives)
    archive = []

    for epoch in range(args.epochs):
        start = time.time()

        model.train()
        time_stamp_epoch = time.time()
        for n_batch, (input, target) in enumerate(train_queue):
            individual = pop[n_batch % args.n_population]
            individual_architect = unpack_alphas(individual.X, model.alphas_dim)
            time_stamp = time.time()
            std_acc, adv_acc, loss = run_batch_epoch(model, individual_architect, input, target, criterion, optimizer, attack_f, args)
            if n_batch % args.report_freq == 0:
                print(f">>>> Epoch {epoch + 1}/{args.epochs} Batch {n_batch + 1}/{len(train_queue)} ({(time.time() - time_stamp):.4f}) seg : std_acc {std_acc / args.batch_size * 100:.2f}%, adv_acc {adv_acc / args.batch_size * 100:.2f}%, loss {loss:.4f}")
        scheduler.step()
        print(f">>>> Epoch {epoch + 1} training DONE in {(time.time() - time_stamp_epoch):.4f} seg")

        # Evaluate parents
        model.eval()
        eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, args.device)
        print(f"Tiempo total de entrenamiento/validacion {args.epochs} (1): {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} horas")

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

        print(f'>>>>> size parents: {len(parents)}, size offsprings: {len(offsprings)}')
        # Evaluate offspring
        eval_population(model, mutation, valid_queue, args, criterion, attack_f, weights_r2, args.device)
        print(f"Tiempo total de entrenamiento/validacion {args.epochs}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} horas")

        archive = archive_update_pq(archive, Population.merge(pop, mutation))
        pop = update_population_r2(pop, mutation, weights_r2)

    return model, archive

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
