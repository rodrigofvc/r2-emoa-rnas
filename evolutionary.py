import logging
import random as rd
import numpy as np
from pymoo.operators.mutation.pm import mut_pm

from individual import Individual
from indicators import contribution_r2
from archivers import dominates
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

def tournament_selection(pop_, n_select, tournament_size=5):
    winners = []
    pop = [ind for ind in pop_ if ind.feasible]
    while len(winners) < n_select:
        P = rd.sample(pop, k=tournament_size)
        winner = min(P, key=lambda ind: ind.c_r2)
        if winner not in winners:
            winners.append(winner)
    return winners

def point_crossover(parents, n_childs, prob_cross):
    offsprings = []
    while len(offsprings) < n_childs:
        parent1 = rd.choice(parents)
        parent2 = rd.choice(parents)
        if np.random.rand() < prob_cross and not np.array_equal(parent1.X, parent2.X):
            point = rd.randint(1, parent1.X.shape[0] - 1)
            child1_X = np.concatenate((parent1.X[:point], parent2.X[point:]))
            child2_X = np.concatenate((parent2.X[:point], parent1.X[point:]))
            offsprings.append(Individual(X=child1_X.copy(), k=parent1.k, search_space='discrete'))
            if len(offsprings) < n_childs:
                offsprings.append(Individual(X=child2_X.copy(), k=parent2.k, search_space='discrete'))
    return offsprings

def binary_crossover(pop, n_childs, eta, prob_cross):
    offsprings = []
    n_var = pop[0].X.shape[0]
    while len(offsprings) < n_childs:
        parent1 = rd.choice(pop)
        parent2 = rd.choice(pop)
        if np.random.rand() < prob_cross and not np.array_equal(parent1.X, parent2.X):
            child1_X = np.empty(n_var, dtype=np.float32)
            child2_X = np.empty(n_var, dtype=np.float32)
            for j in range(n_var):
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                child1_X[j] = 0.5 * ((1 + beta) * parent1.X[j] + (1 - beta) * parent2.X[j])
                child2_X[j] = 0.5 * ((1 - beta) * parent1.X[j] + (1 + beta) * parent2.X[j])
            offsprings.append(Individual(X=child1_X.copy(), k=parent1.k, search_space='continuous'))
            if len(offsprings) < n_childs:
                offsprings.append(Individual(X=child2_X.copy(), k=parent2.k, search_space='continuous'))
    return offsprings

def polynomial_mutation(pop, prob_mut, eta, random_state, steps, n_ops, search_space, at_least_once=False):
    if len(pop) == 0:
        return pop

    X = np.asarray([individual.X for individual in pop], dtype=float)

    if X.ndim != 2:
        raise ValueError("The decision matrix must have shape (n_individuals, n_variables).")

    n_individuals, n_variables = X.shape

    genes_per_node = 4
    genes_per_cell = genes_per_node * steps

    # Normally 2: normal cell and reduction cell
    n_cells = n_variables // genes_per_cell

    xl = []
    xu = []

    if search_space == 'discrete':
        for _ in range(n_cells):
            for node in range(steps):
                # [op1, input1, op2, input2]
                xl.extend([0, 0, 0, 0])
                xu.extend([n_ops - 1, node + 1, n_ops - 1, node + 1])
        xl = np.array(xl, dtype=float)
        xu = np.array(xu, dtype=float)
    else:
        xl = np.zeros(n_variables, dtype=float)
        xu = np.ones(n_variables, dtype=float)

    # mut_pm expects one eta value per individual
    eta_values = np.full(n_individuals, eta, dtype=float)

    prob_values = np.full(n_individuals, prob_mut, dtype=float)

    X_mutated = mut_pm(X=X, xl=xl, xu=xu, eta=eta_values, prob=prob_values, at_least_once=at_least_once, random_state=random_state)

    if search_space == 'discrete':
        X_mutated = np.rint(X_mutated)
        X_mutated = np.clip(X_mutated, xl, xu)
        X_mutated = X_mutated.astype(int)

    for individual, x_mutated in zip(pop, X_mutated):
        individual.X = x_mutated.copy()

    return pop

def polynomial_mutation_dep(pop, prob_mut, eta):
    xl = np.zeros_like(pop[0].X)
    xu = np.ones_like(pop[0].X)
    for individual in pop:
        for i in range(individual.X.shape[0]):
            if np.random.rand() < prob_mut:
                u = np.random.rand()
                delta = 0.0
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                individual.X[i] = individual.X[i] + delta * (xu[i] - xl[i])
                individual.X[i] = np.clip(individual.X[i], 0, 1)
    return pop

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
    z_ref = np.min([ind.F for ind in c], axis=0)
    nadir_point = np.max([ind.F for ind in c], axis=0)
    logging.info('z_ref %s', z_ref)
    logging.info('nadir point %s', nadir_point)
    weights = weights_r2[n]
    while len(c) > n:
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
        for ind in front_k:
            ind.c_r2 = contribution_r2(front_k, ind, weights, nadir_point, z_ref)
            logging.info(f"Individual {ind.F} R2 contribution {ind.c_r2}")
        worst = min(front_k, key=lambda x: x.c_r2)
        tied_worst = [ind for ind in front_k if np.allclose(ind.c_r2, worst.c_r2, rtol=1e-10, atol=1e-12)]
        if len(tied_worst) > 1:
            crowding_distances = get_crowding_distances(front_k)
            logging.info('Crowding distances: %s', [(ind.F, cd) for ind, cd in crowding_distances])
            worst = min(tied_worst, key=lambda x: next(cd for ind, cd in crowding_distances if ind == x))
        logging.info(f"removed individual {worst.F} with R2 contribution {worst.c_r2}")
        c.remove(worst)
        front_k.remove(worst)
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return c

def get_crowding_distances(front):
    F = np.array([ind.F for ind in front])
    rank_and_crowding = RankAndCrowding(crowding_func="cd")
    crowding_function = rank_and_crowding.crowding_func.do(F)
    crowding_distances = []
    for ind, cd in zip(front, crowding_function):
        crowding_distances.append((ind, cd))
    return crowding_distances
