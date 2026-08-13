import random as rd
import numpy as np
from individual import Individual
from indicators import contribution_r2
from archivers import dominates

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

def polynomial_mutation(pop, prob_mut, eta):
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
    print('z_ref', z_ref)
    print('nadir point', nadir_point)
    weights = weights_r2[n]
    while len(c) > n:
        #weights = weights_r2[len(c)]
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
            print(f"Individual {ind.F} R2 contribution {ind.c_r2}")
        worst = min(front_k, key=lambda x: x.c_r2)
        print('removed individual', worst.F, 'with R2 contribution', worst.c_r2)
        c.remove(worst)
        front_k.remove(worst)
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return c
