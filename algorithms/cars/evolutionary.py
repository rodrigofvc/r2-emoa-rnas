import random

import numpy as np
import torch
from individual import Individual

def tournament_selection(pop, n_select, tournament_size=5):
    winners = []
    while len(winners) < n_select:
        P = random.sample(pop, k=tournament_size)
        winner = min(P, key=lambda ind: ind.c_r2)
        if winner not in winners:
            winners.append(winner)
    return winners

def point_crossover(parents, n_childs, prob_cross):
    offsprings = []
    while len(offsprings) < n_childs:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        if np.random.rand() < prob_cross and not np.array_equal(parent1.X, parent2.X):
            point = random.randint(1, parent1.X.shape[0] - 1)
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
        parent1 = random.choice(pop)
        parent2 = random.choice(pop)
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

