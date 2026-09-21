import json
import os
import torch
import numpy as np

from archivers import archive_update_pq
from individual import create_from_json
from micro_space.micro_encoding import PRIMITIVES, convert, decode, Genotype
from micro_space.model_search import alphas_to_genotype


def save_model(model, model_path, name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path += os.sep + name
    torch.save(model, model_path)

def genotype_key(individual, search_space, decimals=12):
    X = np.asarray(individual.X)
    if search_space == "discrete":
        return tuple(X.astype(np.int64).ravel())
    return tuple(np.round(X.astype(np.float64).ravel(), decimals=decimals))


def get_genotypes_from_archive(archs_path, search_space):
    with open(archs_path, "r") as file:
        population_data = json.load(file)

    original_population = [create_from_json(ind_json, search_space) for ind_json in population_data["archive"]]

    # Keep only valid evaluated individuals.
    valid_population = [individual for individual in original_population if (individual.feasible and individual.F is not None and np.isfinite(individual.F).all())]

    unique_population = []
    seen_genotypes = set()

    for individual in valid_population:
        key = genotype_key(individual, search_space)

        if key in seen_genotypes:
            continue

        seen_genotypes.add(key)
        unique_population.append(individual)

    nondominated_population = archive_update_pq(archive=[], population_=unique_population, k=4)

    genotypes = []

    for individual in nondominated_population:
        genotype_dict = individual.genotype

        genotype = Genotype(
            normal=genotype_dict[0],
            normal_concat=genotype_dict[1],
            reduce=genotype_dict[2],
            reduce_concat=genotype_dict[3],
        )

        genotypes.append(genotype)

    if not genotypes:
        raise ValueError("No valid nondominated genotypes found in the archive.")

    print("Original archive:", len(original_population))
    print("Valid individuals:", len(valid_population))
    print("Unique genotypes:", len(unique_population))
    print("Unique nondominated genotypes:", len(nondominated_population))
    print("Repeated genotypes removed:", len(valid_population) - len(unique_population))

    return genotypes

def get_genotypes_from_archive_dep(archs_path, args):
    with open(archs_path, 'r') as f:
        population_data = json.load(f)
    genotypes = []

    pop = [create_from_json(ind_json, args.search_space) for ind_json in population_data['archive']]
    n_pop = len(pop)
    pop = archive_update_pq([], pop, k=4)
    assert len(pop) == n_pop, "Archive update changed the number of individuals."
    for p in pop:
        genotype_dict = p.genotype
        genotype = Genotype(normal=genotype_dict[0],
                                normal_concat=genotype_dict[1],
                                reduce=genotype_dict[2],
                                reduce_concat=genotype_dict[3])
        genotypes.append(genotype)
    assert len(genotypes) > 0, "No genotypes found in the archive."
    return genotypes

def get_best_genotype_adversarial(archs_path, args):
    best_adv_loss = 100
    best_individual = None
    if args.algorithm == 'r2-emoa' or args.algorithm == 'r2-emoa-one-shot' or args.algorithm == 'cars':
        with open(archs_path, 'r') as f:
            population_data = json.load(f)
        pop = [create_from_json(ind_json, args.search_space) for ind_json in population_data['population']]

        for p in pop:
            if p.F[1] < best_adv_loss:
                best_adv_loss = p.F[1]
                best_individual = p
        genotype_dict = best_individual.genotype
        genotype = Genotype(normal=genotype_dict[0],
                            normal_concat=genotype_dict[1],
                            reduce=genotype_dict[2],
                            reduce_concat=genotype_dict[3])
        return genotype
    elif args.algorithm == 'nsganet' or args.algorithm == 'nevonas':
        with open(archs_path, 'r') as f:
            population_data = json.load(f)
        if 'archive_genotype' in population_data.keys():
            genotypes = population_data['archive_genotype']
            archive_obj = population_data['archive_objectives']
            for (genotype_dict, obj) in zip(genotypes, archive_obj):
                if obj[1] < best_adv_loss:
                    best_adv_loss = obj[1]
                    best_individual = genotype_dict
            genotype = Genotype(normal=best_individual[0],
                                normal_concat=best_individual[1],
                                reduce=best_individual[2],
                                reduce_concat=best_individual[3])
            return genotype
        else:
            pop_X = population_data['pop_X']
            pop_F = population_data['pop_obj']
            for (genome, obj) in zip(pop_X, pop_F):
                if obj[1] < best_adv_loss:
                    best_adv_loss = obj[1]
                    best_individual = genome
            if args.algorithm == 'nsganet':
                genome = convert(best_individual)
                genotype = decode(genome, args.steps, args.multiplier)
            else:
                k = sum(2 + i for i in range(args.steps))
                alphas_dim = (k, len(PRIMITIVES))
                best_individual = np.array(best_individual, dtype=np.float32)
                genotype = alphas_to_genotype(best_individual, alphas_dim, args)
            return genotype
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented for loading best architecture.")