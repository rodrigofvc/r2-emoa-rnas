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

def get_genotypes_from_archive(archs_path, args):
    with open(archs_path, 'r') as f:
        population_data = json.load(f)
    genotypes = []
    if args.algorithm == 'r2-emoa' or args.algorithm == 'r2-emoa-one-shot' or args.algorithm == 'cars':
        pop = [create_from_json(ind_json, args.search_space) for ind_json in population_data['population']]
        pop = archive_update_pq([], pop, k=4)
        for p in pop:
            genotype_dict = p.genotype
            genotype = Genotype(normal=genotype_dict[0],
                                normal_concat=genotype_dict[1],
                                reduce=genotype_dict[2],
                                reduce_concat=genotype_dict[3])
            genotypes.append(genotype)
    elif args.algorithm == 'nsganet' or args.algorithm == 'nevonas':
        if 'archive_genotype' in population_data.keys():
            genotypes_data = population_data['archive_genotype']
            for genotype_dict in genotypes_data:
                genotype = Genotype(normal=genotype_dict[0],
                                    normal_concat=genotype_dict[1],
                                    reduce=genotype_dict[2],
                                    reduce_concat=genotype_dict[3])
                genotypes.append(genotype)
        else:
            pop_X = population_data['pop_X']
            pop_F = population_data['pop_obj']
            pop = [create_from_json({'X': genome, 'F': obj, 'k': 4, 'feasible': True, 'c_r2': 0,  'std_acc': 0, 'adv_acc': 0, 'genotype': genome}, args.search_space) for genome, obj in zip(pop_X, pop_F)]
            pop = archive_update_pq([], pop, k=4)
            for ind in pop:
                genome = ind.X
                if args.algorithm == 'nsganet':
                    genome = convert(genome)
                    genotype = decode(genome, args.steps, args.multiplier)
                else:
                    k = sum(2 + i for i in range(args.steps))
                    alphas_dim = (k, len(PRIMITIVES))
                    genome = np.array(genome, dtype=np.float32)
                    genotype = alphas_to_genotype(genome, alphas_dim, args)
                genotypes.append(genotype)
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented for loading architectures.")
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