import json
import lzma
import os
import pickle
import torch

from individual import create_from_json
from micro_space.micro_encoding import PRIMITIVES, convert, decode
from micro_space.model_search import alphas_to_genotype


def save_model(model, model_path, name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path += os.sep + name
    torch.save(model, model_path)


def get_best_genotype_adversarial(archs_path, args):
    best_adv_loss = 100
    best_individual = None
    if args.algorithm == 'r2-emoa' or args.algorithm == 'r2-emoa-one-shot':
        with open(archs_path, 'r') as f:
            population_data = json.load(f)
        pop = [create_from_json(ind_json, args.search_space) for ind_json in population_data['population']]

        for p in pop:
            if p.F[1] < best_adv_loss:
                best_adv_loss = p.F[1]
                best_individual = p

        if args.search_space == 'continuous':
            k = sum(2 + i for i in range(args.steps))
            alphas_dim = (k, len(PRIMITIVES))
            genotype = alphas_to_genotype(best_individual.X, alphas_dim, args)
        else:
            genome = convert(best_individual.X)
            genotype = decode(genome, args.steps, args.multiplier)
        return genotype

    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented for loading best architecture.")