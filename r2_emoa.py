import multiprocessing as mp
import time
from collections import defaultdict
import copy
import random
import ssl
import torch, torchvision
from rnas_train import train_individual, infer
from micro_space.model_search import alphas_to_genotype
import gc

import numpy as np
import time

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy, dominates
from micro_space.micro_encoding import PRIMITIVES, convert, decode
from individual import Individual
from evolutionary import tournament_selection, binary_crossover, polynomial_mutation, point_crossover

from indicators import contribution_r2, update_ref_points


"""
 python3 rnas_search.py --seed 18906049 --algorithm r2-emoa --dataset cifar10 --batch_size 32  \
 --n_population 10 --epochs_train_individual 2 --generations 2 \
 --prob_cross 0.9 --prob_mut 0.1 --eta_cross 15 --eta_mut 20 --mu 0.1 \
 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 \
 --report_freq 50 --gpu 0 --init_channels 16 --reduction True --layers 5 --steps 6 --multiplier 6 \
 --attack FGSM --fgsm_eps 8/255 --cutout False --cutout_length 16 --drop_path_prob 0.3 \
 --grad_clip 0.5 --train_portion 0.5
"""
def prepare_args_standard(args):

    np.random.seed(args.seed)
    random.seed(args.seed)

    ssl._create_default_https_context = ssl._create_unverified_context

    weights_r2 = utils.get_weights_r2(args.n_population)

    k = sum(2 + i for i in range(args.steps))
    num_ops = len(PRIMITIVES)
    alphas_dim = (k, num_ops)

    return alphas_dim, weights_r2

def initial_population(n_population, alphas_dim, k, args):
    individuals = []
    for i in range(n_population):
        if args.search_space == 'discrete':
            # Each group of 4 integers represents two operations for a given node (op1, from_node1, op2, from_node2)
            n_var = 4 * args.steps * 2
            n_ops = len(PRIMITIVES)
            flattened = np.zeros(n_var, dtype=np.int32)
            h = 1
            for b in range(0, n_var // 2, 4):
                flattened[b] = np.random.randint(0, n_ops)
                flattened[b+1] = np.random.randint(0, h + 1)
                flattened[b+2] = np.random.randint(0, n_ops)
                flattened[b+3] = np.random.randint(0, h + 1)
                h += 1
            flattened[n_var // 2:] = flattened[:n_var // 2]
        else:
            flattened = np.random.rand(alphas_dim[0]*alphas_dim[1]*2)
        individuals.append(Individual(X=flattened.copy(), k=k, search_space=args.search_space))
    return individuals

def get_model_from_individual(individual_X, args):
    from micro_space.model import NetworkCIFAR
    from micro_space.model_search import alphas_to_genotype
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.search_space == 'continuous':
        k = sum(2 + i for i in range(args.steps))
        alphas_dim = (k, len(PRIMITIVES))
        genotype = alphas_to_genotype(individual_X, alphas_dim, args)
    else:
        genome = convert(individual_X)
        genotype = decode(genome, args.steps, args.multiplier)

    model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs_train_individual, eta_min=args.learning_rate_min)
    flops, params = utils.get_model_metrics(model)

    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=False, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=False, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96
        num_train = split + 96

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    criterion = torch.nn.CrossEntropyLoss()

    return model, optimizer, scheduler, flops, params, train_queue, valid_queue, criterion

def sanity_check_individual(model):
    seen = defaultdict(list)
    for name, module in model.named_modules():
        seen[id(module)].append(name)

    for module_id, names in seen.items():
        if len(names) > 1:
            print("Shared module detected:")
            for n in names:
                print("   ", n)
            raise RuntimeError("Shared module detected.")

def worker_evaluate_individual(gen, i, individual_X, pop_len, weight_individual, nadir_point, ideal_point, args, return_dict):
    try:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        args.device = device
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(args.seed + i) # Different seed for each process to avoid identical weight initialization
            torch.backends.cudnn.enabled = True
        model, optimizer, scheduler, individual_flops, individual_params, train_queue, valid_queue, criterion = get_model_from_individual(
            individual_X,
            args)
        time_training = time.time()
        train_individual(model, individual_flops, individual_params, train_queue, criterion, optimizer, args,
                         weight_individual, nadir_point, ideal_point, scheduler)
        print(
            f'Gen {gen} Training {i + 1}/{pop_len} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_training))} (HH:MM:SS)')

        time_evaluation = time.time()
        std_acc, adv_acc, std_loss, adv_loss = infer(valid_queue, model, criterion, args)
        print(
            f'Gen {gen} Evaluation {i + 1}/{pop_len} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_evaluation))} (HH:MM:SS) std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, std_loss {std_loss:.4f}, adv_loss {adv_loss:.4f} ,flops {individual_flops:.2f}, params {individual_params:.2f}')
        assert np.isfinite(std_acc) and np.isfinite(adv_acc) and np.isfinite(std_loss) and np.isfinite(adv_loss), f"Non-finite evaluation results for individual {i} of generation {gen}: std_acc {std_acc}, adv_acc {adv_acc}, std_loss {std_loss}, adv_loss {adv_loss}"

        if args.search_space == 'continuous':
            k = sum(2 + i for i in range(args.steps))
            alphas_dim = (k, len(PRIMITIVES))
            genotype = alphas_to_genotype(individual_X, alphas_dim, args)
        else:
            genome = convert(individual_X)
            genotype = decode(genome, args.steps, args.multiplier)

        return_dict[i] = {
            "std_acc": std_acc,
            "adv_acc": adv_acc,
            "std_loss": std_loss,
            "adv_loss": adv_loss,
            "flops": individual_flops,
            "params": individual_params,
            "genotype": genotype
        }
    except Exception as e:
        print(f"Error in evaluating individual {i} of generation {gen}: {e}")
    finally:
        del model, optimizer, scheduler, train_queue, valid_queue, criterion
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def evaluate_population_multiprocessing(gen, pop, weights_r2, nadir_point, ideal_point, args):
    with mp.Manager() as manager:
        return_dict = manager.dict()

        pop_len = len(pop)
        to_remove = []
        for i, individual in enumerate(pop):
            weight_individual = weights_r2[len(pop)][i].copy()
            args_individual = copy.deepcopy(args)
            p = mp.Process(
                target=worker_evaluate_individual,
                args=(gen, i, individual.X.copy(), pop_len, weight_individual, nadir_point.copy(), ideal_point.copy(), args_individual, return_dict)
            )
            p.start()
            p.join(timeout=args.timestamp*60)  # 10 args.timestamp timeout per individual
            
            if p.is_alive():
                print(f"Timeout: Individual {i} of generation {gen} took too long to evaluate and will be removed from the population.")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
                to_remove.append(i)
                time.sleep(8)
                continue
            time.sleep(10)    

            if p.exitcode != 0 or i not in return_dict:
                # If the process did not finish successfully, we skip this individual in the population update
                print(f"Error: Individual {i} of generation {gen} did not finish successfully (exit code {p.exitcode}) and will be removed from the population")
                to_remove.append(i)
                time.sleep(10)
            elif not p.is_alive() and p.exitcode == 0:
                # Update individual's results from the return_dict
                res = return_dict.get(i)
                if res is not None:
                    individual.std_acc = res["std_acc"]
                    individual.adv_acc = res["adv_acc"]
                    individual.F[args.std_loss_index] = res["std_loss"]
                    individual.F[args.adv_loss_index] = res["adv_loss"]
                    individual.F[args.flops_index] = res["flops"]
                    individual.F[args.params_index] = res["params"]
                    individual.genotype = res["genotype"]
                else:
                    to_remove.append(i)    
            else:
                to_remove.append(i)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


        for i in sorted(to_remove, reverse=True):
            print(f"Removing individual {i} from generation {gen} due to evaluation failure.")
            del pop[i]
        assert len(pop) == pop_len - len(to_remove), f"Expected population size {pop_len - len(to_remove)}, but got {len(pop)} after removing failed evaluations."
        return len(to_remove)

# R2 version where each architecture has its own weights (no supernet training). This is a baseline to compare with the supernet version.
def r2_emoa_rnas(args):
    alphas_dim, weights_r2 = prepare_args_standard(args)
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    nadir_point = np.ones(4,)
    ideal_point = np.zeros(4,)
    time_search = time.time()
    pop = initial_population(args.n_population, alphas_dim, args.objectives, args)
    print(f">>>> Initial population of size {len(pop)} created.")
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
    individuals_failed = 0
    individuals_failed += evaluate_population_multiprocessing(0, pop, weights_r2, nadir_point, ideal_point, args)
    update_ref_points(pop, nadir_point, ideal_point)

    archive = archive_update_pq(archive, pop)
    archive_losses = archive_update_pq(archive_losses, pop, k=2)
    hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                         weights_r2, statistics)
    print(f">>>> Gen 0 | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    for generation in range(args.generations):
        time_stamp_gen = time.time()

        parents = tournament_selection(pop, n_select=args.n_population // 2, tournament_size=5)
        if args.search_space == 'discrete':
            offsprings = point_crossover(parents, n_childs=args.n_population, prob_cross=args.prob_cross)
        else:
            offsprings = binary_crossover(parents, n_childs=args.n_population, eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        individuals_failed += evaluate_population_multiprocessing(generation+1, mutation, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += args.n_population
        update_ref_points(mutation, nadir_point, ideal_point)

        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(args.n_population, pop, mutation, weights_r2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                             weights_r2, statistics)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        print(f">>>> Gen {generation + 1} | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        print(
            f">>>> Gen {generation + 1} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
    print(f">>>> Total architectures evaluated: {architectures_evaluated}, Individuals failed during training // evaluation: {individuals_failed}")
    print(
        f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return archive, archive_accuracy, archive_losses, statistics


def non_dominated_sort(population):
    N = len(population)
    S = [[] for _ in range(N)] # solutions dominated by i
    n = [0] * N # number of solutions dominating i
    fronts = [[]]
    print(f"DEBUG: type of n is {type(n)}") 
    print(f"DEBUG: type of N is {type(N)}")
    print(f"DEBUG: type of population is {type(population)}")
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
    fronts = non_dominated_sort(c)
    last_front = len(fronts) - 1
    while len(c) > n:
        weights = weights_r2[len(c)]
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
        z_ref = np.min([ind.F for ind in front_k], axis=0)
        nadir_point = np.max([ind.F for ind in front_k], axis=0)
        for ind in front_k:
            ind.c_r2 = contribution_r2(front_k, ind, weights, nadir_point, z_ref)
            print(f"Individual {ind.F} R2 contribution {ind.c_r2}")
        worst = sorted(front_k, key=lambda x: x.c_r2)[0]
        c.remove(worst)
        front_k.remove(worst)
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return c
