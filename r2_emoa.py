import gc
import multiprocessing as mp
import time
from collections import defaultdict

import numpy as np

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy, dominates
from micro_space import micro_encoding
from micro_space.micro_encoding import PRIMITIVES
from micro_space.model_search import discretize, Network, alphas_to_genotype
from micro_space.model import NetworkCIFAR
from individual import Individual
from rnas_train import run_batch_epoch, train_individual, infer
from evolutionary import unpack_alphas, tournament_selection, binary_crossover, polynomial_mutation, point_crossover
import torch
import torchvision

from indicators import contribution_r2, update_ref_points


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
            flattened = torch.rand(alphas_dim[0]*alphas_dim[1]*2).detach().cpu().numpy()
        individuals.append(Individual(X=flattened.copy(), k=k, search_space=args.search_space))
    return individuals

def eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, device, statisctics):
    model.eval()
    objective_space = np.empty((len(pop), args.objectives))
    attack = attack_f(model)
    for i, individual in enumerate(pop):
        individual_architect = unpack_alphas(individual.X, model.alphas_dim, args)
        model.update_arch_parameters(individual_architect)
        discrete = discretize(individual_architect, model.genotype(), device)
        model.update_arch_parameters(discrete)
        time_stamp = time.time()
        std_acc, adv_acc, std_loss, adv_loss = infer(valid_queue, model, criterion, attack, args)
        individual.std_acc = std_acc
        individual.adv_acc = adv_acc
        individual.F[args.std_loss_index] = std_loss
        individual.F[args.adv_loss_index] = adv_loss
        model_flops, model_parameters = utils.get_model_metrics(model.genotype(), model)
        individual.F[args.flops_index] = model_flops
        individual.F[args.params_index] = model_parameters
        individual.genotype = model.genotype()
        individual.F_norm = np.zeros(args.objectives)
        objective_space[i, :] = individual.F
        print(f"Evaluation {i + 1}/{len(pop)}: std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, std_loss {std_loss:.4f}, adv_loss {adv_loss:.4f} ,flops {model_flops:.2f}, params {model_parameters} ({time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp))}) (HH:MM:SS)")
    utils.store_statisctics(statisctics, objective_space)
    return len(pop)

def eval_individual(model, valid_queue, args, criterion):
    std_acc, adv_acc, std_loss, adv_loss = infer(valid_queue, model, criterion, args)
    return std_acc, adv_acc, std_loss, adv_loss

def train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, attack_f, gen, scaler, args, r2_weights, nadir_point, ideal_point, warmup=False):
    model.train()
    attack = attack_f(model)
    if warmup:
        epochs = args.epochs_warmup
    else:
        epochs = args.epochs_train_supernet
    r2_weights_pop = r2_weights[len(pop)]
    r2_weights_pop = torch.tensor(r2_weights_pop, device=args.device, dtype=torch.float32)
    z_ref_stch = torch.zeros(4, device=args.device)
    assert r2_weights_pop.shape[0] == len(pop)
    model_flops, model_parameters = utils.get_model_metrics(model)
    model_flops, model_parameters = torch.tensor(float(model_flops), device=args.device), torch.tensor(
        float(model_parameters), device=args.device)
    for epoch in range(epochs):
        for n_batch, (input, target) in enumerate(train_queue):
            individual = pop[n_batch % args.n_population]
            individual_r2_weights = r2_weights_pop[n_batch % args.n_population]
            individual_architect = unpack_alphas(individual.X, model.alphas_dim, args)
            model.update_arch_parameters(individual_architect)
            discrete = discretize(individual_architect, model.genotype(), args.device)
            model.update_arch_parameters(discrete)
            time_stamp = time.time()
            std_acc, adv_acc, loss = run_batch_epoch(model, input, target, criterion, optimizer, attack, scaler, args, model_flops, model_parameters, individual_r2_weights, z_ref_stch, ideal_point, nadir_point)
            if n_batch % args.report_freq == 0:
                print(f'>>>> Gen {gen}/{args.generations} | Epoch {epoch}/{epochs} | Batch {n_batch}/{len(train_queue)} | Loss {loss:.4f} | Std Acc {std_acc:.2f}% | Adv Acc {adv_acc:.2f}% | Time {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_stamp))} (HH:MM:SS)')
        scheduler.step()
    update_ref_points(pop, nadir_point, ideal_point)

def r2_emoa_rnas_oneshot(args, train_queue, valid_queue, model, criterion, optimizer, scheduler, attack_f, weights_r2):
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    time_search = time.time()
    pop = initial_population(args.n_population, model.alphas_dim, args.objectives, args)
    print(f">>>> Initial population of size {len(pop)} created.")
    scaler = None
    nadir_point = torch.ones(4, device=args.device)
    ideal_point = torch.zeros(4, device=args.device)
    if args.epochs_warmup > 0:
        print(">>>> Warmup training of the supernet...")
        train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, attack_f, 0, scaler, args, weights_r2, nadir_point, ideal_point, warmup=True)
        print(">>>> Warmup training DONE.")
    train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, attack_f, 0, scaler, args, weights_r2, nadir_point, ideal_point)
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'), 'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': [], 'lr_log': []}
    eval_population(model, pop, valid_queue, args, criterion, attack_f, weights_r2, args.device, statistics)
    archive = archive_update_pq(archive, pop)
    archive_losses = archive_update_pq(archive_losses, pop, k=2)
    hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args, weights_r2, statistics)
    print(f"Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    for generation in range(args.generations):
        start = time.time()
        time_stamp_epoch = time.time()
        train_supernet(pop, train_queue, model, criterion, optimizer, scheduler, attack_f, generation + 1, scaler, args, weights_r2, nadir_point, ideal_point)
        print(f">>>> Gen {generation + 1} training DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_epoch))} (HH:MM:SS)")

        parents = tournament_selection(pop, n_select=len(pop)//2, tournament_size=5)
        offsprings = binary_crossover(parents, n_childs=len(pop), eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        # Evaluate offspring
        architectures_evaluated += eval_population(model, mutation, valid_queue, args, criterion, attack_f, weights_r2, args.device, statistics)
        print(f"Tiempo total de entrenamiento/validacion {args.generations}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))} (HH:MM:SS)")

        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(pop, mutation, weights_r2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args, weights_r2, statistics)
        utils.save_model(model, args.save_path_final_model, f"super-net.pt")
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        statistics['lr_log'].append(optimizer.param_groups[0]['lr'])
        print(f"Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
    print(f">>>> Total search time: ({(time.time() - time_search) // 86400:02.0f}:{time.strftime('%H:%M:%S)', time.gmtime(time.time() - time_search))} (DD:HH:MM:SS)")
    return model, archive, archive_accuracy, archive_losses, statistics

def get_model_from_individual(individual_X, args):
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
        genome = micro_encoding.convert(individual_X)
        genotype = micro_encoding.decode(genome, args.steps, args.multiplier)

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
        split = 32
        num_train = split + 32

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
        model, optimizer, scheduler, individual_flops, individual_params, train_queue, valid_queue, criterion = get_model_from_individual(
            individual_X,
            args)
        time_training = time.time()
        train_individual(model, individual_flops, individual_params, train_queue, criterion, optimizer, args,
                         weight_individual, nadir_point, ideal_point, scheduler)
        print(
            f'Gen {gen} Training {i + 1}/{pop_len} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_training))} (HH:MM:SS)')

        time_evaluation = time.time()
        std_acc, adv_acc, std_loss, adv_loss = eval_individual(model, valid_queue, args, criterion)
        print(
            f'Gen {gen} Evaluation {i + 1}/{pop_len} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_evaluation))} (HH:MM:SS) std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, std_loss {std_loss:.4f}, adv_loss {adv_loss:.4f} ,flops {individual_flops:.2f}, params {individual_params:.2f}')

        assert np.isfinite(std_acc) and np.isfinite(adv_acc) and np.isfinite(std_loss) and np.isfinite(adv_loss), f"Non-finite evaluation results for individual {i} of generation {gen}: std_acc {std_acc}, adv_acc {adv_acc}, std_loss {std_loss}, adv_loss {adv_loss}"

        return_dict[i] = {
            "std_acc": std_acc,
            "adv_acc": adv_acc,
            "std_loss": std_loss,
            "adv_loss": adv_loss,
            "flops": individual_flops,
            "params": individual_params,
        }
    except Exception as e:
        import traceback
        print(f"Error in evaluating individual {i} of generation {gen}: {e}")
        traceback.print_exc()
    finally:
        del model, optimizer, scheduler, train_queue, valid_queue, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_population_multiprocessing(gen, pop, weights_r2, nadir_point, ideal_point, args):
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    pop_len = len(pop)
    to_remove = []
    for i, individual in enumerate(pop):
        weight_individual = weights_r2[len(pop)][i]
        p = mp.Process(
            target=worker_evaluate_individual,
            args=(gen, i, individual.X, pop_len, weight_individual, nadir_point, ideal_point, args, return_dict)
        )
        p.start()
        p.join(timeout=10*60)  # 10 minutes timeout per individual

        if p.is_alive():
            print(f"Timeout: Individual {i} of generation {gen} took too long to evaluate and will be removed from the population.")
            p.terminate()
            p.join()
            to_remove.append(i)
            continue

        if p.exitcode != 0 or i not in return_dict:
            # If the process did not finish successfully, we skip this individual in the population update
            if p.exitcode == 11:
                print(f"Segmentation fault detected for individual {i} of generation {gen}, and will be removed from the population")
            else:
                print(f"Error: Individual {i} of generation {gen} did not finish successfully (exit code {p.exitcode}) and will be removed from the population")
            to_remove.append(i)
        else:
            # Update individual's results from the return_dict
            res = return_dict[i]
            individual.std_acc = res["std_acc"]
            individual.adv_acc = res["adv_acc"]
            individual.F[args.std_loss_index] = res["std_loss"]
            individual.F[args.adv_loss_index] = res["adv_loss"]
            individual.F[args.flops_index] = res["flops"]
            individual.F[args.params_index] = res["params"]

    for i in sorted(to_remove, reverse=True):
        print(f"Removing individual {i} from generation {gen} due to evaluation failure.")
        del pop[i]
    assert len(pop) == pop_len - len(to_remove), f"Expected population size {pop_len - len(to_remove)}, but got {len(pop)} after removing failed evaluations."
    return len(to_remove)

# R2 version where each architecture has its own weights (no supernet training). This is a baseline to compare with the supernet version.
def r2_emoa_rnas(args, alphas_dim, weights_r2):
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    nadir_point = np.ones(4, )
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

        parents = tournament_selection(pop, n_select=len(pop) // 2, tournament_size=5)
        if args.search_space == 'discrete':
            offsprings = point_crossover(parents, n_childs=len(pop), prob_cross=args.prob_cross)
        else:
            offsprings = binary_crossover(parents, n_childs=len(pop), eta=args.eta_cross, prob_cross=args.prob_cross)
        mutation = polynomial_mutation(offsprings, prob_mut=args.prob_mut, eta=args.eta_mut)

        individuals_failed += evaluate_population_multiprocessing(generation, mutation, weights_r2, nadir_point, ideal_point, args)
        architectures_evaluated += len(mutation)
        update_ref_points(pop, nadir_point, ideal_point)

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
    while len(c) > n:
        weights = weights_r2[len(c)]
        fronts = non_dominated_sort(c)
        front_k = fronts[-1]
        if len(front_k) == 1:
            worst = front_k[0]
            c.remove(worst)
            continue
        z_ref = np.min([ind.F for ind in front_k], axis=0)
        nadir_point = np.max([ind.F for ind in front_k], axis=0)
        for ind in front_k:
            ind.c_r2 = contribution_r2(front_k, ind, weights, nadir_point, z_ref)
            print(f"Individual {ind.F} R2 contribution {ind.c_r2}")
        worst = sorted(front_k, key=lambda x: x.c_r2)[0]
        c.remove(worst)
    assert len(c) == n, f"len(c)={len(c)}, n={n}"
    return c
