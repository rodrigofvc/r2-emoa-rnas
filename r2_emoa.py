import gc
import time
from collections import defaultdict

import numpy as np

import utils
from archivers import archive_update_pq, archive_update_pq_accuracy, dominates
from micro_space import micro_encoding
from micro_space.micro_encoding import PRIMITIVES
from micro_space.model_search import discretize, Network
from micro_space.model import NetworkCIFAR
from individual import Individual
from rnas_train import run_batch_epoch, train_individual, infer
from evolutionary import unpack_alphas, tournament_selection, binary_crossover, polynomial_mutation, point_crossover
import torch


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
    gc.collect()
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

def eval_individual(individual, model, valid_queue, args, criterion, attack_f):
    #attack = attack_f(model)
    std_acc, adv_acc, std_loss, adv_loss = infer(valid_queue, model, criterion, attack_f, args)
    individual.std_acc = std_acc
    individual.adv_acc = adv_acc
    individual.F[args.std_loss_index] = std_loss
    individual.F[args.adv_loss_index] = adv_loss
    model_flops, model_parameters = utils.get_model_metrics(None, model, discrete=True)
    individual.F[args.flops_index] = model_flops
    individual.F[args.params_index] = model_parameters
    print(f"Evaluation: std_acc {std_acc:.2f}%, adv_acc {adv_acc:.2f}%, std_loss {std_loss:.4f}, adv_loss {adv_loss:.4f} ,flops {model_flops:.2f}, params {model_parameters}")

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
    model_flops, model_parameters = utils.get_model_metrics(model.genotype(), model)
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

def get_model_from_individual(individual, args):
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    if args.search_space == 'continuous':
        continuous_model = Network(
            C=args.init_channels,
            num_classes=n_classes,
            layers=args.layers,
            criterion=torch.nn.CrossEntropyLoss(),
            steps=args.steps,
            multiplier=args.multiplier,
            stem_multiplier=3,
            device=args.device,
        ).to(args.device)
        individual_architect = unpack_alphas(individual.X, continuous_model.alphas_dim, args)
        continuous_model.update_arch_parameters(individual_architect)
        discrete = discretize(individual_architect, continuous_model.genotype(), args.device)
        continuous_model.update_arch_parameters(discrete)
        genotype = continuous_model.genotype()
        del continuous_model
    else:
        genome = micro_encoding.convert(individual.X)
        genotype = micro_encoding.decode(genome, args.steps, args.multiplier)
    individual.genotype = genotype

    gc.collect()
    if args.device.type == 'cuda' and args.synchronize:
        torch.cuda.empty_cache()
    model = NetworkCIFAR(args.init_channels, n_classes, args.layers, False, genotype).to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs_train_individual, eta_min=args.learning_rate_min)
    return model, optimizer, scheduler

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

# R2 version where each architecture has its own weights (no supernet training). This is a baseline to compare with the supernet version.
def r2_emoa_rnas(args, alphas_dim, train_queue, valid_queue, attack_f, weights_r2):
    archive = []
    archive_accuracy = []
    archive_losses = []
    architectures_evaluated = 0
    nadir_point = torch.ones(4, device=args.device)
    ideal_point = torch.zeros(4, device=args.device)
    time_search = time.time()
    pop = initial_population(args.n_population, alphas_dim, args.objectives, args)
    print(f">>>> Initial population of size {len(pop)} created.")
    statistics = {'max_f1': 0, 'max_f2': 0, 'max_f3': 0, 'max_f4': 0, 'min_f1': float('inf'), 'min_f2': float('inf'),
                  'min_f3': float('inf'), 'min_f4': float('inf'), 'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
    for i, individual in enumerate(pop):
        criterion = torch.nn.CrossEntropyLoss()
        model, optimizer, scheduler = get_model_from_individual(individual, args)
        weight_individual = torch.tensor(weights_r2[len(pop)][i], device=args.device, dtype=torch.float32)
        time_training = time.time()
        try:
            sanity_check_individual(model)
            train_individual(model, train_queue, criterion, optimizer, attack_f, args, weight_individual, nadir_point, ideal_point, scheduler)
            print(f'Gen 0 Training {i+1}/{len(pop)} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_training))} (HH:MM:SS)')
            time_evaluation = time.time()
            eval_individual(individual, model, valid_queue, args, criterion, attack_f)
            print(f'Gen 0 Evaluation {i+1}/{len(pop)} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_evaluation))} (HH:MM:SS)')
        except RuntimeError as e:
            print(f"Error training/evaluating individual {i} in generation 0: {e.__str__()}")
            individual.std_acc = 0
            individual.adv_acc = 0
            individual.F[args.std_loss_index] = float('inf')
            individual.F[args.adv_loss_index] = float('inf')
            individual.F[args.flops_index] = 1000
            individual.F[args.params_index] = 1000
        finally:
            model.cpu()
            del model, optimizer, scheduler, criterion, weight_individual
        gc.collect()
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(0.1)
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

        for i, individual in enumerate(mutation):
            model, optimizer, scheduler = get_model_from_individual(individual, args)
            criterion = torch.nn.CrossEntropyLoss()
            weight_individual = torch.tensor(weights_r2[len(pop)][i], device=args.device, dtype=torch.float32)
            time_training = time.time()
            try:
                sanity_check_individual(model)
                train_individual(model, train_queue, criterion, optimizer, attack_f, args, weight_individual, nadir_point,
                                 ideal_point, scheduler)
                print(f'Gen {generation + 1} Training {i+1}/{len(mutation)} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_training))} (HH:MM:SS)')
                time_evaluation = time.time()
                eval_individual(individual, model, valid_queue, args, criterion, attack_f)
                print(f'Gen {generation + 1} Evaluation {i+1}/{len(mutation)} done in {time.strftime("%H:%M:%S", time.gmtime(time.time() - time_evaluation))} (HH:MM:SS)')
            except RuntimeError as e:
                print(f"Error training/evaluating individual {i} in generation {generation + 1}: {e}")
                individual.std_acc = 0
                individual.adv_acc = 0
                individual.F[args.std_loss_index] = float('inf')
                individual.F[args.adv_loss_index] = float('inf')
                individual.F[args.flops_index] = 1000
                individual.F[args.params_index] = 1000
            finally:
                model.cpu()
                del model, optimizer, scheduler, criterion, weight_individual
            gc.collect()
            if args.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                time.sleep(0.1)
        architectures_evaluated += len(mutation)
        update_ref_points(pop, nadir_point, ideal_point)

        archive = archive_update_pq(archive, pop + mutation)
        archive_accuracy = archive_update_pq_accuracy(archive_accuracy, pop + mutation)
        archive_losses = archive_update_pq(archive_losses, pop + mutation, k=2)
        pop = update_population_r2(pop, mutation, weights_r2)
        hyp_archive, hyp_2, r2_archive = utils.store_metrics(architectures_evaluated, archive, archive_losses, args,
                                                             weights_r2, statistics)
        utils.save_architectures(archive, args.save_path_final_architect)
        utils.plot_hypervolume(statistics, args.save_path_final_architect)
        utils.plot_hypervolume2(statistics, args.save_path_final_architect)
        utils.plot_r2(statistics, args.save_path_final_architect)
        print(f">>>> Gen {generation + 1} | Hypervolume (4 objs): {hyp_archive}, Hypervolume (2 objs): {hyp_2}, R2: {r2_archive}")
        print(
            f">>>> Gen {generation + 1} DONE in {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))} (HH:MM:SS)")
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

def update_population_r2(pop, offspring, weights_r2):
    c = pop + offspring
    n = len(pop)
    assert len(c) >= 2*n
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
