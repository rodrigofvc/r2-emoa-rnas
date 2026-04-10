import json
import os
import subprocess
import sys
import time

import numpy as np

from micro_space.micro_encoding import Genotype


def worker_evaluate_individual(gen, i, individual_X, weight_individual, nadir_point, ideal_point, args, return_dict):
    log_file = "logs" + os.sep + f"worker_{gen}_{i}.log"
    result_file = "logs" + os.sep + f"result_gen{gen}_ind{i}.json"

    if args.algorithm == 'r2-emoa':
        file = 'individual_worker.py'
    elif args.algorithm == 'r2-emoa-one-shot':
        file = 'individual_worker_one_shot.py'
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    process_args = [
        sys.executable, "-X dev -u", file,
        '--gen', str(gen),
        '--i', str(i),
        '--seed', str(args.seed + i),  # Different seed for each process to avoid identical weight initialization
        '--individual_x', np.array2string(individual_X, separator=',', max_line_width=np.inf),
        '--search_space', str(args.search_space),
        '--dataset', str(args.dataset),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--epochs_train_individual', str(args.epochs_train_individual),
        '--data', str(args.data),
        '--mu', str(args.mu),
        '--learning_rate', str(args.learning_rate),
        '--learning_rate_min', str(args.learning_rate_min),
        '--momentum', str(args.momentum),
        '--weight_decay', str(args.weight_decay),
        '--init_channels', str(args.init_channels),
        '--layers', str(args.layers),
        '--steps', str(args.steps),
        '--multiplier', str(args.multiplier),
        '--fgsm_eps', str(args.fgsm_eps),
        '--cutout_length', str(args.cutout_length),
        '--drop_path_prob', str(args.drop_path_prob),
        '--grad_clip', str(args.grad_clip),
        '--train_portion', str(args.train_portion),
        '--weight_individual', np.array2string(weight_individual, separator=',', max_line_width=np.inf),
        '--nadir_point', np.array2string(nadir_point, separator=',', max_line_width=np.inf),
        '--ideal_point', np.array2string(ideal_point, separator=',', max_line_width=np.inf)
    ]
    if args.algorithm == 'r2-emoa-one-shot':
        process_args.append('--supernet_path')
        process_args.append(str(args.save_path_final_model) + os.sep + "super-net.pt")

    if args.reduction:
        process_args.append('--reduction')
    if args.cutout:
        process_args.append('--cutout')
    env_worker = os.environ.copy()
    env_worker['CUDA_VISIBLE_DEVICES'] = str(args.gpu) # Set the GPU device for the subprocess
    if args.debug_cuda:
        env_worker['CUDA_LAUNCH_BLOCKING'] = '1'
        env_worker['TORCH_USE_CUDA_DSA'] = '1'
    clean_file = False
    with open(log_file, 'w') as f_log:
        process = subprocess.Popen(process_args,
                                   stdout=f_log,
                                   stderr=f_log,
                                   text=True,
                                   env=env_worker)

        try:
            process.communicate(timeout=args.timestamp * 60)

            if process.returncode == 0 and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    res_dict = json.load(f)
                    res_dict['genotype'] = Genotype(**res_dict['genotype'])
                    return_dict[i] = res_dict
                os.remove(result_file)
                clean_file = True
                print(f"Gen {gen} Individual {i}: std_acc {return_dict[i]['std_acc']:.2f}, adv_acc {return_dict[i]['adv_acc']:.2f} std_loss {return_dict[i]['std_loss']:.3f}, adv_loss {return_dict[i]['adv_loss']:.3f}, flops {return_dict[i]['flops']:.2f}, params {return_dict[i]['params']:.2f}")
            else:
                print(f"Gen {gen} Individual {i} failed with return code {process.returncode}")

        except subprocess.TimeoutExpired:
            print(f"Individual {i} exceed timestamp, it will be removed from the population.")
            process.kill()
            process.communicate(timeout=10)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt received. Terminating individual {i} process.")
            process.terminate()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            sys.exit('Search interrupted by user.')
        except Exception as e:
            print(f"Failed {i}: {e}")
        finally:
            # wait a bit to ensure the process has terminated and released resources before starting the next one
            time.sleep(1)
            # set default values for failed individuals
            if i not in return_dict:
                return_dict[i] = {
                    "std_acc": 0.0,
                    "adv_acc": 0.0,
                    "std_loss": 1000.0,
                    "adv_loss": 1000.0,
                    "flops": 1000.0,
                    "params": 1000.0,
                    "genotype": None
                }
    # Remove log file of successful evaluations, keep logs of failed evaluations for debugging
    if clean_file and os.path.exists(log_file):
        os.remove(log_file)

def evaluate_population_multiprocessing(n_evaluations, gen, pop, weights_r2, nadir_point, ideal_point, args):
    return_dict = {}
    feasible_solutions = 0
    for i, individual in enumerate(pop):
        if feasible_solutions >= n_evaluations:
            # Skip the evaluation of the remaining individuals in the population if we have already evaluated enough feasible solutions
            break
        weight_individual = weights_r2[len(pop)][i].copy()
        worker_evaluate_individual(gen, i, individual.X.copy(),
                                   weight_individual, nadir_point.copy(),
                                   ideal_point.copy(), args, return_dict)
        individual.std_acc = float(return_dict[i]["std_acc"])
        individual.adv_acc = float(return_dict[i]["adv_acc"])
        individual.F = np.array([
            float(return_dict[i]["std_loss"]),
            float(return_dict[i]["adv_loss"]),
            float(return_dict[i]["flops"]),
            float(return_dict[i]["params"])
        ], dtype=np.float64)
        individual.genotype = return_dict[i]["genotype"]
        individual.feasible = True if individual.genotype is not None else False
        if individual.feasible:
            feasible_solutions += 1
