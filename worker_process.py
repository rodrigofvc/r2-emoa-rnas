import json
import os
import subprocess
import sys
import time
import logging

import numpy as np

from micro_space.micro_encoding import Genotype


def worker_evaluate_individual(gen, i, individual_X, weight_individual, nadir_point, ideal_point, args, return_dict):
    log_file = "logs" + os.sep + f"worker_{gen}_{i}.log"
    result_file = "logs" + os.sep + f"result_gen{gen}_ind{i}.json"

    if args.algorithm == 'r2-emoa' or args.algorithm == 'random-search':
        file = 'individual_worker.py'
    elif args.algorithm == 'r2-emoa-one-shot':
        file = 'individual_worker_one_shot.py'
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    process_args = [
        sys.executable, "-X", "dev", "-u", file,
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
        '--num_workers', str(args.num_workers),
        '--loss_type', str(args.loss_type),
        '--mu', str(args.mu),
        '--lambda_1', str(args.lambda_1),
        '--lambda_2', str(args.lambda_2),
        '--learning_rate', str(args.learning_rate),
        '--learning_rate_min', str(args.learning_rate_min),
        '--momentum', str(args.momentum),
        '--weight_decay', str(args.weight_decay),
        '--init_channels', str(args.init_channels),
        '--layers', str(args.layers),
        '--steps', str(args.steps),
        '--multiplier', str(args.multiplier),
        '--attack_eps', str(args.attack_eps),
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
    if args.proxy_data_dir is not None:
        process_args.append('--proxy_data_dir')
        process_args.append(str(args.proxy_data_dir))
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
            time_stamp_individual = time.time()
            process.communicate(timeout=args.timestamp_individual * 60)

            if process.returncode == 0 and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    res_dict = json.load(f)
                    res_dict['genotype'] = Genotype(**res_dict['genotype'])
                    return_dict[i] = res_dict
                os.remove(result_file)
                clean_file = True
                logging.info(f"Gen {gen} Individual {i}: std_acc {return_dict[i]['std_acc']:.2f}, adv_acc {return_dict[i]['adv_acc']:.2f} std_loss {return_dict[i]['std_loss']:.3f}, adv_loss {return_dict[i]['adv_loss']:.3f}, flops {return_dict[i]['flops']:.2f}, params {return_dict[i]['params']:.2f}, time {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_individual))}")
            else:
                logging.info(f"Gen {gen} Individual {i} failed with return code {process.returncode}")
                if process.returncode < 0 or process.returncode > 128:
                    # Process was killed by the system (segmentation fault, out of memory, etc.)
                    # Wait a bit to ensure the process has terminated and released resources before starting the next one
                    time.sleep(10)

        except subprocess.TimeoutExpired:
            logging.info(f"Individual {i} exceed timestamp: {args.timestamp_individual}, it will be removed from the population. If you want to increase the timestamp, please set --timestamp_individual argument to a higher value (in minutes).")
            process.kill()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                logging.info(f"Failed to kill process for individual {i} after timeout.")
        except KeyboardInterrupt:
            logging.info(f"KeyboardInterrupt received. Terminating individual {i} process.")
            process.terminate()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            sys.exit('Search interrupted by user.')
        except Exception as e:
            logging.info(f"Failed {i}: {e}")
        finally:
            # wait a bit to ensure the process has terminated and released resources before starting the next one
            time.sleep(5)
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

def evaluate_population_multiprocessing(gen, pop, weights_r2, nadir_point, ideal_point, args):
    return_dict = {}
    for i, individual in enumerate(pop):
        weight_individual = weights_r2[i].copy()
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
        individual.F_acc = np.array([
            100-float(return_dict[i]["std_acc"]),
            100-float(return_dict[i]["adv_acc"]),
            float(return_dict[i]["flops"]),
            float(return_dict[i]["params"])
        ], dtype=np.float64)
        individual.genotype = return_dict[i]["genotype"]
        individual.feasible = True if individual.genotype is not None else False

# create a subprocess where the supernet is trained
def train_supernet(pop, gen, args, nadir_point, ideal_point, warmup=False):

    log_file = "logs" + os.sep + f"supernet_{gen}_.log"

    if args.algorithm == 'r2-emoa-one-shot':
        file = 'supernet_worker.py'
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    process_args = [
        sys.executable, "-X", "dev", "-u", file,
        '--gen', str(gen),
        '--seed', str(args.seed + gen),  # Different seed for each process to avoid identical weight initialization
        '--search_space', str(args.search_space),
        '--dataset', str(args.dataset),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--data', str(args.data),
        '--num_workers', str(args.num_workers),
        '--mu', str(args.mu),
        '--learning_rate', str(args.learning_rate),
        '--learning_rate_min', str(args.learning_rate_min),
        '--momentum', str(args.momentum),
        '--weight_decay', str(args.weight_decay),
        '--init_channels', str(args.init_channels),
        '--layers', str(args.layers),
        '--steps', str(args.steps),
        '--multiplier', str(args.multiplier),
        '--attack_eps', str(args.attack_eps),
        '--cutout_length', str(args.cutout_length),
        '--drop_path_prob', str(args.drop_path_prob),
        '--grad_clip', str(args.grad_clip),
        '--train_portion', str(args.train_portion),
        '--nadir_point', np.array2string(nadir_point, separator=',', max_line_width=np.inf),
        '--ideal_point', np.array2string(ideal_point, separator=',', max_line_width=np.inf),
    ]

    individuals_X_dict = {f'individual_{i}': np.array2string(individual.X, separator=',', max_line_width=np.inf) for i, individual in enumerate(pop)}
    with open(args.save_path_final_model + os.sep + f"individuals_X_gen_{gen}.json", 'w') as f:
        json.dump(individuals_X_dict, f)
    process_args.append('--individuals_X_path')
    process_args.append(str(args.save_path_final_model) + os.sep + f"individuals_X_gen_{gen}.json")

    if args.algorithm == 'r2-emoa-one-shot':
        # Pass the path to save the supernet model, so that the individual worker can load it for evaluating the individuals
        process_args.append('--supernet_path')
        process_args.append(str(args.save_path_final_model) + os.sep + "super-net.pt")
    if warmup:
        process_args.append('--warmup')
        process_args.append('--epochs_warmup')
        process_args.append(str(args.epochs_warmup))
    else:
        process_args.append('--epochs_train_supernet')
        process_args.append(str(args.epochs_train_supernet))
    if args.reduction:
        process_args.append('--reduction')
    if args.cutout:
        process_args.append('--cutout')
    if args.proxy_data_dir is not None:
        process_args.append('--proxy_data_dir')
        process_args.append(str(args.proxy_data_dir))
    env_worker = os.environ.copy()
    env_worker['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Set the GPU device for the subprocess
    if args.debug_cuda:
        env_worker['CUDA_LAUNCH_BLOCKING'] = '1'
        env_worker['TORCH_USE_CUDA_DSA'] = '1'
    with open(log_file, 'w') as f_log:
        process = subprocess.Popen(process_args,
                                    stdout=f_log,
                                    stderr=f_log,
                                    text=True,
                                    env=env_worker)

        try:
            time_stamp_gen = time.time()
            process.communicate(timeout=args.timestamp_supernet * 60)

            if process.returncode != 0:
                logging.info(f"Gen {gen} training failed with return code {process.returncode}")
                if process.returncode < 0 or process.returncode > 128:
                    # Process was killed by the system (segmentation fault, out of memory, etc.)
                    # Wait a bit to ensure the process has terminated and released resources before starting the next one
                    time.sleep(5)
            else:
                logging.info(f"Gen {gen} training completed successfully in {time.gmtime(time.time() - time_stamp_gen)} minutes.")

        except subprocess.TimeoutExpired:
            logging.info(f"Gen {gen} training exceed timestamp: {args.timestamp_supernet}, it will be skipped. If you want to increase the timestamp, please set --timestamp_supernet argument to a higher value (in minutes).")
            process.kill()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                logging.info(f"Failed to kill process for generation {gen} after timeout.")
        except KeyboardInterrupt:
            logging.info(f"KeyboardInterrupt received. Terminating generation training {gen} process.")
            process.terminate()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            sys.exit('Search interrupted by user.')
        except Exception as e:
            logging.info(f"Failed generation {gen}: {e}")
        finally:
            # wait a bit to ensure the process has terminated and released resources before starting the next one
            time.sleep(5)
        # Remove log file of successful training, keep logs of failed training for debugging
    if os.path.exists(log_file) and process.returncode == 0:
        os.remove(log_file)