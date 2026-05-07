import json
import os
import subprocess
import sys
import time
import logging
import numpy as np

from micro_space.micro_encoding import Genotype


def _remove_file_with_retries(file_path, retries=5, delay=0.2):
    for attempt in range(retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except PermissionError:
            if attempt == retries - 1:
                logging.warning(f"Could not remove log file '{file_path}' after {retries} attempts.")
                return False
            time.sleep(delay)
    return False


def worker_evaluate_individual(gen, i, individual_X, args):
    log_file = "logs" + os.sep + f"worker_{gen}_{i}.log"
    result_file = "logs" + os.sep + f"result_gen{gen}_ind{i}.json"

    file = 'individual_worker_one_shot.py'

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
        '--data', str(args.data),
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
        '--supernet_path', str(args.save_path_final_model) + os.sep + "super-net.pt"
    ]

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
            start_time = time.time()
            process.communicate(timeout=args.timestamp_individual * 60)

            if process.returncode == 0 and os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    res_dict = json.load(f)
                    res_dict['genotype'] = Genotype(**res_dict['genotype'])
                os.remove(result_file)
                clean_file = True
                logging.info(f"Gen {gen} Individual {i}: std_acc {res_dict['std_acc']:.2f}, adv_acc {res_dict['adv_acc']:.2f} std_loss {res_dict['std_loss']:.3f}, adv_loss {res_dict['adv_loss']:.3f}, flops {res_dict['flops']:.2f}, params {res_dict['params']:.2f}, (HH:MM:SS) {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            else:
                logging.info(f"Gen {gen} Individual {i} failed with return code {process.returncode}")

        except subprocess.TimeoutExpired:
            logging.info(f"Individual {i} exceed timestamp, it will be removed from the population.")
            process.kill()
            process.communicate(timeout=10)
        except KeyboardInterrupt:
            logging.info(f"KeyboardInterrupt received. Terminating individual {i} process.")
            process.terminate()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
            sys.exit('Search interrupted by user.')
        except Exception as e:
            logging.info(f"Failed {i}: {type(e).__name__} - {e}")
        finally:
            # wait a bit to ensure the process has terminated and released resources before starting the next one
            time.sleep(5)
            if process.returncode != 0:
                return {
                    "std_acc": 0.0,
                    "adv_acc": 0.0,
                    "std_loss": 1000.0,
                    "adv_loss": 1000.0,
                    "flops": 1000.0,
                    "params": 1000.0,
                    "genotype": None
                }
            # set default values for failed individuals
    # Remove log file of successful evaluations, keep logs of failed evaluations for debugging
    if clean_file:
        _remove_file_with_retries(log_file)
        return res_dict


def train_supernet(pop, gen, args, warmup=False):

    log_file = "logs" + os.sep + f"supernet_{gen}_.log"

    file = 'supernet_worker.py'

    process_args = [
        sys.executable, "-X", "dev", "-u", file,
        '--gen', str(gen),
        '--seed', str(args.seed + gen),  # Different seed for each process to avoid identical weight initialization
        '--search_space', str(args.search_space),
        '--dataset', str(args.dataset),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--data', str(args.data),
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
    ]

    individuals_X_dict = {f'individual_{i}': np.array2string(individual, separator=',', max_line_width=np.inf) for i, individual in enumerate(pop)}
    with open(args.save_path_final_model + os.sep + f"individuals_X_gen_{gen}.json", 'w') as f:
        json.dump(individuals_X_dict, f)
    process_args.append('--individuals_X_path')
    process_args.append(str(args.save_path_final_model) + os.sep + f"individuals_X_gen_{gen}.json")


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
    env_worker = os.environ.copy()
    env_worker['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # Set the GPU device for the subprocess
    if args.debug_cuda:
        env_worker['CUDA_LAUNCH_BLOCKING'] = '1'
        env_worker['TORCH_USE_CUDA_DSA'] = '1'
    training_succeeded = False
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
                training_succeeded = True
                logging.info(f"Gen {gen} training completed successfully in (HH:MM:SS) {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_stamp_gen))}")

        except subprocess.TimeoutExpired:
            logging.info(f"Gen {gen} training exceed timestamp: {args.timestamp_supernet}, it will be skipped. If you want to increase the timestamp, please set --timestamp_supernet argument to a higher value (in minutes).")
            process.kill()
            try:
                process.communicate(timeout=args.timestamp_supernet * 60)
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
            time.sleep(10)
        # Remove log file of successful training, keep logs of failed training for debugging
    if training_succeeded:
        _remove_file_with_retries(log_file)
