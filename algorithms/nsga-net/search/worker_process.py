import json
import os
import subprocess
import sys
import time
import logging

import numpy as np

from micro_encoding import Genotype


def worker_evaluate_individual(gen, i, individual_X, args):
    log_file = "logs" + os.sep + f"worker_{gen}_{i}.log"
    result_file = "logs" + os.sep + f"result_gen{gen}_ind{i}.json"

    file = 'search' + os.sep + 'individual_worker.py'
    return_dict = {
        'std_acc': 0.0,
        'adv_acc': 0.0,
        'std_loss': 1000.0,
        'adv_loss': 1000.0,
        'flops': 1000.0,
        'params': 1000.0,
        'genotype': None
    }
    process_args = [
        sys.executable, "-X", "dev", "-u", file,
        '--gen', str(gen),
        '--i', str(i),
        '--seed', str(args.seed + i),  # Different seed for each process to avoid identical weight initialization
        '--individual_x', np.array2string(individual_X, separator=',', max_line_width=np.inf),
        '--dataset', str(args.dataset),
        '--gpu', str(args.gpu),
        '--batch_size', str(args.batch_size),
        '--epochs_train_individual', str(args.epochs_train_individual),
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
                    return_dict = json.load(f)
                    return_dict['genotype'] = Genotype(**return_dict['genotype'])
                os.remove(result_file)
                clean_file = True
                logging.info(f"Gen {gen} Individual {i}: std_acc {return_dict['std_acc']:.2f}, adv_acc {return_dict['adv_acc']:.2f} std_loss {return_dict['std_loss']:.3f}, adv_loss {return_dict['adv_loss']:.3f}, flops {return_dict['flops']:.2f}, params {return_dict['params']:.2f}")
            else:
                logging.info(f"Gen {gen} Individual {i} failed with return code {process.returncode}")
                if process.returncode < 0 or process.returncode > 128:
                    # Process was killed by the system (segmentation fault, out of memory, etc.)
                    # Wait a bit to ensure the process has terminated and released resources before starting the next one
                    time.sleep(10)

        except subprocess.TimeoutExpired:
            logging.info(f"Individual {i} exceed timestamp: {args.timestamp}, it will be removed from the population. If you want to increase the timestamp, please set --timestamp argument to a higher value (in minutes).")
            process.kill()
            try:
                process.communicate(timeout=args.timestamp * 60)
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
            #time.sleep(5)
            pass
    # Remove log file of successful evaluations, keep logs of failed evaluations for debugging
    if clean_file and os.path.exists(log_file):
        os.remove(log_file)
    return return_dict

