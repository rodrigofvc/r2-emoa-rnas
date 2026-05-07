import argparse
import logging
import os
import shutil
import subprocess
import sys

"""
python3 -X dev process_arch_search.py --seed 18906049 --dataset cifar10 --batch_size 32 --gpu 0 --init_channels 16 --generations 4 --n_population 10 --learning_rate 0.025 --learning_rate_min 0.001 --momentum 0.9 --weight_decay 3e-4 --mutate_rate 0.1 --report_freq 50 --layers 5 --steps 4 --multiplier 4 --reduction --grad_clip 5 --train_portion 0.5 --epochs_warmup 2 --epochs_train_supernet 1 --timestamp_supernet 45 --timestamp_individual 5 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser("process S1")
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--data', type=str, default='../../../data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='', help='["cifar10", "cifar100"]')
    parser.add_argument('--generations', type=int, default=30, help='num of generations')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--knn', type=int, default=5, help='k-nearest neighbors')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='weight for std loss')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='weight for adv loss')
    parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell')
    parser.add_argument('--multiplier', type=int, default=6, help='multiplier for number of channels')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--mutate_rate', type=float, default=0.1, help='mutation rate')
    parser.add_argument('--fgsm_eps', type=float, default=8 / 255, help='attack epsilon')
    parser.add_argument('--n_population', type=int, default=40, help='population size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--seed', type=int, default=18906049, help='random seed')
    parser.add_argument('--train_portion', type=float, default=0.5, help='split option for CIFAR100')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--epochs_warmup', type=int, default=0, help='number of epochs to warmup the supernet before starting the search process')
    parser.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
    parser.add_argument('--pretrained_supernet', type=str, default=None, help='path to pretrained supernet to load before training')
    parser.add_argument('--save_path_final_model', type=str, default=None, help='path to save the final supernet model after search')
    parser.add_argument('--timestamp_supernet', type=int, default=45, help='timestamp in minutes for training the supernet')
    parser.add_argument('--timestamp_individual', type=int, default=5, help='timestamp in minutes for evaluating each architecture')
    parser.add_argument('--search_space', type=str, default="continuous", choices=['continuous', 'discrete'], help='search space to use')
    parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='Enable CUDA_LAUNCH_BLOCKING for debugging')
    parser.add_argument('--increase_epochs', action='store_true', default=False, help='Increase the number of epochs to train the supernet and individuals as generations progress')
    parser.add_argument('--reload_dir', type=str, default=None, help='Directory to reload the experiment from if --reload is set')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    file = 'arch_search.py'

    if args.reload_dir != 'auto-reload':
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--batch_size", str(args.batch_size),
            "--cutout_length", str(args.cutout_length),
            "--data", str(args.data),
            "--dataset", str(args.dataset),
            "--generations", str(args.generations),
            "--gpu", str(args.gpu),
            "--grad_clip", str(args.grad_clip),
            "--init_channels", str(args.init_channels),
            "--knn", str(args.knn),
            "--layers", str(args.layers),
            "--lambda_1", str(args.lambda_1),
            "--lambda_2", str(args.lambda_2),
            "--steps", str(args.steps),
            "--multiplier", str(args.multiplier),
            "--learning_rate", str(args.learning_rate),
            "--learning_rate_min", str(args.learning_rate_min),
            "--drop_path_prob", str(args.drop_path_prob),
            "--momentum", str(args.momentum),
            "--mutate_rate", str(args.mutate_rate),
            "--fgsm_eps", str(args.fgsm_eps),
            "--n_population", str(args.n_population),
            "--report_freq", str(args.report_freq),
            "--seed", str(args.seed),
            "--train_portion", str(args.train_portion),
            "--weight_decay", str(args.weight_decay),
            "--epochs_warmup", str(args.epochs_warmup),
            "--epochs_train_supernet", str(args.epochs_train_supernet),
            "--timestamp_supernet", str(args.timestamp_supernet),
            "--timestamp_individual", str(args.timestamp_individual),
            "--search_space", str(args.search_space),
        ]
        if args.cutout:
            process_args.append('--cutout')
        if args.reduction:
            process_args.append('--reduction')
        if args.debug_cuda:
            process_args.append('--debug_cuda')
        if args.increase_epochs:
            process_args.append('--increase_epochs')
        if args.pretrained_supernet is not None:
            process_args.extend(['--pretrained_supernet', args.pretrained_supernet])
        if args.save_path_final_model is not None:
            process_args.extend(['--save_path_final_model', args.save_path_final_model])
        if args.reload_dir is not None:
            process_args.extend(['--reload_dir', args.reload_dir])
    else:
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            '--seed', str(args.seed),
            '--dataset', args.dataset,
            '--reload_dir', 'auto-reload'
        ]

    n_executions = 0
    log_file = 'logs' + os.sep + f'searching_{n_executions}.log'
    last_execution_code = -1
    tries = 10
    while last_execution_code != 0 and n_executions < tries:
        clean_file = False
        with open(log_file, 'wb') as f_log:
            process = subprocess.Popen(process_args, stdout=f_log, stderr=f_log, text=True)
            try:
                process.communicate(timeout=5000*60)
                if process.returncode != 0:
                    last_execution_code = process.returncode
                    logging.info(f"Process {n_executions} exited with code {process.returncode}")
                elif process.returncode == 0:
                    last_execution_code = 0
                    clean_file = True
                    sys.exit('Search completed successfully.')
            except subprocess.TimeoutExpired:
                logging.info(f"Process timed out. Killing process.")
                process.kill()
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    logging.info(f"Failed to kill process after timeout.")
            except KeyboardInterrupt:
                logging.info(f"KeyboardInterrupt received. Terminating process.")
                process.terminate()
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)
                sys.exit('Search interrupted by user.')
        if clean_file:
            os.remove(log_file)
            sys.exit('Search completed successfully.')
        # increment execution count and update log file name
        n_executions += 1
        log_file = 'logs' + os.sep + f'searching_{n_executions}.log'
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--seed", str(args.seed),
            "--dataset", str(args.dataset),
            "--reload_dir", "auto-last",
        ]
    logging.info('Search process failed after multiple attempts. Please check the log files for more details.')