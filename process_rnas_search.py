# script that executes rnas_search.py as subprocess continuously and saves the output to a file
import argparse
import logging
import os
import shutil
import subprocess
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running R2-EMOA for RNAS")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa', 'r2-emoa-one-shot', 'random-search'], help='algorithm to run')
    parser.add_argument('--search_space', type=str, default="continuous", choices=['continuous', 'discrete'], help='search space to use')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_population', type=int, default=40, help='population size')
    parser.add_argument('--generations', type=int, default=30, help='number of generations to search')
    parser.add_argument('--epochs_warmup', type=int, default=0, help='number of epochs to warmup supernet')
    parser.add_argument('--pretrained_supernet', type=str, default=None, help='path to pretrained supernet to load before training')
    parser.add_argument('--epochs_train_supernet', type=int, default=0, help='number of epochs to train supernet per generation')
    parser.add_argument('--epochs_train_individual', type=int, default=1, help='number of epochs to train individual per generation')
    parser.add_argument('--objectives', type=int, default=4, help='number of objectives')
    parser.add_argument('--std_loss_index', type=int, default=0, help='index of standard loss in objectives')
    parser.add_argument('--adv_loss_index', type=int, default=1, help='index of adversarial loss in objectives')
    parser.add_argument('--flops_index', type=int, default=2, help='index of flops in objectives')
    parser.add_argument('--params_index', type=int, default=3, help='index of params in objectives')
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--prob_cross', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--prob_mut', type=float, default=0.1, help='mutation probability')
    parser.add_argument('--eta_cross', type=int, default=15, help='crossover eta')
    parser.add_argument('--eta_mut', type=int, default=20, help='mutation eta')
    parser.add_argument('--loss_type', type=str, default='tchebycheff', choices=['tchebycheff', 'ws'], help='type of loss function to use for backpropagation')
    parser.add_argument('--mu', type=float, default=0.1, help='mu for thchebycheff function')
    parser.add_argument('--lambda_1', type=float, default=0.5, help='weight for standard loss in ws scalarization')
    parser.add_argument('--lambda_2', type=float, default=0.5, help='weight for adversarial loss in ws scalarization')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='init channels')
    parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    parser.add_argument('--layers', type=int, default=5, help='total number of layers (cells)')
    parser.add_argument('--steps', type=int, default=6, help='number of steps in one cell (intern nodes except input and output)')
    parser.add_argument('--multiplier', type=int, default=6, help='number of multiplier for number of channels (intern nodes to concat)')
    parser.add_argument('--attack', type=str, default='FGSM', help='adversarial attack to use')
    parser.add_argument('--fgsm_eps', type=float, default=8/255, help='attack epsilon')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--timestamp_supernet', type=int, default=45, help='timestamp in minutes for training supernet (including warmup) per generation in one-shot nas')
    parser.add_argument('--timestamp_individual', type=int, default=7, help='timestamp in minutes for training/eval each architecture')
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

    file = 'rnas_search.py'

    if args.reload_dir != "auto-last":
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--seed", str(args.seed),
            "--algorithm", str(args.algorithm),
            "--search_space", str(args.search_space),
            "--dataset", str(args.dataset),
            "--batch_size", str(args.batch_size),
            "--n_population", str(args.n_population),
            "--generations", str(args.generations),
            "--epochs_warmup", str(args.epochs_warmup),
            "--epochs_train_supernet", str(args.epochs_train_supernet),
            "--epochs_train_individual", str(args.epochs_train_individual),
            "--objectives", str(args.objectives),
            "--std_loss_index", str(args.std_loss_index),
            "--adv_loss_index", str(args.adv_loss_index),
            "--flops_index", str(args.flops_index),
            "--params_index", str(args.params_index),
            "--data", args.data,
            "--prob_cross", str(args.prob_cross),
            "--prob_mut", str(args.prob_mut),
            "--eta_cross", str(args.eta_cross),
            "--eta_mut", str(args.eta_mut),
            "--loss_type", args.loss_type,
            "--mu", str(args.mu),
            "--lambda_1", str(args.lambda_1),
            "--lambda_2", str(args.lambda_2),
            "--learning_rate", str(args.learning_rate),
            "--learning_rate_min", str(args.learning_rate_min),
            "--momentum", str(args.momentum),
            "--weight_decay", str(args.weight_decay),
            "--report_freq", str(args.report_freq),
            "--gpu", str(args.gpu),
            "--init_channels", str(args.init_channels),
            "--layers", str(args.layers),
            "--steps", str(args.steps),
            "--multiplier", str(args.multiplier),
            "--attack", args.attack,
            "--fgsm_eps", str(args.fgsm_eps),
            "--cutout_length", str(args.cutout_length),
            "--drop_path_prob", str(args.drop_path_prob),
            "--grad_clip", str(args.grad_clip),
            "--train_portion", str(args.train_portion),
            "--timestamp_supernet", str(args.timestamp_supernet),
            "--timestamp_individual", str(args.timestamp_individual),
        ]
        if args.debug_cuda:
            process_args.append("--debug_cuda")
        if args.increase_epochs:
            process_args.append("--increase_epochs")
        if args.reload_dir is not None:
            # continue from the specified directory
            process_args.extend(["--reload_dir", str(args.reload_dir)])
        if args.cutout:
            process_args.append("--cutout")
        if args.reduction:
            process_args.append("--reduction")
        if args.pretrained_supernet is not None:
            process_args.extend(["--pretrained_supernet", str(args.pretrained_supernet)])
    else:
        # start from the last execution directory
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--seed", str(args.seed),
            "--algorithm", str(args.algorithm),
            "--dataset", str(args.dataset),
            "--reload_dir", "auto-last"
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
                process.communicate(timeout=4320*60)
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
            "--algorithm", str(args.algorithm),
            "--dataset", str(args.dataset),
            "--reload_dir", "auto-last",
        ]
    logging.info('Search did not complete successfully after multiple attempts. Please check the log files for details.')
