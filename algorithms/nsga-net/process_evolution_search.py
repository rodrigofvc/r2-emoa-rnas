import argparse
import logging
import os
import shutil
import subprocess
import sys

# python3 process_evolution_search.py --seed 1 --dataset cifar100 --reload_dir auto-last
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
    parser.add_argument('--save', type=str, default='NSGA-Net', help='experiment name')
    parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='dataset name')
    parser.add_argument('--n_classes', type=int, choices=[10, 100], help='number of classes')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
    # arguments for micro search space
    parser.add_argument('--n_blocks', type=int, default=6, help='number of blocks in a cell')
    parser.add_argument('--n_ops', type=int, default=10, help='number of operations considered')
    parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
    # arguments for macro search space
    parser.add_argument('--n_nodes', type=int, default=6, help='number of nodes per phases')
    # hyper-parameters for algorithm
    parser.add_argument('--pop_size', type=int, default=5, help='population size of networks')
    parser.add_argument('--n_gens', type=int, default=50, help='population size')
    parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
    # arguments for back-propagation training during search
    parser.add_argument('--init_channels', type=int, default=16, help='# of filters for first cell')
    parser.add_argument('--layers', type=int, default=4, help='equivalent with N = 3')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=45, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--reduction', action='store_true', default=False, help='use reduction cell or not')
    parser.add_argument('--steps', type=int, default=6,
                        help='number of steps in one cell (intern nodes except input and output)')
    parser.add_argument('--multiplier', type=int, default=6,
                        help='number of multiplier for number of channels (intern nodes to concat)')
    parser.add_argument('--fgsm_eps', type=float, default=8 / 255, help='attack epsilon')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--timestamp', type=int, default=6,
                        help='timestamp in minutes for training/eval each architecture')
    parser.add_argument('--debug_cuda', action='store_true', default=False,
                        help='Enable CUDA_LAUNCH_BLOCKING for debugging')
    parser.add_argument('--increase_epochs', action='store_true', default=False,
                        help='Increase the number of epochs to train the supernet and individuals as generations progress')
    parser.add_argument('--reload_dir', type=str, default=None,
                        help='Directory to reload the experiment from if --reload is set')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    file = 'search/evolution_search.py'

    if args.reload_dir != 'auto-last':
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--save", args.save,
            "--data", args.data,
            "--dataset", args.dataset,
            "--n_classes", str(args.n_classes),
            "--seed", str(args.seed),
            "--search_space", args.search_space,
            "--n_blocks", str(args.n_blocks),
            "--n_ops", str(args.n_ops),
            "--n_cells", str(args.n_cells),
            "--n_nodes", str(args.n_nodes),
            "--pop_size", str(args.pop_size),
            "--n_gens", str(args.n_gens),
            "--n_offspring", str(args.n_offspring),
            "--init_channels", str(args.init_channels),
            "--layers", str(args.layers),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--learning_rate_min", str(args.learning_rate_min),
            "--momentum", str(args.momentum),
            "--weight_decay", str(args.weight_decay),
            "--report_freq", str(args.report_freq),
            "--gpu", str(args.gpu),
            "--steps", str(args.steps),
            "--multiplier", str(args.multiplier),
            "--fgsm_eps", str(args.fgsm_eps),
            "--cutout_length", str(args.cutout_length),
            "--drop_path_prob", str(args.drop_path_prob),
            "--grad_clip", str(args.grad_clip),
            "--epochs", str(args.epochs),
            "--train_portion", str(args.train_portion),
            "--timestamp", str(args.timestamp)
        ]
        if args.reduction:
            process_args.append('--reduction')
        if args.cutout:
            process_args.append('--cutout')
        if args.increase_epochs:
            process_args.append('--increase_epochs')
        if args.debug_cuda:
            process_args.append('--debug_cuda')
        if args.reload_dir is not None:
            process_args.extend(['--reload_dir', args.reload_dir])
    else:
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--seed", str(args.seed),
            "--dataset", args.dataset,
            "--reload_dir", "auto-last"
        ]


    n_executions = 0
    log_file = 'logs' + os.sep + f'searching_{n_executions}.log'
    last_execution_code = -1

    while last_execution_code != 0:
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
            "--dataset", str(args.dataset),
            "--reload_dir", "auto-last",
        ]