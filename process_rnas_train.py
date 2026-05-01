
import logging
import shutil
import subprocess
import argparse
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == '__main__':
    """
    python3 rnas_train.py --seed 12 --algorithm r2-emoa --search_space discrete --dataset cifar10 \
    --batch_size 32 --epochs 100 --data ./data --learning_rate 0.025 --learning_rate_min 0.001\
    --momentum 0.9 --weight_decay 3e-4 --grad_clip 5.0 --report_freq 50 --freq_save 10 --gpu 0\
    --init_channels 16 --layers 8 --steps 4 --multiplier 4 --train_portion 0.5\
    --archive_path results/r2-emoa/cifar10/2026-04-20_11-37-00_18906049/search/population_data.json

    python3 rnas_train.py --seed 12 --algorithm r2-emoa --search_space discrete --dataset cifar10 --reload_dir auto-last
    """
    parser = argparse.ArgumentParser(description="Training architectures found by RNAS")
    parser.add_argument('--seed', type=int, default=18906049, help='random seed')
    parser.add_argument('--algorithm', type=str, choices=['r2-emoa', 'nevonas', 'nsganet', 'cars', 'r2-emoa-one-shot'],
                        help='which algorithm was used to search')
    parser.add_argument('--search_space', type=str, default='discrete', help='which search space was used to search')
    parser.add_argument('--dataset', type=str, choices=['cifar10'], help='dataset for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='gradient clipping')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--freq_save', type=int, default=10, help='frequency of saving the model')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
    parser.add_argument('--steps', type=int, default=4, help='number of steps in one cell')
    parser.add_argument('--multiplier', type=int, default=4, help='number of multiplier for channels')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
    parser.add_argument('--debug_cuda', action='store_true', default=False, help='debug cuda')
    parser.add_argument('--reload_dir', type=str, default=None, help='reload from this directory')
    parser.add_argument('--archive_path', type=str, default=None, help='path to the archive of architectures')
    args = parser.parse_args()

    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    file = 'rnas_train.py'
    process_args = [
        sys.executable, "-X", "dev", "-u", file,
        "--seed", str(args.seed),
        "--algorithm", args.algorithm,
        "--search_space", args.search_space,
        "--dataset", args.dataset,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--data", args.data,
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_min", str(args.learning_rate_min),
        "--momentum", str(args.momentum),
        "--weight_decay", str(args.weight_decay),
        "--grad_clip", str(args.grad_clip),
        "--report_freq", str(args.report_freq),
        "--freq_save", str(args.freq_save),
        "--gpu", str(args.gpu),
        "--init_channels", str(args.init_channels),
        "--layers", str(args.layers),
        "--steps", str(args.steps),
        "--multiplier", str(args.multiplier),
        "--train_portion", str(args.train_portion),
    ]

    if args.reload_dir is not None:
        process_args.extend(["--reload_dir", args.reload_dir])
    if args.archive_path is not None:
        process_args.extend(["--archive_path", args.archive_path])
    if args.debug_cuda:
        process_args.extend(["--debug_cuda"])
    n_executions = 0
    log_file = 'logs' + os.sep + f'training_{n_executions}.log'
    last_execution_code = -1

    while last_execution_code != 0:
        clean_file = False
        with open(log_file, 'w') as f_log:
            process = subprocess.Popen(process_args, stdout=f_log, stderr=f_log, text=True)
            try:
                process.communicate(timeout=120*60)
                if process.returncode != 0:
                    last_execution_code = process.returncode
                    logging.info(f"Process exited with code {process.returncode}")
                elif process.returncode == 0:
                    last_execution_code = 0
                    clean_file = True
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
            sys.exit("Training process finished")
        # increment execution count and update log file name
        n_executions += 1
        log_file = 'logs' + os.sep + f'training_{n_executions}.log'
        # update processs_args to reload from the last execution
        process_args = [
            sys.executable, "-X", "dev", "-u", file,
            "--seed", str(args.seed),
            "--algorithm", str(args.algorithm),
            "--dataset", str(args.dataset),
            "--reload_dir", "auto-last",
        ]
