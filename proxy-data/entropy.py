import argparse
import os
import sys
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

sys.path.append('..')

import utils
from adversarial import fgsm_simple
from resnet import resnet20, resnet56

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index

def get_entropy_dataset(args):
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    indexed_train_data = IndexedDataset(train_data)

    num_train = len(train_data)
    split = int(np.floor(args.train_portion * num_train))

    if torch.backends.mps.is_available():
        # testing
        split = 96

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(num_train)
    candidate_indices = indices[:split]

    sampler = torch.utils.data.SubsetRandomSampler(
        candidate_indices.tolist(),
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_queue = torch.utils.data.DataLoader(
        indexed_train_data, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed))

    return train_queue


def predictive_entropy(logits: torch.Tensor):
    probabilities = torch.softmax(logits, dim=1)
    log_probabilities = torch.log_softmax(logits, dim=1)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    return entropy

def compute_adversarial_entropies(model, loader, device, eps=8/255):
    model.eval()

    all_indices = []
    all_adversarial_entropies = []
    all_labels = []
    all_adversarial_predictions = []

    for images, targets, indices in loader:
        images = images.to(device)
        targets = targets.to(device)
        adv_images, _ = fgsm_simple(model=model, x=images, y=targets, eps=eps)

        with torch.no_grad():
            adversarial_logits = model(adv_images)
            adversarial_entropy = predictive_entropy(adversarial_logits)
            adversarial_predictions = (adversarial_logits.argmax(dim=1))

        all_indices.append(indices.cpu().numpy())

        all_adversarial_entropies.append(adversarial_entropy.cpu().numpy())

        all_labels.append(targets.cpu().numpy())

        all_adversarial_predictions.append(adversarial_predictions.cpu().numpy())

    return {
        "indices": np.concatenate(all_indices),
        "adversarial_entropy": np.concatenate(all_adversarial_entropies),
        "labels": np.concatenate(all_labels),
        "adversarial_predictions": np.concatenate(all_adversarial_predictions),
    }

def compute_p1_probabilities(scores, num_bins):
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")

    if len(scores) == 0:
        raise ValueError("scores cannot be empty")

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinite values")

    counts, bin_edges = np.histogram(scores, bins=num_bins)

    bin_indices = np.digitize(scores, bin_edges[1:-1], right=False)

    max_count = counts.max()

    sample_weights = np.zeros(len(scores), dtype=np.float64)

    for index, bin_index in enumerate(bin_indices):
        bin_count = counts[bin_index]

        if bin_count <= 0:
            raise RuntimeError("A sample was assigned to an empty bin")

        bin_weight = (max_count - bin_count + 1)

        sample_weights[index] = (bin_weight / bin_count)

    weight_sum = sample_weights.sum()

    if weight_sum <= 0:
        raise RuntimeError("The sum of sample weights is not positive")

    probabilities = (sample_weights / weight_sum)

    return {
        "probabilities": probabilities,
        "counts": counts,
        "bin_edges": bin_edges,
        "bin_indices": bin_indices,
        "sample_weights": sample_weights,
    }

# usage
"""
python3 entropy.py --dataset cifar10 --seed 42 --batch_size 32 \
--train_portion 0.5 --proxy_ratio 0.1 --num_workers 0 --gpu 0 \
--model_path models/resnet20_cifar10.pth --attack_eps 0.03137254901960784 --num_bins 10
"""
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Train a surrogate model on CIFAR datasets')
    args.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to train the model on')
    args.add_argument('--data', type=str, default='../data', help='Directory to download/load the dataset')
    args.add_argument('--batch_size', type=int, default=128, help='Batch size for training and validation')
    args.add_argument('--train_portion', type=float, default=0.5, help='Portion of the dataset to use for training')
    args.add_argument('--proxy_ratio', type=float, default=0.1, help='Ratio of proxy data to use for training')
    args.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args.add_argument('--gpu', type=int, default=0, help='GPU id to use for training')
    args.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'resnet56'], help='Model architecture to use')
    args.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    args.add_argument('--attack_eps', type=float, default=8/255, help='Epsilon value for adversarial training')
    args.add_argument('--num_bins', type=int, default=10, help='Number of bins for entropy histogram')
    args.add_argument("--cutout", action="store_true", default=False)
    args.add_argument("--cutout_length", type=int, default=16)
    args.add_argument('--output_dir', type=str, default='./proxy_indices/', help='Directory to save the proxy indices')
    args = args.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")

    if args.model == 'resnet20':
        model = resnet20(num_classes=10 if args.dataset == 'cifar10' else 100)
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict)
    elif args.model == 'resnet56':
        model = resnet56(num_classes=10 if args.dataset == 'cifar10' else 100)
        state_dict = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")

    model.to(args.device)
    train_queue = get_entropy_dataset(args)

    entropy_results = compute_adversarial_entropies(
        model=model,
        loader=train_queue,
        device=args.device,
        eps=args.attack_eps,
    )

    #np.savez(f"entropy_data_{args.dataset}.npz", **entropy_results)

    scores = entropy_results[
        "adversarial_entropy"
    ]

    candidate_indices = entropy_results[
        "indices"
    ]

    p1_results = compute_p1_probabilities(
        scores=scores,
        num_bins=args.num_bins,
    )

    probabilities = p1_results["probabilities"]

    num_candidates = len(scores)

    proxy_size = int(
        num_candidates * args.proxy_ratio
    )

    rng = np.random.default_rng(args.seed)

    selected_positions = rng.choice(
        num_candidates,
        size=proxy_size,
        replace=False,
        p=probabilities,
    )

    proxy_indices = candidate_indices[selected_positions]

    proxy_indices = np.sort(proxy_indices)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    np.save(os.path.join(args.output_dir, f"proxy_indices_{args.dataset}_{args.model}_{args.proxy_ratio}.npy"), proxy_indices)

    print(f"Proxy indices saved to proxy_indices_{args.dataset}_{args.proxy_ratio}.npy")

    print(f"Original CIFAR training set: {len(train_queue.dataset)}")

    print(f"Base training candidates: {num_candidates}")

    print(f"Proxy ratio: {args.proxy_ratio}")

    print(f"Selected proxy images: {len(proxy_indices)}")

    # Load the full training dataset to create a proxy dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=utils.data_transforms_cifar10(args)[0],
    )

    proxy_dataset = torch.utils.data.Subset(
        full_train_dataset,
        proxy_indices.tolist(),
    )
    print(f"Proxy dataset size: {len(proxy_dataset)}")
