import argparse
import os

import numpy as np
import utils
import torch, torchvision

def get_evaluation_dataset(args):
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    num_train = len(train_data)
    split = int(np.floor(args.train_portion * num_train))


    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(num_train)
    candidate_indices = indices[split:]

    sampler = torch.utils.data.SubsetRandomSampler(
        candidate_indices.tolist(),
        generator=torch.Generator().manual_seed(args.seed),
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0, pin_memory=False,
        shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed))

    return train_data

def create_evaluation_proxy_indices(dataset, train_portion, eval_size, seed):
    num_samples = len(dataset)

    if not 0.0 < train_portion < 1.0:
        raise ValueError("train_portion must be between zero and one.")

    split = int(np.floor(train_portion * num_samples))

    if eval_size <= 0:
        raise ValueError("eval_size must be positive.")

    rng = np.random.default_rng(seed)

    # Must exactly reproduce the split used to generate proxy_train
    permutation = rng.permutation(num_samples)

    train_candidate_indices = permutation[:split]
    eval_candidate_indices = permutation[split:]


    labels = np.asarray(dataset.targets)
    eval_candidate_labels = labels[eval_candidate_indices]

    classes = np.unique(labels)
    base_per_class = eval_size // len(classes)
    remainder = eval_size % len(classes)

    selected_indices = []

    for position, class_id in enumerate(classes):
        class_indices = eval_candidate_indices[eval_candidate_labels == class_id]

        class_size = base_per_class + int(position < remainder)

        if class_size > len(class_indices):
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} candidates, "
                f"but {class_size} were requested."
            )

        class_selection = rng.choice(class_indices, size=class_size, replace=False)

        selected_indices.extend(class_selection.tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)

    # Shuffle so that examples are not ordered by class
    rng.shuffle(selected_indices)

    assert len(selected_indices) == eval_size
    assert len(np.unique(selected_indices)) == eval_size
    assert np.intersect1d(selected_indices, train_candidate_indices).size == 0

    return selected_indices

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Proxy Evaluation')
    args.add_argument('--dataset', type=str, default='cifar10', help='Dataset name (cifar10 or cifar100)')
    args.add_argument('--train_portion', type=float, default=0.5, help='Portion of the dataset used for training')
    args.add_argument('--eval_size', type=int, default=1000, help='Number of evaluation samples to select')
    args.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args.add_argument('--data', type=str, default='../data', help='Directory to download/load the dataset')
    args.add_argument('--output_dir', type=str, default='./proxy_eval/', help='Directory to save the evaluation proxy indices')
    args.add_argument("--cutout", action="store_true", default=False)
    args.add_argument("--cutout_length", type=int, default=16)
    args = args.parse_args()
    eval_queue = get_evaluation_dataset(args)
    selected_indices = create_evaluation_proxy_indices(eval_queue, args.train_portion, args.eval_size, args.seed)

    eval_indices_path = os.path.join(args.output_dir,f"eval_proxy_indices_{args.dataset}_{args.batch_size}_{len(selected_indices)}.npy")

    np.save(eval_indices_path, selected_indices)

    print(f"Evaluation proxy indices saved to: {eval_indices_path}")

    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    if args.dataset == "cifar10":
        full_eval_dataset = torchvision.datasets.CIFAR10(
            root=args.data,
            train=True,
            download=True,
            transform=train_transform,
        )
    elif args.dataset == "cifar100":
        full_eval_dataset = torchvision.datasets.CIFAR100(
            root=args.data,
            train=True,
            download=True,
            transform=train_transform,
        )

    eval_proxy_dataset = torch.utils.data.Subset(
        full_eval_dataset,
        selected_indices.tolist(),
    )

    eval_queue = torch.utils.data.DataLoader(
        eval_proxy_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Evaluation proxy size: {len(eval_proxy_dataset)}")