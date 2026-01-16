import json
import lzma
import os
import pickle
import torch

def save_model(model, model_path, name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path += os.sep + name
    torch.save(model, model_path)


def save_params(args, trained_arch_path):
    params_path = trained_arch_path + os.sep
    params_dict = vars(args)
    params_dict['device'] = str(params_dict['device'])
    if not os.path.exists(os.path.dirname(params_path)):
        os.makedirs(os.path.dirname(params_path))
    params_path += 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)

def data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def load_architecture(architect_path):
    with lzma.open(architect_path, 'rb') as f:
        architectures = pickle.load(f)
    return architectures

def get_best_architecture_adversarial(archs_path, algorithm):
    best_adv_loss = 100
    best_individual = None
    best_path = ""
    adversarial_loss_index = 1
    for arch_path in os.listdir(archs_path):
        with lzma.open(archs_path + os.sep + arch_path, 'r') as f:
            if algorithm == 'r2-emoa' or algorithm == 'nevonas':
                print(f'------ file={archs_path + os.sep + arch_path}')
                individual = pickle.load(f)
                if individual.F[adversarial_loss_index] < best_adv_loss:
                    best_adv_loss = individual.F[adversarial_loss_index]
                    best_individual = individual
                    best_path = archs_path + os.sep + arch_path
            elif algorithm == 'nsganet':
                individual_pair = pickle.load(f)
                individual_genotype = individual_pair[0]
                F = individual_pair[1]
                if F[adversarial_loss_index] < best_adv_loss:
                    best_adv_loss = F[adversarial_loss_index]
                    best_individual = individual_genotype
                    best_path = archs_path + os.sep + arch_path
            else:
                raise Exception("Algorithm not recognized")
    return best_individual, best_path