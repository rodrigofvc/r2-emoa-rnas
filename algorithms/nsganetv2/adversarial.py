from fractions import Fraction
import torch
import torchattacks
import torch.nn.functional as F


def fgsm_simple(model, x, y, eps):
    assert x.requires_grad, "Input tensor must have requires_grad=True for fgsm_simple attack"
    std_logits = model(x)
    std_loss = F.cross_entropy(std_logits, y)
    grad = torch.autograd.grad(std_loss, x, retain_graph=True, create_graph=False)[0]
    adv = (x + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return adv, std_logits, std_loss

class FGSMAttack:
    def __init__(self, eps=8/255):
        self.eps = eps

    def __call__(self, model, x, y):
        return fgsm_simple(model, x, y, self.eps)


def get_attack_function(attack_params):
    attack_params['params']['eps'] = float(Fraction(attack_params['params']['eps'])) if '/' in attack_params['params']['eps'] else float(attack_params['params']['eps'])
    if 'alpha' in attack_params['params']:
        attack_params['params']['alpha'] = float(Fraction(attack_params['params']['alpha'])) if '/' in attack_params['params']['alpha'] else float(attack_params['params']['alpha'])
    if attack_params['name'] == 'FGSM':
        atk = FGSMAttack(attack_params['params']['eps'])
        return lambda model: lambda x, y: atk(model, x, y)
    elif 'PGD' in attack_params['name']:
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function