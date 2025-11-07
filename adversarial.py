from fractions import Fraction
import torch
import torchattacks
import torch.nn.functional as F

def fgsm(model, x, y, eps=8/255):
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    (grad,) = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return adv

class FGSMAttack():
    def __init__(self, model, eps=8 / 255):
        self.model = model
        self.eps = eps

    def __call__(self, x, y):
        return fgsm(self.model, x, y, eps=self.eps)


def get_attack_function(attack_params):
    attack_params['params']['eps'] = float(Fraction(attack_params['params']['eps'])) if '/' in attack_params['params']['eps'] else float(attack_params['params']['eps'])
    if 'alpha' in attack_params['params']:
        attack_params['params']['alpha'] = float(Fraction(attack_params['params']['alpha'])) if '/' in attack_params['params']['alpha'] else float(attack_params['params']['alpha'])
    if attack_params['name'] == 'FGSM':
        attack_function = lambda model: FGSMAttack(model, eps=attack_params['params']['eps'])
    elif 'PGD' in attack_params['name']:
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function