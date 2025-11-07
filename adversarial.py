from fractions import Fraction
import torch
import torchattacks

def fgsm(model, x, y, eps=8/255):
    was_training = model.training
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    model.train(was_training)
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