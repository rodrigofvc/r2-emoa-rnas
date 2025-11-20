import contextlib
from fractions import Fraction
import torch
import torchattacks
import torch.nn.functional as F
from torch import amp


def fgsm_dep(model, x, y, eps=8/255):
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    (grad,) = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return adv

def fgsm(model, x, y, eps=8/255):
    device = next(model.parameters()).device

    x_adv = (
        x.detach()
         .to(device, non_blocking=True)
         .float()
         .contiguous(memory_format=torch.contiguous_format)
         .clone()
         .requires_grad_(True)
    )
    y = y.to(device, non_blocking=True)

    try:
        from torch.amp import autocast
        amp_ctx = autocast('cuda', enabled=False)
    except ImportError:
        from torch.cuda.amp import autocast
        amp_ctx = autocast(enabled=False)

    prev_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        with amp_ctx:
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
        grad, = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn

    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return adv

def fgsm_simple(model, x, y, eps):
    x_adv = x.detach().clone().to(x.device).float().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    with amp.autocast("cuda", dtype=torch.float16):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach()
    return adv

class FGSMAttack():
    def __init__(self, model, eps=8 / 255):
        self.model = model
        self.eps = eps

    def __call__(self, x, y):
        return fgsm_simple(self.model, x, y, eps=self.eps)


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