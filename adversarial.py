from fractions import Fraction
import torch
import torchattacks
import torch.nn.functional as F
from torch import amp



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

def fgsm_simple(model, x, y, eps=8/255):
    x_adv = x.detach().clone().contiguous().requires_grad_(True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.enable_grad():
        std_logits = model(x_adv)
        std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, only_inputs=True, retain_graph=False, create_graph=False)[0]
    adv = (x_adv + eps * grad.detach().sign()).clamp(0.0, 1.0).detach().clone().contiguous()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del grad, std_loss, std_logits, x_adv
    return adv

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