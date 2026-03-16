import torch
import torch.nn.functional as F


def fgsm_simple(model, x, y, eps=8/255):
    x_adv = x.detach().clone().contiguous().requires_grad_(True)
    with torch.enable_grad():
        std_logits = model(x_adv)
        std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, only_inputs=True, retain_graph=False, create_graph=False)[0]
    adv = (x_adv + eps * grad.detach().sign()).clamp(0.0, 1.0).detach().clone().contiguous()
    del grad, std_loss, std_logits, x_adv
    return adv

