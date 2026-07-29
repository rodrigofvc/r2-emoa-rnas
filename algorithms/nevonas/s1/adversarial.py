import torch
import torch.nn.functional as F


def fgsm_simple(model, x, y, eps=8 / 255):
    x_adv = x.detach().clone().requires_grad_(True)

    std_logits = model(x_adv)
    std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, retain_graph=True, create_graph=False)[0]
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach().clone()
    return adv, std_logits