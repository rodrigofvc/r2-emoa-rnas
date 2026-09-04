import torch
import torch.nn.functional as F


def fgsm_simple_normalized(model, x, y, eps=8 / 255):
    x_adv = x.detach().clone().requires_grad_(True)
    feasible = True
    std_logits = model(x_adv)
    std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, retain_graph=True, create_graph=False, allow_unused=True)[0]
    if grad is None:
        # If the gradient is None, it means the architecture contains unfeasible operations for gradient computation.
        adv = x_adv.detach().clone()
        feasible = False
        return adv, std_logits, feasible
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach().clone()
    if adv is None:
        # If the adversarial example is None, it means the architecture contains unfeasible operations for gradient computation.
        adv = x_adv.detach().clone()
        feasible = False
    return adv, std_logits, feasible

def fgsm_simple(model, x, y, eps=8 / 255):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]

    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    mean = x.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CIFAR_STD).view(1, 3, 1, 1)

    eps_normalized = eps / std

    lower_bound = (0.0 - mean) / std
    upper_bound = (1.0 - mean) / std

    x_adv = x.detach().clone().requires_grad_(True)

    std_logits = model(x_adv)
    std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(outputs=std_loss, inputs=x_adv, retain_graph=True, create_graph=False, allow_unused=True)[0]

    adv = x_adv + eps_normalized * grad.sign()
    adv = torch.maximum(torch.minimum(adv, upper_bound), lower_bound).detach()

    return adv, std_logits


# Fast adversarial training with random-start FGSM.
def fast_adv(model, inputs, targets, criterion, eps=8/255, alpha=10/255):
    delta = torch.empty_like(inputs).uniform_(-eps, eps)
    delta = torch.clamp(inputs + delta, 0.0, 1.0) - inputs
    delta.requires_grad_(True)

    logits = model(inputs + delta)
    loss = criterion(logits, targets)

    grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    delta = delta.detach() + alpha * grad.sign()
    delta = torch.clamp(delta, -eps, eps)

    adv_inputs = torch.clamp(inputs + delta, 0.0, 1.0).detach().clone()

    return adv_inputs