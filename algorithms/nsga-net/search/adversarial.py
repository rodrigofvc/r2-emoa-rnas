import torch
import torch.nn.functional as F

def fgsm_simple_normalized(model, x, y, eps=8 / 255):
    x_adv = x.detach().clone().requires_grad_(True)

    std_logits = model(x_adv)
    std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, retain_graph=True, create_graph=False)[0]
    adv = (x_adv + eps * grad.sign()).clamp(0.0, 1.0).detach().clone()
    return adv, std_logits

def fgsm_simple(model, x, y, eps=8/255):
    """
    FGSM for inputs previously normalized as:

        x_normalized = (x_pixel - mean) / std

    Parameters
    ----------
    model:
        Model that receives normalized inputs.
    x:
        Normalized input tensor with shape (N, C, H, W).
    y:
        True labels.
    eps:
        Maximum L-infinity perturbation in the original pixel space [0, 1].

    Returns
    -------
    adv:
        Adversarial inputs in normalized space.
    std_logits:
        Clean logits calculated from normalized inputs.
    """
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]

    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    mean = x.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)

    std = x.new_tensor(CIFAR_STD).view(1, 3, 1, 1)

    # Convert epsilon from pixel space to normalized space.
    eps_normalized = eps / std

    # Pixel bounds [0,1] expressed in normalized space.
    lower_bound = (0.0 - mean) / std
    upper_bound = (1.0 - mean) / std

    x_adv = x.detach().clone().requires_grad_(True)

    std_logits = model(x_adv)
    std_loss = F.cross_entropy(std_logits, y)
    grad = torch.autograd.grad(std_loss, x_adv, retain_graph=True, create_graph=False)[0]
    adv = x_adv + eps_normalized * grad.sign()
    adv = torch.maximum(torch.minimum(adv, upper_bound), lower_bound).detach()

    return adv, std_logits