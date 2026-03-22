import torch
import torch.nn.functional as F

# Set the model to training mode for all layers, including BatchNorm and Dropout
def set_model_mode(model, training):
    for m in model.modules():
        m.__dict__['training'] = training

# Set the model to attack mode: BatchNorm and Dropout layers are set to evaluation mode, while other layers are set to training mode
def set_attack_mode(model, training):
    for m in model.modules():
        if 'BatchNorm' in m.__class__.__name__ or 'Dropout' in m.__class__.__name__:
            m.__dict__['training'] = False
        else:
            m.__dict__['training'] = training

def fgsm_simple(model, x, y, eps=8/255):
    x_adv = x.detach().requires_grad_(True)
    with torch.enable_grad():
        std_logits = model(x_adv)
        std_loss = F.cross_entropy(std_logits, y)

    grad = torch.autograd.grad(std_loss, x_adv, retain_graph=False, create_graph=False)[0]
    adv = (x_adv + eps * grad.detach().sign()).clamp(0.0, 1.0).detach()
    return adv
