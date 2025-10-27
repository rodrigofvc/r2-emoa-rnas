import math
import time
import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
SEED = 777
BATCH = 96
EPOCHS = 600
DATA_DIR = "./data"
NUM_CLASSES = 10
INIT_CHANNELS = 36
STEM_MULTIPLIER = 3
WEIGHT_DECAY = 3e-4
MOMENTUM = 0.9
BASE_LR = 0.05
MIN_LR = 0.0
CLIP_GRAD = 5.0
ADV_EPS = 8/255.0
ADV_STEPS = 7
ADV_STEP_SIZE = 2/255.0
ADV_RATIO = 1.0            # 1.0 = puramente adversario; 0.5 = mezcla 50/50

# dispositivo (cuda > mps > cpu)
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                            else "cpu"))

# ----------------------------------------------------------
# PRIMITIVES/OPS mínimos para fase discreta
# (debes alinear estos nombres con los de tu búsqueda)
# ----------------------------------------------------------

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    def forward(self, x): return self.op(x)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

def sep_conv(C_in, C_out, k, stride, affine=True, dilation=1):
    padding = ((k - 1) // 2) * dilation
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, k, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_in, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(C_in, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, k, stride=1, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

def dil_conv(C_in, C_out, k, stride, dilation, affine=True):
    padding = ((k - 1) // 2) * dilation
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_in, k, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

OPS = {
    "none": lambda C, stride: nn.Identity(),
    "skip_connect": lambda C, stride: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=False),
    "avg_pool_3x3": lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3": lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    "sep_conv_3x3": lambda C, stride: sep_conv(C, C, 3, stride, affine=False),
    "sep_conv_5x5": lambda C, stride: sep_conv(C, C, 5, stride, affine=False),
    "dil_conv_3x3": lambda C, stride: dil_conv(C, C, 3, stride, dilation=2, affine=False),
    "dil_conv_5x5": lambda C, stride: dil_conv(C, C, 5, stride, dilation=2, affine=False),
}

# ----------------------------------------------------------
# Celdas DISCRETAS (una sola operación por arista)
# ----------------------------------------------------------

class DiscreteEdgeOp(nn.Module):
    """Operación discreta elegida para una arista concreta."""
    def __init__(self, C, stride, op_name: str):
        super().__init__()
        self.op_name = op_name
        self.op = OPS[op_name](C, stride)

    def forward(self, x):
        return self.op(x)

class DiscreteCell(nn.Module):
    """
    Celda discreta:
      - s0 (nodo 0), s1 (nodo 1)
      - nodos internos: 2..(2+steps-1)
      - edges: lista de tuplas (op_name, src, dst)
      - concat: indices de nodos internos a concatenar (por canales)
    """
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev,
                 edges: List[Tuple[str, int, int]], concat: List[int]):
        super().__init__()
        self._steps = steps
        self._multiplier = multiplier
        self.reduction = reduction
        self.concat = concat

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        # construir ops discretas por arista existente
        self.in_edges_per_internal = [[] for _ in range(self._steps)]
        self.ops = nn.ModuleList()
        for (op_name, src, dst) in sorted(edges, key=lambda t: (t[2], t[1])):  # por dst, luego src
            stride = 2 if (reduction and src in (0, 1)) else 1
            op = DiscreteEdgeOp(C, stride, op_name)
            op_idx = len(self.ops)
            self.ops.append(op)
            self.in_edges_per_internal[dst - 2].append((src, op_idx))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]  # indices 0 y 1
        for i in range(self._steps):
            dst = 2 + i
            acc = 0.0
            for (src, op_idx) in self.in_edges_per_internal[i]:
                acc = acc + self.ops[op_idx](states[src])
            states.append(acc)
        outs = [states[i] for i in self.concat]  # típicamente [2,3,4,5]
        return torch.cat(outs, dim=1)

class DiscreteNetwork(nn.Module):
    """
    Red final discreta a partir de 'genos' (lista por celda).
    Cada item: {"reduction": bool, "edges":[(op,src,dst),...], "concat":[...]}
    """
    def __init__(self, C, num_classes, layers, steps, multiplier, genos: List[Dict], stem_multiplier=STEM_MULTIPLIER):
        super().__init__()
        assert len(genos) == layers
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        C_stem = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )

        self.cells = nn.ModuleList()
        C_prev_prev, C_prev, C_curr = C_stem, C_stem, C
        prev_reduction = False
        for layer_idx in range(layers):
            g = genos[layer_idx]
            is_red = bool(g["reduction"])
            cell = DiscreteCell(
                steps=self._steps,
                multiplier=self._multiplier,
                C_prev_prev=C_prev_prev,
                C_prev=C_prev,
                C=C_curr,
                reduction=is_red,
                reduction_prev=prev_reduction,
                edges=g["edges"],
                concat=g["concat"],
            )
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, self._multiplier * C_curr
            if is_red:
                C_curr *= 2
            prev_reduction = is_red

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s = cell(s0, s1)
            s0, s1 = s1, s
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

# ----------------------------------------------------------
# Ataque PGD (l_infinity)
# ----------------------------------------------------------
@torch.no_grad()
def clamp(x, lower=0.0, upper=1.0):
    return x.clamp(lower, upper)

def pgd_attack(model, x, y, eps=ADV_EPS, alpha=ADV_STEP_SIZE, steps=ADV_STEPS):
    model.eval()
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        # proyectar al epsilon-ball
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = clamp(x_adv, 0.0, 1.0)
    model.train()
    return x_adv.detach()

# ----------------------------------------------------------
# Datos (CIFAR-10)
# ----------------------------------------------------------
def get_loaders(batch=BATCH, data_dir=DATA_DIR):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_t)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_t)
    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

# ----------------------------------------------------------
# Entrenamiento adversario
# ----------------------------------------------------------
def adjust_cosine(optimizer, epoch, total_epochs, base_lr=BASE_LR, min_lr=MIN_LR):
    """CosineAnnealing manual (una línea) para claridad."""
    lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*epoch/total_epochs))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

def train_epoch(model, loader, optimizer, epoch, total_epochs):
    model.train()
    correct, total, loss_meter = 0, 0, 0.0
    lr = adjust_cosine(optimizer, epoch, total_epochs, BASE_LR, MIN_LR)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # generar adversario
        x_adv = pgd_attack(model, x, y, eps=ADV_EPS, alpha=ADV_STEP_SIZE, steps=ADV_STEPS)

        optimizer.zero_grad(set_to_none=True)
        logits_nat = model(x)
        logits_adv = model(x_adv)
        loss_nat = F.cross_entropy(logits_nat, y)
        loss_adv = F.cross_entropy(logits_adv, y)
        loss = (1.0 - ADV_RATIO)*loss_nat + ADV_RATIO*loss_adv
        loss.backward()
        if CLIP_GRAD is not None:
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        with torch.no_grad():
            pred = logits_adv.argmax(dim=1) if ADV_RATIO >= 0.5 else logits_nat.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_meter += loss.item() * y.size(0)

    return loss_meter/total, correct/total, lr

@torch.no_grad()
def evaluate(model, loader, adversarial=False):
    model.eval()
    correct, total, loss_meter = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if adversarial:
            x = pgd_attack(model, x, y, eps=ADV_EPS, alpha=ADV_STEP_SIZE, steps=ADV_STEPS)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_meter += loss.item() * y.size(0)
    return loss_meter/total, correct/total

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def set_seed(s=SEED):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main(genos: List[Dict]):
    set_seed(SEED)
    cudnn.benchmark = True
    train_loader, test_loader = get_loaders()

    # deduce steps & multiplier de un genotipo (todos comparten)
    any_cell = genos[0]
    steps = max(d for _,_,d in any_cell["edges"]) - 1  # nodos internos típicamente 4 → dst in [2..5]
    multiplier = len(any_cell["concat"])

    model = DiscreteNetwork(
        C=INIT_CHANNELS,
        num_classes=NUM_CLASSES,
        layers=len(genos),
        steps=steps,
        multiplier=multiplier,
        genos=genos,
        stem_multiplier=STEM_MULTIPLIER,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    print(f"=> device: {device} | params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    best_nat = 0.0
    best_adv = 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc, lr = train_epoch(model, train_loader, optimizer, epoch, EPOCHS)
        nat_loss, nat_acc = evaluate(model, test_loader, adversarial=False)
        adv_loss, adv_acc = evaluate(model, test_loader, adversarial=True)
        dt = time.time() - t0

        best_nat = max(best_nat, nat_acc)
        best_adv = max(best_adv, adv_acc)

        print(f"[{epoch:03d}] lr={lr:.5f} "
              f"| train: loss={tr_loss:.3f} acc={tr_acc*100:.2f}% "
              f"| nat: loss={nat_loss:.3f} acc={nat_acc*100:.2f}% "
              f"| adv: loss={adv_loss:.3f} acc={adv_acc*100:.2f}% "
              f"| {dt:.1f}s "
              f"| best_nat={best_nat*100:.2f}% best_adv={best_adv*100:.2f}%")

if __name__ == "__main__":
    # EJEMPLO: genos ficticio (reemplázalo por el que obtuviste con tu búsqueda)
    # Nota: todos los dst deben pertenecer a [2 .. 2+steps-1] y 'concat' coincidir.
    example_genos = [
        {"reduction": False,
         "edges": [("sep_conv_3x3", 0, 2), ("skip_connect", 1, 2),
                   ("sep_conv_3x3", 0, 3), ("sep_conv_3x3", 2, 3),
                   ("sep_conv_3x3", 1, 4), ("skip_connect", 2, 4),
                   ("sep_conv_3x3", 0, 5), ("skip_connect", 3, 5)],
         "concat": [2,3,4,5]},
        {"reduction": False,
         "edges": [("sep_conv_3x3", 0, 2), ("skip_connect", 1, 2),
                   ("sep_conv_3x3", 0, 3), ("sep_conv_3x3", 2, 3),
                   ("sep_conv_3x3", 1, 4), ("skip_connect", 2, 4),
                   ("sep_conv_3x3", 0, 5), ("skip_connect", 3, 5)],
         "concat": [2,3,4,5]},
        {"reduction": True,
         "edges": [("max_pool_3x3", 0, 2), ("max_pool_3x3", 1, 2),
                   ("skip_connect", 0, 3), ("max_pool_3x3", 1, 3),
                   ("max_pool_3x3", 0, 4), ("skip_connect", 1, 4),
                   ("skip_connect", 0, 5), ("max_pool_3x3", 1, 5)],
         "concat": [2,3,4,5]},
        # ... añade más celdas hasta layers deseado
    ]
    main(example_genos)
