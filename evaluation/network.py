import torch
from torch import nn
import torch.nn.functional as F

from eval import Cell
from genotypes import PRIMITIVES
from operations import *
from torch.autograd import Variable


def edge_key(src, dst):
    return f"{src}->{dst}"

"""
    returns a set of edges (src,dst) for a list of cells dicts 
"""
def edges_union(cells_edges_idx):
    s = set()
    for ed in cells_edges_idx:
        for src, dsts in ed.items():
            for dst in dsts:
                s.add((int(src), int(dst)))
    return sorted(s, key=lambda t: (t[1], t[0]))

class Network(nn.Module):
    """
    Network (búsqueda) que acepta una gráfica por celda.

    Args:
      C: canales base del stem
      num_classes: #clases
      layers: numero de celdas
      steps: nodos intermedios por celda (e.g., 4)
      multiplier_cells: cuantos nodos concatenar por celda
      reduction_layers: lista con indices de celdas que son de reduccion.
                        Si None, usa posiciones: floor(l/3) y floor(2l/3).
      stem_multiplier: multiplicador de canales en el stem (por defecto 3 como en DARTS)
    """
    def __init__(self, C, num_classes, layers, criterion, steps, multiplier_cells,
                 reduction_layers=None, stem_multiplier=3, fairdarts_eval=False, device='cuda'):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier_cells = multiplier_cells
        self._stem_multiplier = stem_multiplier
        self._fairdarts_eval = fairdarts_eval
        self._device = device

        C_stem = stem_multiplier * C
        # Capa inicial que convierte los 3 canales a C_stem en el foward
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )

        if reduction_layers is None:
            r1 = layers // 3
            r2 = (2 * layers) // 3
            reduction_layers = {r1, r2}
        else:
            reduction_layers = set(reduction_layers)

        self.cells = nn.ModuleList()
        self._reduction_layers = reduction_layers

        C_prev_prev, C_prev, C_curr = C_stem, C_stem, C


        prev_reduction = False
        for layer_idx in range(layers):
            is_reduction = layer_idx in reduction_layers

            cell = Cell(
                steps=self._steps_cells,
                multiplier=self._multiplier_cells,
                C_prev_prev=C_prev_prev,
                C_prev=C_prev,
                C=C_curr,
                reduction=is_reduction,
                reduction_prev=prev_reduction
            )
            self.cells.append(cell)

            # En celda normal, el numero de celdas de salida es multiplier*C_curr (nodos * canales)
            C_prev_prev, C_prev = C_prev, self._multiplier_cells * C_curr
            if is_reduction:
                # si es celda de reduccion, la salida sera multiplier*(2*C_curr)
                C_curr *= 2
            prev_reduction = is_reduction


        # Clasificador final (GAP + FC)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def _initialize_alphas(self):
        # i = 0, j = 0,1 sum(s_0,s_1)
        # i = 1, j = 0,1,2 sum(s_0,s_1,x0)
        # i = 2, j = 0,1,2,3 sum(s_0,s_1,x0,x1)
        # i = 3, j = 0,1,2,3,4 sum(s_0,s_1,x0,x1,x2)
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_reduce = (1e-3 * torch.randn(k, num_ops).to(self._device)).requires_grad_()
        self.alphas_normal = (1e-3 * torch.randn(k, num_ops).to(self._device)).requires_grad_()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                            self._steps,self._multiplier, self._reduction_layers,
                            self._stem_multiplier, self._fairdarts_eval, self._device).to(self._device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.detach())
        return model_new

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def weight_parameters(self):
        arch_set = {id(p) for p in self.arch_parameters()}
        for p in self.parameters():
            if id(p) not in arch_set:
                yield p

    def _loss(self, x, target):
        logits = self(x)
        return self._criterion(logits, target)


    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
