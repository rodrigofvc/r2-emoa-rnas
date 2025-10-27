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
    Devuelve una lista de pares (src,dst) ordenados a partir de un dict {src: [dst,...]}.
"""
def edges_from_dict(edges_dict):
    pairs = []
    for src, dsts in edges_dict.items():
        for dst in dsts:
            pairs.append((src, dst))
    pairs.sort(key=lambda t: (t[1], t[0]))  # primero por dst, luego por src
    return pairs

class Network(nn.Module):
    """
    Network (búsqueda) que acepta una gráfica por celda.

    Args:
      C: canales base del stem
      num_classes: #clases
      layers: numero de celdas
      steps_cells: nodos intermedios por celda (e.g., 4)
      multiplier_cells: cuantos nodos concatenar por celda (típicamente = steps)
      cells_edges: lista de dicts (len == layers). Cada item es el edges_dict de ESA celda:
                   {src: [dst,...]}, con convención 0=s0,1=s1,2.. internos y src<dst.
      reduction_layers: set/list con indices de celdas que son de reduccion .
                        Si None, usa posiciones canon: floor(l/3) y floor(2l/3).
      stem_multiplier: multiplicador de canales en el stem (por defecto 3 como en DARTS)
    """
    def __init__(self, C, num_classes, layers, criterion, steps_cells, multiplier_cells,
                 cells_edges, reduction_layers=None, stem_multiplier=3):
        super().__init__()
        assert len(cells_edges) == layers, "cells_edges debe tener un dict por celda"
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps_cells = steps_cells
        self._multiplier_cells = multiplier_cells
        self._cells_edges = cells_edges
        self._reduction_layers = reduction_layers
        self._stem_multiplier = stem_multiplier

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
        self.cells_edges = cells_edges

        C_prev_prev, C_prev, C_curr = C_stem, C_stem, C


        prev_reduction = False
        for layer_idx in range(layers):
            is_reduction = layer_idx in reduction_layers

            cell = Cell(
                steps=self._steps_cells[layer_idx],
                multiplier=self._multiplier_cells[layer_idx],
                C_prev_prev=C_prev_prev,
                C_prev=C_prev,
                C=C_curr,
                reduction=is_reduction,
                reduction_prev=prev_reduction,
                graph=cells_edges[layer_idx]
            )
            self.cells.append(cell)

            # En celda normal, el numero de celdas de salida es multiplier*C_curr (nodos * canales)
            C_prev_prev, C_prev = C_prev, self._multiplier_cells[layer_idx] * C_curr
            if is_reduction:
                # DARTS suele doblar C tras reducción (opcional; puedes mantenerlo)
                # si es celda de reduccion, la salida sera multiplier*(2*C_curr)
                C_curr *= 2
            prev_reduction = is_reduction

        self.alphas = nn.ModuleList()
        for layer_idx in range(layers):
            pd = nn.ParameterDict()
            pairs = edges_from_dict(cells_edges[layer_idx])
            for (src, dst) in pairs:
                pd[edge_key(src, dst)] = nn.Parameter(
                    1e-3 * torch.randn(len(PRIMITIVES))
                )
            self.alphas.append(pd)

        # Clasificador final (GAP + FC)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)  # C_prev ya es multiplier*C_curr al final

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                            self._steps_cells,self._multiplier_cells,self._cells_edges,
                            self._reduction_layers, self._stem_multiplier)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.detach())
        return model_new

    # --- utilidades para exponer α ---
    def arch_parameters(self):
        return [p for pd in self.alphas for p in pd.parameters()]

    def weight_parameters(self):
        arch_set = {id(p) for p in self.arch_parameters()}
        for p in self.parameters():
            if id(p) not in arch_set:
                yield p


    @staticmethod
    def _softmax_paramdict(paramdict):
        out = {}
        for k, alpha in paramdict.items():
            # k es "src->dst"
            src, dst = map(int, k.split("->"))
            out[(src, dst)] = F.softmax(alpha, dim=-1)
        return out

    def _loss(self, x, target):
        logits = self(x)
        return self._criterion(logits, target)

        # --- forward ---
    def forward(self, x):
        # Capa inicial que usa el mismo estado para s0 y s1
        s0 = s1 = self.stem(x)
        # Para cada celda, construye los pesos y ejecuta
        for layer_idx, cell in enumerate(self.cells):
            weights_dict = self._softmax_paramdict(self.alphas[layer_idx])
            # Ejecutar la celda con su propia grafica y pesos
            s = cell(s0, s1, weights_dict)
            s0, s1 = s1, s

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
