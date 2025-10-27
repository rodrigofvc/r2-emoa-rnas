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
      steps_cells: nodos intermedios por celda (e.g., 4)
      multiplier_cells: cuantos nodos concatenar por celda
      cells_edges: lista de dicts (len == layers).
      reduction_layers: lista con indices de celdas que son de reduccion.
                        Si None, usa posiciones: floor(l/3) y floor(2l/3).
      stem_multiplier: multiplicador de canales en el stem (por defecto 3 como en DARTS)
    """
    def __init__(self, C, num_classes, layers, criterion, steps_cells, multiplier_cells,
                 cells_edges, reduction_layers=None, stem_multiplier=3, fairdarts_eval=False):
        super().__init__()
        assert len(cells_edges) == layers, "cells_edges debe tener un dict por celda"
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps_cells = steps_cells
        self._multiplier_cells = multiplier_cells
        self._cells_edges = cells_edges
        self._stem_multiplier = stem_multiplier
        self.fairdarts_eval = fairdarts_eval

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
        self._reduction_layers = reduction_layers

        C_prev_prev, C_prev, C_curr = C_stem, C_stem, C


        prev_reduction = False
        for layer_idx in range(layers):
            is_reduction = layer_idx in reduction_layers

            cell = Cell(
                steps=self._steps_cells,
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
            C_prev_prev, C_prev = C_prev, self._multiplier_cells * C_curr
            if is_reduction:
                # si es celda de reduccion, la salida sera multiplier*(2*C_curr)
                C_curr *= 2
            prev_reduction = is_reduction

        normals_idx = [i for i in range(layers) if i not in reduction_layers]
        reduces_idx = [i for i in range(layers) if i in reduction_layers]
        union_norm = edges_union([cells_edges[i] for i in normals_idx]) if normals_idx else []
        union_red = edges_union([cells_edges[i] for i in reduces_idx]) if reduces_idx else []

        self.alphas_normal = nn.ParameterDict({
            edge_key(s, d): nn.Parameter(1e-3 * torch.randn(len(PRIMITIVES)))
            for (s, d) in union_norm
        })
        self.alphas_reduce = nn.ParameterDict({
            edge_key(s, d): nn.Parameter(1e-3 * torch.randn(len(PRIMITIVES)))
            for (s, d) in union_red
        })

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
        return list(self.alphas_normal.parameters()) + list(self.alphas_reduce.parameters())

    def weight_parameters(self):
        arch_set = {id(p) for p in self.arch_parameters()}
        for p in self.parameters():
            if id(p) not in arch_set:
                yield p

    def _make_weights_for_cell(self, cell, layer_idx):
        pd = self.alphas_reduce if cell.reduction else self.alphas_normal
        weights = {}
        for src, dsts in self.cells_edges[layer_idx].items():
            for dst in dsts:
                k = edge_key(src, dst)
                if k not in pd:
                    raise ValueError(f"Edge {k} not in parameter dict for layer {layer_idx}")
                alpha_vec = pd[k]
                if self.fairdarts_eval:
                    probs = torch.sigmoid(alpha_vec)      # FairDARTS
                else:
                    probs = F.softmax(alpha_vec, dim=-1)  # DARTS
                weights[(int(src), int(dst))] = probs
        return weights

    def _loss(self, x, target):
        logits = self(x)
        return self._criterion(logits, target)

        # --- forward ---
    def forward(self, x):
        # Capa inicial que usa el mismo estado para s0 y s1
        s0 = s1 = self.stem(x)
        # Para cada celda, construye los pesos y ejecuta
        for layer_idx, cell in enumerate(self.cells):
            weights_dict = self._make_weights_for_cell(cell, layer_idx)
            # Ejecutar la celda con su propia grafica y pesos
            s = cell(s0, s1, weights_dict)
            s0, s1 = s1, s
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

