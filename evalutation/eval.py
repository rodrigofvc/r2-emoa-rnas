from genotypes import PRIMITIVES
from operations import *
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    """
    steps: numero de nodos internos
    multiplier: numero de nodos de salida a concatenar
    C_prev_prev: numero de canales en cell[-2]
    C_prev: numero de canales en cell[-1]
    C: numero de canales en la celda actual
    reduction: es esta una celda de reduccion
    reduction_prev: fue la celda previa una celda de reduccion
    graph: dict[nodes, list[nodes]]
      0 -> s0 (input_prev_prev)
      1 -> s1 (input_prev)
      2,3... (steps) -> intern nodes
      2..(2+steps-1) -> nodos internos x^(0..steps-1)
    weights_graph: dict[(src, dst)] ->
        (0,1) -> tensor[len(PRIMITIVES)]
    """
    def __init__(self, steps, multiplier,
                 C_prev_prev, C_prev, C,
                 reduction, reduction_prev,
                 graph):
        super().__init__()
        self._steps = steps
        self._multiplier = multiplier
        self.reduction = reduction
        self.primitives = PRIMITIVES

        # Preprocesado para casar canales con C
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)  # para s0
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)        # para s1

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        # Por nodo destino interno i (0..steps-1), guardamos lista de (src, op_idx)
        # para el i-esimo nodo, se guarda el indice de operacion op_idx del j-esimo vecino
        self.in_edges_per_internal = [[] for _ in range(self._steps)]

        # Normalizamos graph a un conjunto de aristas válidas
        for src, dst_list in graph.items():
            for dst in dst_list:

                stride = 2 if (self.reduction and src in (0, 1)) else 1

                op = MixedOp(C, stride)
                op_idx = len(self._ops)
                self._ops.append(op)
                #self.edge_to_opidx[(src, dst)] = op_idx

                # Convertimos dst "global" a indice interno de nodo: i = dst - 2
                # dst-2 porque 0 y 1 son s0 y s1
                i = dst - 2
                self.in_edges_per_internal[i].append((src, op_idx))

        for i in range(self._steps):
            self.in_edges_per_internal[i].sort(key=lambda t: t[0])

    def forward(self, s0, s1, weight_graph):
        # Preprocesar entradas para tener C canales
        s0 = self.preprocess0(s0)  # nodo global 0
        s1 = self.preprocess1(s1)  # nodo global 1

        # states[k] contiene el tensor del nodo global k (k=0..1 ya definidos; luego 2..)
        states = {0: s0, 1: s1}

        # Crear nodos internos
        for i in range(self._steps):
            dst = 2 + i
            acc = 0.0
            for (src, op_idx) in self.in_edges_per_internal[i]:
                if src not in states:
                    raise ValueError(f"Arista ({src}->{dst}) inválida: el src aún no existe.")
                h = states[src]

                # Pesos de la arista (src->dst)
                try:
                    w = weight_graph[(src, dst)]
                except KeyError:
                    raise KeyError(f"Faltan pesos para la arista ({src}->{dst}).")

                y = self._ops[op_idx](h, w)
                acc = acc + y

            states[dst] = acc

        outs = [states[2 + i] for i in range(self._steps)]
        outs = outs[-self._multiplier:]
        return torch.cat(outs, dim=1)

