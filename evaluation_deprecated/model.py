from genotypes import PRIMITIVES
import torch
import torch.nn as nn
from operations import ReLUConvBN, FactorizedReduce, OPS


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
    def __init__(self, genotype, steps, multiplier,
                 C_prev_prev, C_prev, C,
                 reduction, reduction_prev):
        super().__init__()
        self.genotype = genotype
        self._steps = steps
        self._multiplier = multiplier
        self.reduction = reduction
        self.primitives = PRIMITIVES

        # Preprocesado para casar canales con C
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._ops = nn.ModuleList()
        self._in_edges = [[] for _ in range(steps)]  # por nodo interno

        for op_name, src, dst in sorted(genotype.edges, key=lambda t: (t[2], t[1])):
            stride = 2 if (reduction and src in (0, 1)) else 1
            op = OPS[op_name](C, stride, True)
            idx = len(self.ops)
            self._ops.append(op)
            self._in_edges[dst - 2].append((src, idx))

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(self._steps):
            acc = 0.0
            for (src, idx) in self._in_edges[i]:
                acc = acc + self._ops[idx](states[src])
            states.append(acc)
        outs = [states[i] for i in self.genotype.concat]
        return torch.cat(outs, dim=1)

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=True),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=True)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, genotypes, stem_multiplier=3):
        super().__init__()
        C_stem = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_stem, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_stem)
        )
        self.cells = nn.ModuleList()
        C_pp, C_p, C_c = C_stem, C_stem, C
        prev_red = False
        for i, g in enumerate(genotypes):
            reduction = bool(g["reduction"])
            steps = int(g["steps"])
            multiplier = int(g["multiplier"])
            cell = Cell(g, steps, multiplier, C_pp, C_p, C_c, reduction, prev_red)
            self.cells.append(cell)
            C_pp, C_p = C_p, multiplier * C_c
            if reduction:
                C_c *= 2
            prev_red = reduction
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_p, num_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s = cell(s0, s1)
            s0, s1 = s1, s
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits