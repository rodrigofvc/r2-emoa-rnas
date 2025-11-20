from evaluation.genotypes import PRIMITIVES, Genotype
from evaluation.operations import *
from torch import nn
import torch.nn.functional as F


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
    weights = [w.to(x.device) for w in weights]
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
    """
    steps: numero de nodos internos
    multiplier: numero de nodos de salida a concatenar
    C_prev_prev: numero de canales en cell[-2]
    C_prev: numero de canales en cell[-1]
    C: numero de canales en la celda actual
    reduction: indica si es esta una celda de reduccion
    reduction_prev: fue la celda previa una celda de reduccion
    """
    def __init__(self, steps, multiplier,
                 C_prev_prev, C_prev, C,
                 reduction, reduction_prev):
        super().__init__()
        self._steps = steps
        self._multiplier = multiplier
        self.reduction = reduction
        self.primitives = PRIMITIVES

        # Preprocesado para casar canales con C
        # preprocess0 para s0 (cell[-2])
        # preprocess1 para s1 (cell[-1])
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)  # para s0
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()

        for i in range(self._steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        # i = 0, j = 0,1 sum(s_0,s_1)
        # i = 1, j = 0,1,2 sum(s_0,s_1,x0)
        # i = 2, j = 0,1,2,3 sum(s_0,s_1,x0,x1)
        # i = 3, j = 0,1,2,3,4 sum(s_0,s_1,x0,x1,x2)
        for i in range(self._steps):
            s = sum(self._ops[offset + j](states[j], weights[offset + j]) for j in range(i + 2))
            offset += i + 2
            states.append(s)
        return torch.cat(states[-self._multiplier:], dim=1)

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

    def __init__(self, C, num_classes, layers, criterion, steps, multiplier, stem_multiplier=3, device='cuda'):
        super().__init__()
        self.C = C
        self.num_classes = num_classes
        self.layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self._stem_multiplier = stem_multiplier
        self._device = device
        k = sum(1 for i in range(self._steps) for _ in range(2 + i))
        num_ops = len(PRIMITIVES)
        self.alphas_dim = (k, num_ops)

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

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
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(self._device), requires_grad=False)
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops).to(self._device), requires_grad=False)

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion,
                            self._steps, self._multiplier, self._stem_multiplier, self._device).to(self._device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.detach())
        return model_new

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def update_arch_parameters(self, new_alphas):
        assert new_alphas[0].is_cuda == self.alphas_normal.is_cuda
        assert new_alphas[1].is_cuda == self.alphas_reduce.is_cuda
        with torch.no_grad():
            self.alphas_normal.copy_(new_alphas[0])
            self.alphas_reduce.copy_(new_alphas[1])

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
                weights = F.softmax(self.alphas_reduce, dim=-1).to(self._device)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1).to(self._device)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def genotype(self):

        def _parse(weights):
          gene = []
          n = 2
          start = 0
          for i in range(self._steps):
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
            for j in edges:
              k_best = None
              for k in range(len(W[j])):
                if k != PRIMITIVES.index('none'):
                  if k_best is None or W[j][k] > W[j][k_best]:
                    k_best = k
              gene.append((PRIMITIVES[k_best], j))
            start = end
            n += 1
          return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
          normal=gene_normal, normal_concat=concat,
          reduce=gene_reduce, reduce_concat=concat
        )
        return genotype



"""
    Return a genotype representation of the model
    where each dict in genotype_cells corresponds to a cell in the model with keys:
    - "reduction": bool, whether the cell is a reduction cell
    - "edges": list of tuples (op_name, src, dst) representing the operations and their connections
    - "concat": list of node indices to concatenate as output (2,.., 2 + multiplier - 1) 0 and 1 are the two input nodes
    - "steps": number of intermediate nodes in the cell
    - "multiplier": number of nodes to concatenate for the cell output
"""
def genotype(model, PRIMITIVES):
    genotype_cells = []
    for layer_idx, cell in enumerate(model.cells):
        if cell.reduction:
            pd = model.alphas_reduce
        else:
            pd = model.alphas_normal
        get_alpha = lambda s,d: pd[f"{s}->{d}"]
        edges_spec = []
        for src, dsts in model.cells_edges[layer_idx].items():
            for dst in dsts:
                alpha = get_alpha(src, dst).detach()
                op_idx = int(alpha.argmax().item())
                op_name = PRIMITIVES[op_idx]
                edges_spec.append((op_name, int(src), int(dst)))
        genotype_cells.append({
            "reduction": cell.reduction,
            "edges": edges_spec,
            "concat": list(range(2, 2 + cell._multiplier)),
            "steps": cell._steps,
            "multiplier": cell._multiplier
        })
    return genotype_cells




def discretize(alphas, arch_genotype, device):
    normal_cell = arch_genotype.normal
    reduction_cell = arch_genotype.reduce

    # Discretizing the normal cell
    index = 0
    offset = 0
    new_normal = torch.zeros_like(alphas[0]).to(device)
    while index < len(normal_cell):
        op, cell = normal_cell[index]
        idx = PRIMITIVES.index(op)
        new_normal[int(offset + cell)][idx] = 1
        index += 1
        op, cell = normal_cell[index]
        idx = PRIMITIVES.index(op)
        new_normal[int(offset + cell)][idx] = 1
        offset += (index // 2) + 2
        index += 1

    # Discretizing the reduction cell
    index = 0
    offset = 0
    new_reduce = torch.zeros_like(alphas[1]).to(device)
    while index < len(reduction_cell):
        op, cell = reduction_cell[index]
        idx = PRIMITIVES.index(op)
        new_reduce[int(offset + cell)][idx] = 1
        index += 1
        op, cell = reduction_cell[index]
        idx = PRIMITIVES.index(op)
        new_reduce[int(offset + cell)][idx] = 1
        offset += (index // 2) + 2
        index += 1
    return [new_normal, new_reduce]