from genotypes import PRIMITIVES
from operations import *
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
    graph: dict[nodes, list[nodes]]
      0 -> s0 (input_prev_prev)
      1 -> s1 (input_prev)
      2,3... (steps) -> intern nodes
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
        # preprocess0 para s0 (cell[-2])
        # preprocess1 para s1 (cell[-1])
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)  # para s0
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()

        # Por nodo destino interno i (0..steps-1), guardamos lista de (src, op_idx)
        # para el i-esimo nodo, se guarda el indice de operacion op_idx del j-esimo vecino
        self.to_edges = [[] for _ in range(self._steps)]

        # Normalizamos graph a un conjunto de aristas válidas
        for src, dst_list in graph.items():
            for dst in dst_list:
                stride = 2 if (self.reduction and src in (0, 1)) else 1
                op = MixedOp(C, stride)
                op_idx = len(self._ops)
                self._ops.append(op)
                # dst-2 porque 0 y 1 son s0 y s1 y dst >= 2
                i = dst - 2
                self.to_edges[i].append((src, op_idx))

        for i in range(self._steps):
            self.to_edges[i].sort(key=lambda t: t[0])

    def forward(self, s0, s1, weight_graph):
        # Preprocesar entradas para tener C canales
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = {0: s0, 1: s1}

        # Crear nodos internos
        for i in range(self._steps):
            dst = 2 + i
            acc = 0.0
            for (src, op_idx) in self.to_edges[i]:
                if src not in states:
                    raise ValueError(f"Edge ({src}->{dst}) invalid: src not processed.")
                h = states[src]
                # Pesos de la arista (src->dst)
                try:
                    w = weight_graph[(src, dst)]
                except KeyError:
                    raise KeyError(f"No edge ({src}->{dst}).")
                y = self._ops[op_idx](h, w)
                acc = acc + y

            states[dst] = acc

        outs = [states[2 + i] for i in range(self._steps)]
        outs = outs[-self._multiplier:]
        return torch.cat(outs, dim=1)

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
      C: canales base del stem
      num_classes: numero de clases
      layers: numero de celdas
      steps_cells: lista de nodos intermedios por celda (e.g., 4)
      multiplier_cells: lista de cuantos nodos internos concatenar por celda
      cells_edges: lista de cell dicts (len == layers).
      reduction_layers: lista con indices de celdas que son de reduccion.
                        Si None, usa posiciones: floor(len/3) y floor(2*len/3).
      stem_multiplier: multiplicador de canales en el stem (por defecto 3 como en DARTS)
      fairdarts_eval: si True, usa sigmoid en vez de softmax para las alphas (FairDARTS)
    """
    def __init__(self, C, num_classes, layers, criterion, steps_cells, multiplier_cells,
                 cells_edges, reduction_layers=None, stem_multiplier=3, fairdarts_eval=False):
        super().__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps_cells = steps_cells
        self._multiplier_cells = multiplier_cells
        self._cells_edges = cells_edges
        self._stem_multiplier = stem_multiplier
        self._fairdarts_eval = fairdarts_eval

        C_stem = stem_multiplier * C
        # Capa inicial que convierte los 3 canales a C_stem
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
                            self._steps_cells, self._multiplier_cells, self._cells_edges,
                            self._reduction_layers, self._stem_multiplier, self._fairdarts_eval)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.detach())
        return model_new

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
                if self._fairdarts_eval:
                    probs = torch.sigmoid(alpha_vec)      # FairDARTS
                else:
                    probs = F.softmax(alpha_vec, dim=-1)  # DARTS
                weights[(int(src), int(dst))] = probs
        return weights

    def _loss(self, x, target):
        logits = self(x)
        return self._criterion(logits, target)

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for layer_idx, cell in enumerate(self.cells):
            weights_dict = self._make_weights_for_cell(cell, layer_idx)
            s = cell(s0, s1, weights_dict)
            s0, s1 = s1, s
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

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