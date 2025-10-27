import torch
import torch.nn.functional as F

PRIMITIVES = ['sep_conv_3x3', 'sep_conv_5x5', 'max_pool_3x3', 'skip_connect']

edges_dict = {
    0: [2, 3],
    1: [2]
}

# parámetros de arquitectura (α) para cada arista
alpha_dict = {
    (0, 2): torch.nn.Parameter(torch.randn(len(PRIMITIVES))),
    (1, 2): torch.nn.Parameter(torch.randn(len(PRIMITIVES))),
    (0, 3): torch.nn.Parameter(torch.randn(len(PRIMITIVES))),
}

# convertir α → pesos blandos (softmax)
weights_dict = {
    key: F.softmax(alpha, dim=-1)
    for key, alpha in alpha_dict.items()
}

for k, w in weights_dict.items():
    print(f"{k}: {w.shape}, sum={w.sum():.3f}")
