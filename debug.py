import torch
import inspect
from collections import deque

def inspect_model(model):
    print("\n========== INSPECTION START ==========\n")

    # --------------------------
    # 1. ATRIBUTOS DEL MODELO
    # --------------------------
    print(">>> Atributos de model :")
    attrs = dir(model)
    for attr in attrs:
        if attr.startswith("_"):
            continue
        try:
            v = getattr(model, attr)
            print(f"  model.{attr} = {type(v)}")
        except Exception as e:
            print(f"  model.{attr} -> ERROR al acceder ({e})")

    # -----------------------------------------
    # 2. SUBMÓDULOS (recursivo)
    # -----------------------------------------
    print("\n>>> Submódulos del modelo:")
    for name, module in model.named_modules():
        print(f"  {name}: {type(module)}")

    # -----------------------------------------
    # 3. PARÁMETROS
    # -----------------------------------------
    print("\n>>> Parámetros del modelo:")
    for name, p in model.named_parameters(recurse=True):
        try:
            print(f"  {name}: dtype={p.dtype}, device={p.device}, shape={tuple(p.shape)}")
        except Exception as e:
            print(f"  {name}: ERROR ({e})")

    # -----------------------------------------
    # 4. BUFFERS
    # -----------------------------------------
    print("\n>>> Buffers del modelo:")
    for name, b in model.named_buffers(recurse=True):
        try:
            print(f"  {name}: dtype={b.dtype}, device={b.device}, shape={tuple(b.shape)}")
        except Exception as e:
            print(f"  {name}: ERROR ({e})")

    # ---------------------------------------------------------
    # 5. DETECTAR ATRIBUTOS *NO* DEFINIDOS POR LA CLASE MODEL
    # ---------------------------------------------------------
    print("\n>>> Atributos NO definidos en nn.Module o Network:")

    allowed = set(dir(torch.nn.Module)) | set(dir(model.__class__))

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        if attr not in allowed:
            try:
                v = getattr(model, attr)
                print(f"  >>> ILEGAL: model.{attr} = {type(v)}")
            except Exception as e:
                print(f"  >>> ILEGAL: model.{attr} -> ERROR ({e})")

    print("\n========== INSPECTION END ==========\n")
