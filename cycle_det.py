
from collections import defaultdict

def find_module_cycles(root):
    """Busca ciclos reales en la jerarquía de submódulos (padre→hijo).
    Devuelve: (has_cycle: bool, cycles: list[str] con rutas legibles)
    """
    cycles = []
    stack = [(root, iter(list(root.named_children())), [root.__class__.__name__], [id(root)])]

    while stack:
        module, it_children, path_names, path_ids = stack[-1]
        try:
            name, child = next(it_children)
        except StopIteration:
            stack.pop()
            continue

        cid = id(child)
        if cid in path_ids:
            idx = path_ids.index(cid)
            cycle_path = path_names[idx:] + [name]
            cycles.append(" -> ".join(cycle_path))
        else:
            stack.append((
                child,
                iter(list(child.named_children())),
                path_names + [name],
                path_ids + [cid]
            ))

    return (len(cycles) > 0), cycles


def find_shared_instances(root):
    """Reporta instancias de módulos que aparecen en múltiples rutas de named_modules()."""
    idx = defaultdict(list)
    for name, module in root.named_modules():
        idx[id(module)].append(name)

    shared = {mid: paths for mid, paths in idx.items() if len(paths) > 1}
    return shared


def quick_modules_uniqueness(root):
    """Imprime conteo de módulos únicos y duplicados (chequeo rápido)."""
    mods = list(root.modules())
    total = len(mods)
    unique = len(set(map(id, mods)))
    print(f"modules(): total={total}, unique={unique}, duplicates={total-unique}")


def assert_no_cycles(model):
    """Chequea automáticamente que el modelo no tenga ciclos ni instancias compartidas.
    Lanza AssertionError si encuentra alguno.
    """
    print("[CycleDetector] Checking for module cycles and shared instances...")
    quick_modules_uniqueness(model)

    has_cycle, cycles = find_module_cycles(model)
    if has_cycle:
        print("Ciclos detectados:")
        for p in cycles:
            print("  ", p)
        raise AssertionError("Se detectaron ciclos en la jerarquía de módulos.")

    shared = find_shared_instances(model)
    if shared:
        print("Instancias compartidas detectadas")
        for mid, ps in shared.items():
            print(f"  id={mid}")
            for p in ps:
                print("    -", p)
    else:
        print("Sin ciclos ni instancias compartidas detectadas.")
