import numpy as np


def dominates_dep(ind1, ind2, k):
    if np.allclose(ind1.F[:k], ind2.F[:k], atol=1e-8):
        return False
    return all(f1 <= f2 for f1, f2 in zip(ind1.F[:k], ind2.F[:k]))

def dominates(ind1, ind2, k):
    f1 = [float(x) for x in ind1.F[:k]]
    f2 = [float(x) for x in ind2.F[:k]]

    is_equal = True
    for a, b in zip(f1, f2):
        if abs(a - b) > 1e-8:
            is_equal = False
            break
    
    if is_equal:
        return False

    better_in_any = False
    for a, b in zip(f1, f2):
        if a > b:
            return False
        if a < b:
            better_in_any = True
            
    return better_in_any

# Return non-dominated points in archive
def archive_update_pq(archive, population, k=4):
    for ind in population:
        dominated = False
        to_remove = []
        for i, arch_ind in enumerate(archive):
            if dominates(arch_ind, ind, k):
                dominated = True
                break
            elif dominates(ind, arch_ind, k):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                archive.pop(i)
            archive.append(ind)
    return archive

def archive_update_pq_losses(archive, population):
    for ind in population:
        dominated = False
        to_remove = []
        for i, arch_ind in enumerate(archive):
            if ((arch_ind.adv_loss <= ind.adv_loss and
                arch_ind.std_loss <= ind.std_loss) and
                    not np.isclose(arch_ind.adv_loss, ind.adv_loss) and
                    not np.isclose(arch_ind.std_loss, ind.std_loss)):
                dominated = True
                break
            elif ((ind.adv_loss <= arch_ind.adv_loss and
                    ind.std_loss <= arch_ind.std_loss) and
                    not np.isclose(arch_ind.adv_loss, ind.adv_loss) and
                    not np.isclose(arch_ind.std_loss, ind.std_loss)):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                archive.pop(i)
            archive.append(ind)
    return archive

def archive_update_pq_accuracy(archive, population):
    for ind in population:
        dominated = False
        to_remove = []
        for i, arch_ind in enumerate(archive):
            if ((arch_ind.adv_acc >= ind.adv_acc and
                arch_ind.std_acc >= ind.std_acc) and
                    not np.isclose(arch_ind.adv_acc, ind.adv_acc) and
                    not np.isclose(arch_ind.std_acc, ind.std_acc)):
                dominated = True
                break
            elif ((ind.adv_acc >= arch_ind.adv_acc and
                    ind.std_acc >= arch_ind.std_acc) and
                  not np.isclose(arch_ind.adv_acc, ind.adv_acc) and
                  not np.isclose(arch_ind.std_acc, ind.std_acc)):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                archive.pop(i)
            archive.append(ind)
    return archive
