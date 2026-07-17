import numpy as np


def dominates(ind1, ind2, k, losses=True):
    if losses:
        # Use the vector with losses for dominance comparison
        if np.allclose(ind1.F[:k], ind2.F[:k], atol=1e-8):
            return False
        return all(f1 <= f2 for f1, f2 in zip(ind1.F[:k], ind2.F[:k]))
    else:
        # Use the vector with accuracies for dominance comparison (100-stc_acc, 100-adv_acc, FLOPs, params)
        if np.allclose(ind1.F_acc[:k], ind2.F_acc[:k], atol=1e-8):
            return False
        return all(f1 <= f2 for f1, f2 in zip(ind1.F_acc[:k], ind2.F_acc[:k]))


# Return non-dominated points in archive
def archive_update_pq(archive, population_, k=4, losses=True):
    population = [ind for ind in population_ if ind.feasible]
    for ind in population:
        dominated = False
        to_remove = []
        for i, arch_ind in enumerate(archive):
            if dominates(arch_ind, ind, k, losses=losses):
                dominated = True
                break
            elif dominates(ind, arch_ind, k, losses=losses):
                to_remove.append(i)
        if not dominated:
            for i in reversed(to_remove):
                archive.pop(i)
            archive.append(ind)
    return archive

def archive_update_pq_losses(archive, population_):
    population = [ind for ind in population_ if ind.feasible]
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

def archive_update_pq_accuracy(archive, population_):
    population = [ind for ind in population_ if ind.feasible]
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
