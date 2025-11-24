import numpy as np


def dominates(ind1, ind2):
    if np.allclose(ind1.F, ind2.F, atol=1e-8):
        return False
    return all(f1 <= f2 for f1, f2 in zip(ind1.F, ind2.F))



# Return non-dominated points in archive
def archive_update_pq(archive, population):
    for ind in population:
        dominated = False
        to_remove = []
        for i, arch_ind in enumerate(archive):
            if dominates(arch_ind, ind):
                dominated = True
                break
            elif dominates(ind, arch_ind):
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
