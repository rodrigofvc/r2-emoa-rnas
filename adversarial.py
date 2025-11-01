from fractions import Fraction

import torchattacks


def get_attack_function(attack_params):
    attack_params['params']['eps'] = float(Fraction(attack_params['params']['eps'])) if '/' in attack_params['params']['eps'] else float(attack_params['params']['eps'])
    if attack_params['name'] == 'FGSM':
        attack_function = lambda model: torchattacks.FGSM(model, **attack_params['params'])
    elif attack_params['name'] == 'PGD':
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function