import pandas as pd
import matplotlib.pyplot as plt

# plot the history log in evaluations for all algorithms
def plot_evaluations_algorithms(dataset, dirs_, indicator='hv', threshold=36, plot_threshold=True):
    dirs = [d[0] for d in dirs_]
    labels = [d[1] for d in dirs_]
    files = [d[2] for d in dirs_]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'lime', 'gray', 'yellow', 'teal', 'navy', 'magenta', 'black']
    plt.figure(figsize=(10, 6))
    for file, dir_, label, color in zip(files, dirs, labels, colors):
        print(f'Reading file: {file} for directory: {dir_}')
        data = pd.read_csv(file)
        hypervolumes = data[(data['dir'] == dir_) & (data['dataset'] == dataset) & (data['indicator'] == indicator) & (data['evaluations'] <= 1240) & (data['evaluations'] >= 40)][['evaluations', 'value']]
        hypervolumes.sort_values(by='evaluations', inplace=True)
        if hypervolumes.empty:
            print(f'No data found for directory: {dir_}, dataset: {dataset}, indicator: {indicator}')
            raise ValueError(f'No data found for directory: {dir_}, dataset: {dataset}, indicator: {indicator}')
        plt.plot(hypervolumes['evaluations']- 40, hypervolumes['value'], label=label, color=color)
    plt.xlabel('Evaluations')
    plt.ylabel('Hypervolume')
    if plot_threshold:
        plt.axhline(y=threshold, color='gray', linestyle='--')
    plt.title(f'Hypervolume over evaluations for different algorithms in {dataset.upper()}')
    plt.legend(loc='upper left')
    plt.grid()
    output_file = 'evaluations_' + dataset + '_' + indicator + '.pdf'
    plt.savefig(output_file)

def plot_evaluations_algorithm(algorithm, file, dirs, title, indicator='hv', threshold=36, plot_threshold=True, initial_eval=40):
    plt.figure(figsize=(10, 6))
    plt.xlabel('Evaluations')
    plt.ylabel('Hypervolume')
    for dir_ in dirs:
        data = pd.read_csv(file)
        hypervolumes = data[(data['dir'] == dir_) & (data['indicator'] == indicator) & (data['evaluations'] <= 1200) & (data['evaluations'] >= 40)][['evaluations', 'value']]
        hypervolumes.sort_values(by='evaluations', inplace=True)
        if hypervolumes.empty:
            print(f'No data found for directory: {dir_}, indicator: {indicator}')
            raise ValueError(f'No data found for directory: {dir_}, indicator: {indicator}')
        # the initial_eval is subtracted to align the x-axis with the evaluations starting from 0
        plt.plot(hypervolumes['evaluations'] - initial_eval, hypervolumes['value'], label=dir_)
    if plot_threshold:
        plt.axhline(y=threshold, color='gray', linestyle='--')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid()
    output_file = 'evaluations_' + algorithm + '_' + indicator + '.pdf'
    plt.savefig(output_file)

def get_median_algorithm(algorithm, file, dirs, indicator='hv'):

    dirs_vals = []
    for dir_ in dirs:
        data = pd.read_csv(file)
        hypervolumes = data[(data['dir'] == dir_) & (data['indicator'] == indicator) & (data['evaluations'] <= 1200) & (data['evaluations'] >= 40)][['evaluations', 'value']]
        hypervolumes.sort_values(by='evaluations', inplace=True)
        if hypervolumes.empty:
            print(f'No data found for directory: {dir_}, indicator: {indicator}')
            raise ValueError(f'No data found for directory: {dir_}, indicator: {indicator}')
        last_hyp = hypervolumes['value'].iloc[-1]
        dirs_vals.append((dir_, last_hyp))
    dirs_vals.sort(key=lambda x: x[1])
    median_dir = dirs_vals[len(dirs_vals) // 2][0]
    return median_dir



if __name__ == '__main__':

    dirs_r2_emoa_60_40 = [
        'results/r2-emoa/cifar10/2026-09-04_20-19-31_18906049/search/',
        'results/r2-emoa/cifar10/2026-09-05_00-54-01_15798821/search/',
        'results/r2-emoa/cifar10/2026-09-05_05-24-57_65381509/search/',
        'results/r2-emoa/cifar10/2026-09-05_09-46-19_27293207/search/',
        'results/r2-emoa/cifar10/2026-09-05_14-12-18_27522793/search/',
    ]

    dirs_r2_emoa_75_25 = [
        'results/r2-emoa/cifar10/2026-09-09_15-04-16_27522793/search/',
        'results/r2-emoa/cifar10/2026-09-09_10-32-49_27293207/search/',
        'results/r2-emoa/cifar10/2026-09-09_05-58-45_65381509/search/',
        'results/r2-emoa/cifar10/2026-09-08_04-07-38_15798821/search/',
        'results/r2-emoa/cifar10/2026-09-07_23-57-40_18906049/search/',
    ]

    dirs_r2_emoa_unif = [
        'results/r2-emoa/cifar10/2026-09-07_16-46-58_27522793/search/',
        'results/r2-emoa/cifar10/2026-09-07_12-26-06_27293207/search/',
        'results/r2-emoa/cifar10/2026-09-07_08-04-18_65381509/search/',
        'results/r2-emoa/cifar10/2026-09-07_03-40-31_15798821/search/',
        'results/r2-emoa/cifar10/2026-09-06_23-22-40_18906049/search/',
    ]

    dirs_sms_emoa = [
        'results/sms-emoa/cifar10/2026-09-05_12-55-21_27522793/search/',
        'results/sms-emoa/cifar10/2026-09-05_08-47-45_27293207/search/',
        'results/sms-emoa/cifar10/2026-09-04_20-03-01_18906049/search/',
        'results/sms-emoa/cifar10/2026-09-05_04-29-39_65381509/search/',
        'results/sms-emoa/cifar10/2026-09-05_00-21-30_15798821/search/',
    ]
    dirs_cars = [
        'results/cars/cifar10/2026-09-10_12-51-30_15798821/search/',
        'results/cars/cifar10/2026-09-10_17-06-01_65381509/search/',
        'results/cars/cifar10/2026-09-10_23-43-02_27293207/search/',
    ]
    dirs_moead = [
        'results/moead/cifar10/2026-09-05_23-26-35_18906049/search/',
        'results/moead/cifar10/2026-09-06_03-30-00_15798821/search/',
        'results/moead/cifar10/2026-09-06_07-31-47_65381509/search/',
        'results/moead/cifar10/2026-09-06_11-43-56_27293207/search/',
        'results/moead/cifar10/2026-09-06_15-53-52_27522793/search/'
    ]
    dirs_moras = [
        'results/moras/cifar10/2026-09-06_00-04-08_18906049/search/',
        'results/moras/cifar10/2026-09-06_04-28-39_15798821/search/',
        'results/moras/cifar10/2026-09-06_08-52-59_65381509/search/',
        'results/moras/cifar10/2026-09-06_13-16-25_27293207/search/',
        'results/moras/cifar10/2026-09-06_17-47-48_27522793/search/',
    ]
    dirs_nevonas = [
        'search-S1-20260907-233615-cifar10',
        'search-S1-20260908-094614-cifar10',
        'search-S1-20260908-193347-cifar10',
        'search-S1-20260909-005433-cifar10',
        'search-S1-20260909-153544-cifar10',
    ]
    dirs_nsganet = [
        'search-NSGA-Net-micro-20260906-232401',
        'search-NSGA-Net-micro-20260907-034602',
        'search-NSGA-Net-micro-20260907-082105',
        'search-NSGA-Net-micro-20260907-125237',
        'search-NSGA-Net-micro-20260907-171050',
    ]
    dirs_random = [
        'results/random-search/cifar10/2026-09-09_23-13-57_18906049/search/',
        'results/random-search/cifar10/2026-09-10_04-14-40_15798821/search/',
        'results/random-search/cifar10/2026-09-10_09-17-06_65381509/search/',
        'results/random-search/cifar10/2026-09-10_14-18-16_27293207/search/',
        'results/random-search/cifar10/2026-09-10_19-20-07_27522793/search/',
    ]
    median_sms_emoa = get_median_algorithm('sms-emoa', 'evaluations-sms-emoa.csv', dirs_sms_emoa, indicator='hv')
    median_cars = get_median_algorithm('cars', 'evaluations-cars.csv', dirs_cars, indicator='hv')
    median_moead = get_median_algorithm('moead', 'evaluations-moead.csv', dirs_moead, indicator='hv')
    median_moras = get_median_algorithm('moras', 'evaluations-moras.csv', dirs_moras, indicator='hv')
    median_nevonas = get_median_algorithm('nevonas', 'evaluations-nevonas.csv', dirs_nevonas, indicator='hv')
    median_nsganet = get_median_algorithm('nsganet', 'evaluations-nsganet.csv', dirs_nsganet, indicator='hv')
    median_random = get_median_algorithm('random', '../evaluations.csv', dirs_random, indicator='hv')
    median_r2_emoa_rnas_60_40 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_60_40, indicator='hv')
    median_r2_emoa_rnas_75_25 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_75_25, indicator='hv')
    median_r2_emoa_rnas_unif = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_unif, indicator='hv')
    plot_evaluations_algorithms('cifar10',
                                [(median_sms_emoa, 'SMS-EMOA', 'evaluations-sms-emoa.csv'),
                                 #(median_cars, 'CARS', 'evaluations-cars.csv'),
                                 (median_moead, 'MOEA/D', 'evaluations-moead.csv'),
                                 (median_moras, 'MORAS', 'evaluations-moras.csv'),
                                 #(median_nevonas, 'NevoNAS', 'evaluations-nevonas.csv'),
                                 (median_nsganet, 'NSGA-Net', 'evaluations-nsganet.csv'),
                                 (median_random, 'Random Search', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_60_40, 'R2-EMOA-RNAS$_{0.60}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_75_25, 'R2-EMOA-RNAS$_{0.75}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_unif, 'R2-EMOA-RNAS$_{Unif}$', '../evaluations.csv')], indicator='hv', threshold=182250, plot_threshold=False)

    median_sms_emoa = get_median_algorithm('sms-emoa', 'evaluations-sms-emoa.csv', dirs_sms_emoa, indicator='hv_2obj')
    median_cars = get_median_algorithm('cars', 'evaluations-cars.csv', dirs_cars, indicator='hv_2obj')
    median_moead = get_median_algorithm('moead', 'evaluations-moead.csv', dirs_moead, indicator='hv_2obj')
    median_moras = get_median_algorithm('moras', 'evaluations-moras.csv', dirs_moras, indicator='hv_2obj')
    median_nevonas = get_median_algorithm('nevonas', 'evaluations-nevonas.csv', dirs_nevonas, indicator='hv_2obj')
    median_nsganet = get_median_algorithm('nsganet', 'evaluations-nsganet.csv', dirs_nsganet, indicator='hv_2obj')
    median_random = get_median_algorithm('random', '../evaluations.csv', dirs_random, indicator='hv_2obj')
    median_r2_emoa_rnas_60_40 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_60_40, indicator='hv_2obj')
    median_r2_emoa_rnas_75_25 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_75_25, indicator='hv_2obj')
    median_r2_emoa_rnas_unif = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_unif, indicator='hv_2obj')
    plot_evaluations_algorithms('cifar10',
                                [(median_sms_emoa, 'SMS-EMOA', 'evaluations-sms-emoa.csv'),
                                 #(median_cars, 'CARS', 'evaluations-cars.csv'),
                                 (median_moead, 'MOEA/D', 'evaluations-moead.csv'),
                                 (median_moras, 'MORAS', 'evaluations-moras.csv'),
                                 #(median_nevonas, 'NevoNAS', 'evaluations-nevonas.csv'),
                                 (median_nsganet, 'NSGA-Net', 'evaluations-nsganet.csv'),
                                 (median_random, 'Random Search', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_60_40, 'R2-EMOA-RNAS$_{0.60}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_75_25, 'R2-EMOA-RNAS$_{0.75}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_unif, 'R2-EMOA-RNAS$_{Unif}$', '../evaluations.csv')], indicator='hv_2obj', threshold=182250, plot_threshold=False)
    dirs_sms_emoa_100 = [
        'results/sms-emoa/cifar100/2026-09-14_15-43-07_18906049/search/',
        'results/sms-emoa/cifar100/2026-09-14_20-56-00_15798821/search/',
        'results/sms-emoa/cifar100/2026-09-15_02-05-49_65381509/search/',
        'results/sms-emoa/cifar100/2026-09-15_08-10-46_27293207/search/',
        'results/sms-emoa/cifar100/2026-09-15_21-31-10_27522793/search/'
    ]
    dirs_moead_100 = [
        'results/moead/cifar100/2026-09-17_22-26-44_18906049/search/',
        'results/moead/cifar100/2026-09-18_03-42-22_15798821/search/',
        'results/moead/cifar100/2026-09-18_08-48-48_65381509/search/',
        'results/moead/cifar100/2026-09-18_14-20-48_27293207/search/',
        'results/moead/cifar100/2026-09-18_19-51-14_27522793/search/',
    ]
    dirs_moras_100 = [
        'results/moras/cifar100/2026-09-16_18-00-03_18906049/search/',
        'results/moras/cifar100/2026-09-16_23-43-16_15798821/search/',
        'results/moras/cifar100/2026-09-17_05-24-53_65381509/search/',
        'results/moras/cifar100/2026-09-17_10-56-53_27293207/search/',
        'results/moras/cifar100/2026-09-17_16-34-46_27522793/search/',
    ]
    dirs_nsganet_100 = [
        'search-NSGA-Net-micro-20260916-111315',
        'search-NSGA-Net-micro-20260916-213121',
        'search-NSGA-Net-micro-20260917-105458',
        'search-NSGA-Net-micro-20260918-002906',
        'search-NSGA-Net-micro-20260918-140331',
    ]
    dirs_random_100 = [
        'results/random-search/cifar100/2026-09-14_18-24-33_18906049/search/',
        'results/random-search/cifar100/2026-09-15_13-46-11_15798821/search/',
        'results/random-search/cifar100/2026-09-15_20-12-11_65381509/search/',
        'results/random-search/cifar100/2026-09-16_02-35-18_27293207/search/',
        'results/random-search/cifar100/2026-09-16_08-59-33_27522793/search/',
    ]
    dirs_r2_emoa_60_40_100 = [
        'results/r2-emoa/cifar100/2026-09-11_00-58-43_18906049/search/',
        'results/r2-emoa/cifar100/2026-09-11_06-22-44_15798821/search/',
        'results/r2-emoa/cifar100/2026-09-11_11-57-11_65381509/search/',
        'results/r2-emoa/cifar100/2026-09-11_17-36-00_27293207/search/',
        'results/r2-emoa/cifar100/2026-09-11_23-18-18_27522793/search/',
    ]
    dirs_r2_emoa_75_25_100 = [
        'results/r2-emoa/cifar100/2026-09-12_05-48-46_18906049/search/',
        'results/r2-emoa/cifar100/2026-09-12_11-30-14_15798821/search/',
        'results/r2-emoa/cifar100/2026-09-12_17-13-07_65381509/search/',
        'results/r2-emoa/cifar100/2026-09-12_22-44-03_27293207/search/',
        'results/r2-emoa/cifar100/2026-09-13_04-29-18_27522793/search/',
    ]
    dirs_r2_emoa_unif_100 = [
        'results/r2-emoa/cifar100/2026-09-14_10-13-09_27522793/search/',
        'results/r2-emoa/cifar100/2026-09-14_04-08-30_27293207/search/',
        'results/r2-emoa/cifar100/2026-09-13_22-31-39_65381509/search/',
        'results/r2-emoa/cifar100/2026-09-13_16-35-53_15798821/search/',
        'results/r2-emoa/cifar100/2026-09-13_10-40-39_18906049/search/',
    ]
    median_sms_emoa_100 = get_median_algorithm('sms-emoa', 'evaluations-sms-emoa.csv', dirs_sms_emoa_100, indicator='hv')
    median_moras_100 = get_median_algorithm('moras', 'evaluations-moras.csv', dirs_moras_100, indicator='hv')
    median_moead_100 = get_median_algorithm('moead', 'evaluations-moead.csv', dirs_moead_100, indicator='hv')
    median_nsganet_100 = get_median_algorithm('nsganet', 'evaluations-nsganet.csv', dirs_nsganet_100, indicator='hv')
    median_random_100 = get_median_algorithm('random-search', '../evaluations.csv', dirs_random_100, indicator='hv')
    median_r2_emoa_rnas_60_40_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_60_40_100, indicator='hv')
    median_r2_emoa_rnas_75_25_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_75_25_100, indicator='hv')
    median_r2_emoa_rnas_unif_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_unif_100, indicator='hv')
    plot_evaluations_algorithms('cifar100',
                                [(median_sms_emoa_100, 'SMS-EMOA', 'evaluations-sms-emoa.csv'),
                                 (median_moras_100, 'MORAS', 'evaluations-moras.csv'),
                                 (median_moead_100, 'MOEA/D', 'evaluations-moead.csv'),
                                 (median_nsganet_100, 'NSGA-Net', 'evaluations-nsganet.csv'),
                                 (median_random_100, 'Random Search', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_60_40_100, 'R2-EMOA-RNAS$_{0.60}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_75_25_100, 'R2-EMOA-RNAS$_{0.75}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_unif_100, 'R2-EMOA-RNAS$_{Unif}$', '../evaluations.csv')], indicator='hv', threshold=182250, plot_threshold=False)

    median_sms_emoa_100 = get_median_algorithm('sms-emoa', 'evaluations-sms-emoa.csv', dirs_sms_emoa_100, indicator='hv_2obj')
    median_moras_100 = get_median_algorithm('moras', 'evaluations-moras.csv', dirs_moras_100, indicator='hv_2obj')
    median_moead_100 = get_median_algorithm('moead', 'evaluations-moead.csv', dirs_moead_100, indicator='hv_2obj')
    median_nsganet_100 = get_median_algorithm('nsganet', 'evaluations-nsganet.csv', dirs_nsganet_100, indicator='hv_2obj')
    median_random_100 = get_median_algorithm('random-search', '../evaluations.csv', dirs_random_100, indicator='hv_2obj')
    median_r2_emoa_rnas_60_40_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_60_40_100, indicator='hv_2obj')
    median_r2_emoa_rnas_75_25_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_75_25_100, indicator='hv_2obj')
    median_r2_emoa_rnas_unif_100 = get_median_algorithm('r2-emoa', '../evaluations.csv', dirs_r2_emoa_unif_100, indicator='hv_2obj')
    plot_evaluations_algorithms('cifar100',
                                [(median_sms_emoa_100, 'SMS-EMOA', 'evaluations-sms-emoa.csv'),
                                 (median_moras_100, 'MORAS', 'evaluations-moras.csv'),
                                 (median_moead_100, 'MOEA/D', 'evaluations-moead.csv'),
                                 (median_nsganet_100, 'NSGA-Net', 'evaluations-nsganet.csv'),
                                 (median_random_100, 'Random Search', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_60_40_100, 'R2-EMOA-RNAS$_{0.60}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_75_25_100, 'R2-EMOA-RNAS$_{0.75}$', '../evaluations.csv'),
                                 (median_r2_emoa_rnas_unif_100, 'R2-EMOA-RNAS$_{Unif}$', '../evaluations.csv')], indicator='hv_2obj', threshold=182250, plot_threshold=False)