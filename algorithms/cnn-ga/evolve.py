import os
import time
from datetime import datetime

from archivers import archive_update_pq
from utils_search import store_metrics, save_archive, save_archive_losses, plot_archive_losses, plot_hypervolume, \
    plot_r2, plot_hypervolume2, save_statistics_to_csv, save_params, save_architecture, get_weights_r2
from utils import StatusUpdateTool, Utils, Log
from genetic.population import Population
from genetic.evaluate import FitnessEvaluate, store_model_script
from genetic.crossover_and_mutation import CrossoverAndMutation
from genetic.selection_operator import Selection
import torch
import numpy as np
import copy

class EvolveCNN(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        # non-dominated solutions (4 objs)
        self.archive = []
        # non-dominated solutions (2 objs)
        self.archive_2 = []
        self.statistics = {'hyp_log': [], 'hyp2_log': [], 'r2_log': []}
        self.weights_r2 = get_weights_r2(params['pop_size'])

    def initialize_population(self):
        #StatusUpdateTool.begin_evolution()
        pops = Population(params, 0)
        pops.initialize()
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def fitness_evaluate(self):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        fitness.evaluate()


    def crossover_and_mutation(self):
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log, self.pops.individuals, _params={'gen_no': self.pops.gen_no})
        offspring = cm.process()
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        v_list = []
        indi_list = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.scalar_fitness())
        for indi in self.parent_pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.scalar_fitness())

        _str = []
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'Indi-%s-%s-%s'%(indi.id, np.array_str(indi.F), indi.uuid()[0])
            _str.append(_t_str)
        for _, indi in enumerate(self.parent_pops.individuals):
            _t_str = 'Pare-%s-%s-%s'%(indi.id, np.array_str(indi.F), indi.uuid()[0])
            _str.append(_t_str)


        #add log
        # find the largest one's index
        max_index = np.argmax(v_list)
        selection = Selection()
        selected_index_list = selection.RouletteSelection(v_list, k=self.params['pop_size'])
        if max_index not in selected_index_list:
            first_selectd_v_list = [v_list[i] for i in selected_index_list]
            min_idx = np.argmin(first_selectd_v_list)
            selected_index_list[min_idx] = max_index

        next_individuals = [indi_list[i] for i in selected_index_list]

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no+1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%s-%s'%(indi.id, np.array_str(indi.F), indi.uuid()[0])
            _str.append(_t_str)
        _file = './populations/ENVI_%2d.txt'%(self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def do_work(self, max_gen):
        Log.info('*'*25)
        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation'%(gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(gen_no))
        self.fitness_evaluate()
        Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(gen_no))
        gen_no += 1
        evaluated_solutions = 0
        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            #step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation'%(curr_gen))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation'%(curr_gen))

            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness'%(curr_gen))
            self.fitness_evaluate()
            Log.info('EVOLVE[%d-gen]-Finish the evaluation'%(curr_gen))

            # store the non-dominated solutions
            # guarda los individuos
            self.archive = archive_update_pq(self.archive, self.pops.individuals)
            # store the non-dominated solutions (2 objs)
            self.archive_2 = archive_update_pq(self.archive_2, self.pops.individuals, k=2)
            evaluated_solutions += len(self.pops.individuals)
            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection'%(curr_gen))
            hyp, hyp_2, r2 = store_metrics(evaluated_solutions, self.archive, self.archive_2, self.params, self.weights_r2, self.statistics)
            print('>>>>>>> Gen {}: hyp={}, hyp_2={}, r2={}'.format(curr_gen, hyp, hyp_2, r2))
            plot_hypervolume(self.statistics, self.params['save_dir'])
            plot_hypervolume2(self.statistics, self.params['save_dir'])
            plot_r2(self.statistics, self.params['save_dir'])

        dir_arch = self.params['save_dir'] + 'architectures' + os.sep + 'scripts'
        if not os.path.exists(dir_arch):
            os.makedirs(dir_arch)
        for i, ind in enumerate(self.archive):
            # Store the architecture as a pytorch file
            file_name = Utils.generate_pytorch_file(ind, dir_arch)
            # Run the generated script for storing the model
            store_model_script(file_name, i, self.params['save_dir'], dir_arch, ind.F)

        save_archive(self.archive, self.params['save_dir'])
        save_archive_losses(self.archive_2, self.params['save_dir'])
        plot_archive_losses(self.archive_2, self.params['save_dir'])
        plot_hypervolume(self.statistics, self.params['save_dir'])
        plot_hypervolume2(self.statistics, self.params['save_dir'])
        plot_r2(self.statistics, self.params['save_dir'])
        save_statistics_to_csv(self.statistics, self.params['save_dir'])
        save_params(self.params, self.params['save_dir'])

        StatusUpdateTool.end_evolution()
if __name__ == '__main__':
    params = StatusUpdateTool.get_init_params()
    params['save_dir'] = 'search_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")) + os.sep
    if not os.path.exists(params['save_dir']):
        os.mkdir(params['save_dir'])
    if torch.cuda.is_available():
        params['device'] = 'cuda'
    elif torch.backends.mps.is_available():
        params['device'] = 'mps'
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    start = time.time()
    evoCNN = EvolveCNN(params)
    evoCNN.do_work(max_gen=30)
    print('>>>>>> Results stored in {}'.format(params['save_dir']))
    print('Total time: {:.2f} HOURS'.format((time.time() - start)/3600))
