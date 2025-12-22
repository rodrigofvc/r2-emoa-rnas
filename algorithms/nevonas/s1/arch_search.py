import os
import sys

from adversarial import get_attack_function
from archivers import archive_update_pq
from utils_search import save_archive, save_archive_2, plot_archive_losses, plot_hypervolume, plot_hypervolume2, \
  plot_r2, save_statistics_to_csv, save_params, save_architecture, save_supernet, get_model_metrics, store_metrics, \
  get_weights_r2

sys.path.insert(0, './s1')

import argparse
import logging
import random
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils
import time
import ut as utils

from get_datasets                          import get_dataloader
from model import NetworkCIFAR
from model_search                          import Network
from pymoo.core.problem                    import ElementwiseProblem
from pymoo.algorithms.moo.nsga2            import NSGA2
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.sbx         import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm           import PolynomialMutation

parser = argparse.ArgumentParser("S1")
parser.add_argument('--autoaug',           default=False, action='store_true')
parser.add_argument('--batch_size',        type=int, default = 96, help = 'batch size')
parser.add_argument('--config_path',       type=str, help='The config path.')
parser.add_argument('--config_root',       type=str, help='The root path of the config directory')
parser.add_argument('--cutout',            action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length',     type = int, default = 16, help = 'cutout length')
parser.add_argument('--data_dir',          type = str, default = '../../data', help = 'location of the data corpus')
parser.add_argument('--dataset',           type = str, default = 'cifar10', help = '["cifar10", "cifar100"]')
parser.add_argument('--epochs',            type = int, default = 30, help = 'num of generations')
parser.add_argument('--gpu',               type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip',         type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels',     type = int, default = 16, help = 'num of init channels')
parser.add_argument('--knn',               type = int, default = 5, help = 'k-nearest neighbors')
parser.add_argument('--layers',            type = int, default = 5, help = 'total number of layers')
parser.add_argument('--lambda_1',         type = float, default = 0.5, help = 'weight for std loss')
parser.add_argument('--lambda_2',         type = float, default = 0.5, help = 'weight for adv loss')
parser.add_argument('--steps',             type = int, default = 6, help = 'number of steps in one cell')
parser.add_argument('--multiplier',        type = int, default = 6, help = 'multiplier for number of channels')
parser.add_argument('--learning_rate',     type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum',          type = float, default = 0.9, help = 'momentum')
parser.add_argument('--mutate_rate',       type = float, default = 0.1, help = 'mutation rate')
parser.add_argument('--output_dir',        type = str, default = None, help = 'location of trials')
parser.add_argument('--pop_size',          type = int, default = 40, help = 'population size')
parser.add_argument('--report_freq',       type = float, default = 50, help = 'report frequency')
parser.add_argument('--seed',              type = int, default = 1, help = 'random seed')
parser.add_argument('--split_option',      type = int, default = 0.5, help = 'split option for CIFAR100')
parser.add_argument('--train_discrete',    default=False, action='store_true')
parser.add_argument('--train_epochs',      type = int, default = 0, help = 'num of training epochs')
parser.add_argument('--valid_batch_size',  type = int, default = 64, help = 'validation batch size')
parser.add_argument('--weight_decay',      type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--train_portion',     type = float, default = 0.5, help = 'portion of training data')
parser.add_argument('--workers',           type=int, default=0, help='number of data loading workers (default: 2)')
args = parser.parse_args([])

def train(model, train_queue, criterion, optimizer, gen, attack_f, device, pop=None):
  model.train()
  attack = attack_f(model)
  if pop is None:
    logging.info(f'In warm-up training')
    std_correct = 0
    adv_correct = 0
    total_inputs = 0
    for step, (inputs, targets) in enumerate(train_queue):
      # Sample a random architecture
      rnd = model.random_alphas(discrete=False)
      assert model.check_alphas(rnd), "Given alphas has not been copied successfully to the model"
      if args.train_discrete:
        discrete_alphas = utils.discretize(alphas=rnd, arch_genotype=model.genotype())
        model.update_alphas(discrete_alphas)
        assert model.check_alphas(discrete_alphas), "Given alphas has not been copied successfully to the model"

      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()

      adv_input, std_logits = attack(inputs, targets)
      adv_input = adv_input.to(device)
      adv_logits = model(adv_input)
      adv_loss = criterion(adv_logits, targets)
      std_loss = criterion(std_logits, targets)
      total_loss = args.lambda_1 * std_loss + args.lambda_2 * adv_loss
      total_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      std_predicts = std_logits.argmax(dim=1)
      adv_predicts = adv_logits.argmax(dim=1)
      std_correct += (std_predicts == targets).sum().item()
      adv_correct += (adv_predicts == targets).sum().item()
      total_inputs += targets.size(0)

    logging.info(f"Training supernet gen {gen} with loss: {total_loss.item():.5f}, std_acc: {std_correct/total_inputs*100:.5f}%, adv_acc: {adv_correct/total_inputs*100:.5f}%")

  else:
    std_correct = 0
    adv_correct = 0
    total_inputs = 0
    for step, (inputs, targets) in enumerate(train_queue):
      #Copying and checking the discretized alphas
      tx = torch.tensor(pop[step % len(pop)].X).type(torch.float).to(device)
      tx = list(torch.chunk(tx, 2))
      tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]    
      model.update_alphas(tx)
      assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
      # Discretizing the architecture
      discrete_alphas = utils.discretize(alphas=tx, arch_genotype=model.genotype(), device=device)
      model.update_alphas(discrete_alphas)
      assert model.check_alphas(discrete_alphas)
      #logging.info(f'step % len(pop): {step % len(pop)}')


      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()

      adv_input, std_logits = attack(inputs, targets)
      adv_input = adv_input.to(device)
      adv_logits = model(adv_input)
      adv_loss = criterion(adv_logits, targets)
      std_loss = criterion(std_logits, targets)
      total_loss = args.lambda_1 * std_loss + args.lambda_2 * adv_loss
      total_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      std_predicts = std_logits.argmax(dim=1)
      adv_predicts = adv_logits.argmax(dim=1)
      std_correct += (std_predicts == targets).sum().item()
      adv_correct += (adv_predicts == targets).sum().item()
      total_inputs += targets.size(0)

    logging.info(f"Training supernet gen {gen} with loss: {total_loss.item():.5f}, std_acc: {std_correct/total_inputs*100:.5f}%, adv_acc: {adv_correct/total_inputs*100:.5f}%")

class NAS(ElementwiseProblem):
  def __init__(self, n_var, n_obj, xl, xu):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
    self.archive = []
    self.archive_2 = []
    self.statistics = {'hyp_log': [], 'hyp2_log': [], 'r2_log': []}

  def validation(self, ind, model, valid_queue, criterion, gen, ind_idx, pop_size, attack_f, device):
    valid_start = time.time()
    
    tx = torch.tensor(ind.X).type(torch.float).to(device)
    tx = list(torch.chunk(tx, 2))
    tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]    
    #Copying and checking the discretized alphas
    model.update_alphas(tx)
    assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
    g1 = model.genotype()
    # Discretizing the architecture
    discrete_alphas = utils.discretize(alphas=tx, arch_genotype=g1, device=device)
    model.update_alphas(discrete_alphas)
    assert model.check_alphas(discrete_alphas)
    assert utils.compare_genotypes(arch1=model.genotype(), arch2=g1), 'Something wrong with discretization'
    if not ('genotype' in ind.data): ind.set('genotype', g1)
    
    model.eval()
    attack = attack_f(model)

    std_correct = 0
    adv_correct = 0
    std_loss_mean = 0
    adv_loss_mean = 0
    total_loss_mean = 0
    total = 0

    for step, (inputs, targets) in enumerate(valid_queue):
      inputs = inputs.to(device)
      targets = targets.to(device)

      adv_input, std_logits = attack(inputs, targets)
      adv_input = adv_input.to(device)

      with torch.no_grad():
        adv_logits = model(adv_input)
        adv_loss = criterion(adv_logits, targets)
        std_loss = criterion(std_logits, targets)
        total_loss = args.lambda_1 * std_loss + args.lambda_2 * adv_loss

      std_predicts = std_logits.argmax(dim=1)
      adv_predicts = adv_logits.argmax(dim=1)
      std_correct += (std_predicts == targets).sum().item()
      adv_correct += (adv_predicts == targets).sum().item()
      total += targets.size(0)
      std_loss_mean += std_loss.item()
      adv_loss_mean += adv_loss.item()
      total_loss_mean += total_loss.item()

    std_accuracy = std_correct / total
    adv_accuracy = adv_correct / total
    std_loss_mean /= total
    adv_loss_mean /= total
    total_loss_mean /= total

    logging.info(f"[{gen} Generation] {ind_idx}/{pop_size} finished with std_acc {std_accuracy * 100.0:.5f}, adv_acc {adv_accuracy * 100.0:.5f}, std_loss: {std_loss_mean:.5f}, adv_loss: {adv_loss_mean:.5f}")
    logging.info(f"Validation finished in {time.time() - valid_start} seconds")
    return std_loss_mean, adv_loss_mean, g1

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
DIR = "search-S1-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
args.save_dir = DIR
utils.create_exp_dir(DIR)
#utils.create_exp_dir(os.path.join(DIR, "output_genotypes"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
  device = torch.device("cuda:{}".format(args.gpu))
  torch.cuda.set_device(args.gpu)
  cpu_device = torch.device("cpu")
elif torch.backends.mps.is_available():
  # test
  device = torch.device("mps")
else:
  device = torch.device("cpu")

cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

#logging.info(f'python {" ".join([ar for ar in sys.argv])}')
#logging.info(f'torch version: {torch.__version__}, torchvision version: {torch.__version__}')
#logging.info("gpu device = {}".format(args.gpu))
#logging.info("args =  %s", args)
#logging.info("[INFO] First Train and then evolve and repeat the cycle")

# Configuring dataset and dataloader
train_transform, valid_transform, train_queue, valid_queue = get_dataloader(args)
#logging.info(f'train_transform: {train_transform}, \nvalid_transform: {valid_transform}')
if args.dataset == 'cifar10':    num_classes = 10
elif args.dataset == 'cifar100': num_classes = 100
#logging.info("#classes: {}".format(num_classes))
#logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue)*args.batch_size, len(valid_queue)*args.valid_batch_size))

# Model Initialization
model = Network(args.init_channels, num_classes, args.layers, device, args.steps, args.multiplier)
model = model.to(device)

# Configuring the optimizer and the scheduler
optimizer = torch.optim.SGD(model.parameters(),
                            args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min = args.learning_rate_min)
#lr = scheduler.get_lr()[0]

r2_weights = get_weights_r2(args.pop_size)

# Initializing the problem
n_params = model.arch_parameters()[0].view(-1).shape[0] * 2

nas = NAS(n_var=n_params,
          n_obj=4,
          xl=np.zeros(n_params),
          xu=np.ones(n_params)
         )

# create the algorithm object
algorithm = NSGA2(pop_size=args.pop_size,
                  crossover=SimulatedBinaryCrossover(eta=15, prob=0.7),
                  mutation=PolynomialMutation(prob=args.mutate_rate, eta=20)
                  )

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

# create an algorithm object that never terminates
algorithm.setup(problem=nas, termination=termination, seed=args.seed, save_history=True)

attack_params = {
  'name': 'FGSM',
  'params': {
    'eps': '8/255',
  }
}

attack_f = get_attack_function(attack_params)

# STAGE 1
start = time.time()
if args.train_epochs > 0: logging.info('[INFO] Training the Supernet (Warmup)')
for train_epoch in range(args.train_epochs):
  train_time = time.time()
  logging.info("[INFO] Epoch {} with learning rate {}".format(train_epoch + 1, scheduler.get_lr()[0]))
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=train_epoch+1, attack_f=attack_f, device=device)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - train_time) / 60))
  scheduler.step()

architectures_evaluated = 0
for n_gen in range(args.epochs):
  start_time = time.time()
  # ask the algorithm for the next solution to be evaluated
  pop = algorithm.ask()

  ## Training using the whole population
  #logging.info("[INFO] Generation {} training with learning rate {}".format(n_gen + 1, scheduler.get_lr()[0]))
  #def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=n_gen+1, device=device, attack_f=attack_f, pop=pop)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
  scheduler.step()
  
  # Evaluating the individuals in the population
  logging.info("[INFO] Evaluating Generation {} ".format(n_gen + 1))
  std_loss_pop, adv_loss_pop, flops_pop, params_pop = [], [], [], []
  for ind_idx, ind in enumerate(pop):
    std_loss, adv_loss, arch_genotype = nas.validation(ind=ind,
                                                  model=model,
                                                  valid_queue=valid_queue,
                                                  criterion=criterion,
                                                  gen=n_gen+1,
                                                  ind_idx=ind_idx+1,
                                                  pop_size=len(pop),
                                                  attack_f=attack_f,
                                                  device=device)

    model_flops, model_parameters = get_model_metrics(genotype=arch_genotype, model=model)
    std_loss_pop.append(std_loss)
    adv_loss_pop.append(adv_loss)
    flops_pop.append(model_flops)
    params_pop.append(model_parameters)
    ind.set('genotype', arch_genotype)
    ind.set('F_norm', np.zeros(4))

  pop.set("F", np.column_stack([std_loss_pop, adv_loss_pop, flops_pop, params_pop]))
  #pop.set("F_norm", np.zeros(4))
  architectures_evaluated += len(pop)
  for ind in pop:
    print(f'Ind Fitness: {ind.F}')
  nas.archive = archive_update_pq(nas.archive, pop)
  nas.archive_2 = archive_update_pq(nas.archive_2, pop, k=2)
  hyp, hyp2, r2 = store_metrics(architectures_evaluated, nas.archive, nas.archive_2, args, r2_weights, nas.statistics)
  print(f'>>>>>>> Generation {n_gen + 1}')
  print(f'        hyp: {hyp}, hyp_2: {hyp2}, R2: {r2}')

  # this line is necessary to set the CV and feasbility status - even for unconstrained
  #set_cv(pop)
  
  # returned the evaluated individuals which have been evaluated or even modified
  algorithm.tell(infills=pop)
  logging.info(f'Algorithm generation #{algorithm.n_gen} completed')
  # print evaluations so far
  logging.info(f'Architectures evaluated so far: {architectures_evaluated}')
  
  # do same more things, printing, logging, storing or even modifying the algorithm object
    
  last = time.time() - start_time
  logging.info("[INFO] {}/{} generation finished in {} minutes".format(n_gen + 1, args.epochs, last / 60))

# obtain the result objective from the algorithm
res = algorithm.result()

# save supernet
save_supernet(model, DIR)

for i, ind in enumerate(nas.archive):
    logging.info(f'Archive individual fitness: {ind.F}')
    # check architecture
    model = NetworkCIFAR(args.init_channels, 10, args.layers, auxiliary=False, genotype=ind.get('genotype'))
    save_architecture(i, ind, DIR)

save_archive(nas.archive, DIR)
save_archive_2(nas.archive_2, DIR)
plot_archive_losses(nas.archive_2, DIR)
plot_hypervolume(nas.statistics, DIR)
plot_hypervolume2(nas.statistics, DIR)
plot_r2(nas.statistics, DIR)
save_statistics_to_csv(nas.statistics, DIR)
save_params(args, DIR)


