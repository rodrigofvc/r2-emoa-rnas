import time
import logging

import torch
import torchattacks
from torch import nn

import utils
from model_search import Network, discretize


def get_attack_function(attack_params):
    if attack_params['name'] == 'FGSM':
        attack_function = lambda model: torchattacks.FGSM(model, **attack_params['params'])
    elif attack_params['name'] == 'PGD':
        attack_function = lambda model: torchattacks.PGD(model, **attack_params['params'])
    else:
        raise ValueError(f"Attack {attack_params['name']} not defined")
    return attack_function

# Train a model for one epoch
def train(train_queue, model, lambda_1, lambda_2, criterion, optimizer, args, attack_f, device):
  std_correct = 0
  adv_correct = 0
  total = 0
  model.train()
  for step, (input, target) in enumerate(train_queue):
      timesstamp = time.time()
      n = input.size(0)
      prepare_time = time.time()
      input = input.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)


      optimizer.zero_grad()
      print('data prep DONE in %.3f seconds' % (time.time() - prepare_time))
      weights_time = time.time()

      time_stamp = time.time()
      attack = attack_f(model)
      adv_X = attack(input, target)
      logits_adv = model(adv_X)
      adv_loss = criterion(logits_adv, target)
      print('--- adv loss DONE in %.3f seconds' % (time.time() - time_stamp))

      time_stamp = time.time()
      logits = model(input)
      natural_loss = criterion(logits, target)
      print('--- std loss DONE in %.3f seconds' % (time.time() - time_stamp))

      total_loss = lambda_1 * natural_loss + lambda_2 * adv_loss

      total_loss.backward()
      nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
      optimizer.step()
      print('weights step DONE in %.3f seconds' % (time.time() - weights_time))

      std_predicts = logits.argmax(dim=1)
      adv_predicts = logits_adv.argmax(dim=1)
      std_correct += (std_predicts == target).sum().item()
      adv_correct += (adv_predicts == target).sum().item()
      total += target.size(0)
      std_accuracy = std_correct / total
      adv_accuracy = adv_correct / total

      # if step % args.report_freq == 0:
      print ('-- train step %d/%d loss_ws %.5f std_acc %.3f adv_acc %.3f %f segs' % (step+1, len(train_queue), total_loss.item(), std_accuracy, adv_accuracy, time.time() - timesstamp))

  return std_accuracy * 100.0, adv_accuracy * 100.0, total_loss.item()

def train_batch(input, target, model, lambda_1, lambda_2, criterion, optimizer, args, attack_f, device):
    prepare_time = time.time()
    input = input.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad()
    print('data prep DONE in %.3f seconds' % (time.time() - prepare_time))
    weights_time = time.time()

    time_stamp = time.time()
    attack = attack_f(model)
    adv_X = attack(input, target)
    logits_adv = model(adv_X)
    adv_loss = criterion(logits_adv, target)
    print('--- adv loss DONE in %.3f seconds' % (time.time() - time_stamp))

    time_stamp = time.time()
    logits = model(input)
    natural_loss = criterion(logits, target)
    print('--- std loss DONE in %.3f seconds' % (time.time() - time_stamp))

    total_loss = lambda_1 * natural_loss + lambda_2 * adv_loss

    total_loss.backward()
    nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
    optimizer.step()
    print('weights step DONE in %.3f seconds' % (time.time() - weights_time))

    std_predicts = logits.argmax(dim=1)
    adv_predicts = logits_adv.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()



def infer(valid_queue, model, lambda_1, lambda_2, criterion, attack_f, device):
    std_correct = 0
    adv_correct = 0
    total = 0
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input  = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        attack = attack_f(model)
        adv_input = attack(input, target).to(device, non_blocking=True)

        with torch.no_grad():
            std_logits = model(input)
            std_loss = criterion(std_logits, target)
            adv_logits = model(adv_input)
            adv_loss = criterion(adv_logits, target)
            total_loss = lambda_1 * std_loss + lambda_2 * adv_loss

        std_predicts = std_logits.argmax(dim=1)
        adv_predicts = adv_logits.argmax(dim=1)
        std_correct += (std_predicts == target).sum().item()
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)
        std_accuracy = std_correct / total
        adv_accuracy = adv_correct / total
        print('infer step %d loss_ws %.5f std_acc %.3f adv_acc %.3f' % (step, total_loss.item(), std_accuracy, adv_accuracy))
    return std_accuracy * 100.0, adv_accuracy * 100.0, total_loss.item()

def setup_logger(debug_mode):
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level
    )

# Normal epoch run function
def run_epoch(epoch, model, individuals, n_population, train_queue, valid_queue, criterion, optimizer, attack_f, scheduler, args, device):
    print(f">>>> Epoch {epoch+1}/{args.epochs}")

    individual = individuals[epoch % n_population]
    model.update_arch_parameters(individual)
    discrete = discretize(individual, model.genotype(), device)
    model.update_arch_parameters(discrete)

    time_train = time.time()
    # training
    std_accuracy, adv_accuracy, loss = train(train_queue, model, args.lambda_1, args.lambda_2, criterion, optimizer, attack_f=attack_f, device=device)
    print('train_acc %f, adv_acc %f, loss %f ', std_accuracy, adv_accuracy, loss)
    print(f"Tiempo de entrenamiento epoca {epoch+1}/{args.epochs}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_train))}")

    time_valid = time.time()
    # validation
    std_accuracy, adv_accuracy, loss = infer(valid_queue, model, args.lambda_1, args.lambda_2, criterion, attack_f=attack_f, device=device)
    print(f"Tiempo de validacion epoca {epoch+1}/{args.epochs}: {time.strftime('%H:%M:%S', time.gmtime(time.time() - time_valid))}")
    scheduler.step()
    #utils.save(model, os.path.join(args.save, 'weights.pt'))

def run_batch_epoch(model, individual, lambda_1, lambda_2, input, target, criterion, optimizer, attack_f, args, device):

    model.update_arch_parameters(individual)
    discrete = discretize(individual, model.genotype(), device)
    model.update_arch_parameters(discrete)

    prepare_time = time.time()
    input = input.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    optimizer.zero_grad()
    print('data prep DONE in %.3f seconds' % (time.time() - prepare_time))
    weights_time = time.time()

    time_stamp = time.time()
    attack = attack_f(model)
    adv_X = attack(input, target)
    logits_adv = model(adv_X)
    adv_loss = criterion(logits_adv, target)
    print('--- adv loss DONE in %.3f seconds' % (time.time() - time_stamp))

    time_stamp = time.time()
    logits = model(input)
    natural_loss = criterion(logits, target)
    print('--- std loss DONE in %.3f seconds' % (time.time() - time_stamp))

    total_loss = lambda_1 * natural_loss + lambda_2 * adv_loss

    total_loss.backward()
    nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)
    optimizer.step()
    print('weights step DONE in %.3f seconds' % (time.time() - weights_time))

    std_predicts = logits.argmax(dim=1)
    adv_predicts = logits_adv.argmax(dim=1)
    std_correct = (std_predicts == target).sum().item()
    adv_correct = (adv_predicts == target).sum().item()
    return std_correct, adv_correct, total_loss.item()
