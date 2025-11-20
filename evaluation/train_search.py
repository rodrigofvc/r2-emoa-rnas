import time
import logging

import torch
from torch import nn
from torch.cuda.amp import autocast
from evaluation.model_search import discretize


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

def check_input(input):
    assert input.dtype == torch.float32 and input.is_contiguous(), (input.dtype, input.stride())
    assert input.dim() == 4 and input.shape[1] in (1, 3, 16, 24, 32, 64), input.shape
    assert torch.isfinite(input).all(), "input contiene NaN/Inf"

def infer(valid_queue, model, criterion, attack, args):
    std_correct = 0
    adv_correct = 0
    std_loss_mean = 0
    adv_loss_mean = 0
    total_loss_mean = 0
    total = 0
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        input  = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        adv_input = attack(input, target)
        adv_input = adv_input.to(args.device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                std_logits = model(input)
                std_loss = criterion(std_logits, target)
                adv_logits = model(adv_input)
                adv_loss = criterion(adv_logits, target)
                total_loss = args.lambda_1 * std_loss + args.lambda_2 * adv_loss

        std_predicts = std_logits.argmax(dim=1)
        adv_predicts = adv_logits.argmax(dim=1)
        std_correct += (std_predicts == target).sum().item()
        adv_correct += (adv_predicts == target).sum().item()
        total += target.size(0)
        std_loss_mean += std_loss.item()
        adv_loss_mean += adv_loss.item()
        total_loss_mean += total_loss.item()
    std_accuracy = std_correct / total
    adv_accuracy = adv_correct / total
    std_loss_mean /= total
    adv_loss_mean /= total
    total_loss_mean /= total
    return std_accuracy * 100.0, adv_accuracy * 100.0, std_loss_mean, adv_loss_mean, total_loss_mean

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

