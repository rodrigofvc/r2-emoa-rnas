import time

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



class Architect(object):

  def __init__(self, model, args, lambda_1, lambda_2, criterion, attack_f, device):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.lambda_1 = lambda_1
    self.lambda_2 = lambda_2
    self.criterion = criterion
    self.attack_f = attack_f
    self.device = device

  def _concat(self, xs):
    return torch.cat([x.view(-1) for x in xs])

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    with torch.no_grad():
      theta = self._concat(self.model.parameters())

    timestamp = time.time()
    attack = self.attack_f(self.model)
    adv_input = attack(input, target)
    adv_loss = self.model._loss(adv_input, target)
    natural_loss = self.model._loss(input, target)
    total_loss = self.lambda_1 * natural_loss + self.lambda_2 * adv_loss
    print("-------get loss: {:.3f}".format(time.time() - timestamp))

    timestamp = time.time()
    try:
      moment = self._concat(
        network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()
      ).mul_(self.network_momentum)
    except Exception:
      moment = torch.zeros_like(theta)
    print("-------get momentum: {:.3f}".format(time.time() - timestamp))

    # \nabla_w (0.5 * L^{adv}_train (w, alpha) + 0.5 * L^{std}_train (w, alpha))
    dtheta_w = self._concat(torch.autograd.grad(total_loss, self.model.parameters()))
    # eta * (moment + dw + weight_decay * w)
    dtheta = eta * (moment + dtheta_w + self.network_weight_decay*theta)
    # w' = w - eta*(moment + dw + weight_decay * w)
    time_stamp = time.time()
    unrolled_model = self._construct_model_from_theta(theta.sub(dtheta))
    print("-------construct_model_from_theta: {:.3f}".format(time.time() - time_stamp))
    return unrolled_model


  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        # second-order
        time_stamp = time.time()
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        print("-------_backward_step_unrolled: {:.3f}".format(time.time() - time_stamp))
    else:
        # first-order
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    attack = self.attack_f(self.model)
    adv_input = attack(input_valid, target_valid)
    adv_loss = self.model._loss(adv_input, target_valid)
    natural_loss = self.model._loss(input_valid, target_valid)
    # L_val(w, alpha) = self.lambda_1 * L^{adv}_val(w, alpha) + self.lambda_2 * L^{std}_val(w, alpha)
    loss = self.lambda_1 * natural_loss + self.lambda_2 * adv_loss
    # \nabla_alpha L_val(w, alpha)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    time_stamp = time.time()
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    print("-------compute_unrolled_model: {:.3f}".format(time.time() - time_stamp))

    attack = self.attack_f(unrolled_model)
    adv_val_input = attack(input_valid, target_valid)
    adv_val_loss = unrolled_model._loss(adv_val_input, target_valid)
    natural_val_loss = unrolled_model._loss(input_valid, target_valid)
    total_val_loss = self.lambda_1 * natural_val_loss + self.lambda_2 * adv_val_loss

    time_stamp = time.time()
    # First order terms: \nabla_alpha (0.5 * L^{adv}_val(w', alpha) + 0.5 * L^{std}_val(w', alpha))
    grad_first_order_terms = torch.autograd.grad(total_val_loss, unrolled_model.arch_parameters(), retain_graph=True)
    print("-------first order terms: {:.3f}".format(time.time() - time_stamp))

    # Second-order (1st term) \nabla_alpha L^{std}_val(w', alpha) \nabla^2_{alpha,w} L^{adv}_train(w', alpha)
    grad_w_std_loss = torch.autograd.grad(natural_val_loss, unrolled_model.parameters(), retain_graph=True)
    vector = [v.detach() for v in grad_w_std_loss]
    time_stamp = time.time()
    implicit_grads_first_term = self._hessian_vector_product_adv(vector, std_loss=False, input=input_train, target=target_train)
    print("-------hessian_vector_product_adv 1: {:.3f}".format(time.time() - time_stamp))

    # Second-order (2nd term) \nabla_alpha L^{adv}_val(w', alpha) \nabla^2_{alpha,w} L^{std}_train(w', alpha)
    grad_w_adv_loss = torch.autograd.grad(adv_val_loss, unrolled_model.parameters(), retain_graph=False)
    vector = [v.detach() for v in grad_w_adv_loss]
    time_stamp = time.time()
    implicit_grads_second_term = self._hessian_vector_product_adv(vector, std_loss=True, input=input_train, target=target_train)
    print("-------hessian_vector_product_adv 2: {:.3f}".format(time.time() - time_stamp))

    # Second-order (3rd term) \nabla_alpha L^{std}_val(w', alpha) \nabla^2_{alpha,w} L^{std}_train(w', alpha)
    #grad_w_std_loss = torch.autograd.grad(natural_val_loss, unrolled_model.parameters())
    vector = [v.detach() for v in grad_w_std_loss]
    time_stamp = time.time()
    implicit_grads_third_term = self._hessian_vector_product_adv(vector, std_loss=True, input=input_train, target=target_train)
    print("-------hessian_vector_product_adv 3: {:.3f}".format(time.time() - time_stamp))

    # Second-order (4th term) \nabla_alpha L^{adv}_val(w', alpha) \nabla^2_{alpha,w} L^{adv}_train(w', alpha)
    #grad_w_adv_loss = torch.autograd.grad(adv_val_loss, unrolled_model.parameters())
    vector = [v.detach() for v in grad_w_adv_loss]
    time_stamp = time.time()
    implicit_grads_fourth_term = self._hessian_vector_product_adv(vector, std_loss=False, input=input_train, target=target_train)
    print("-------hessian_vector_product_adv 4: {:.3f}".format(time.time() - time_stamp))

    time_stamp = time.time()
    # \nabla_alpha = \nabla_alpha (0.5 * L^{adv}_val(w', alpha) + 0.5 * L^{std}_val(w', alpha)) -
    # eta * lambda_1 * lambda_2 * \nabla^2_{alpha,w} L^{adv}_train(w', alpha) -
    # eta * lambda_1 * lambda_2 * \nabla^2_{alpha,w} L^{std}_train(w', alpha) -
    # eta * lambda_1 * lambda_1 * \nabla^2_{alpha,w} L^{std}_train(w', alpha) -
    # eta * lambda_2 * lambda_2 * \nabla^2_{alpha,w} L^{adv}_train(w', alpha)
    grads = []
    for dalpha, h_adv_v_std, h_std_v_adv, h_std_v_std, h_adv_v_adv in zip(
        grad_first_order_terms,
        implicit_grads_first_term,
        implicit_grads_second_term,
        implicit_grads_third_term,
        implicit_grads_fourth_term
    ):
        g = dalpha \
            - eta * ( self.lambda_1 * self.lambda_2 * h_adv_v_std
                    + self.lambda_1 * self.lambda_2 * h_std_v_adv
                    + (self.lambda_1 ** 2) * h_std_v_std
                    + (self.lambda_2 ** 2) * h_adv_v_adv)
        grads.append(g)
    print("-------compute final grads: {:.3f}".format(time.time() - time_stamp))

    time_stamp = time.time()
    for v, g in zip(self.model.arch_parameters(), grads):
        if v.grad is None:
            v.grad = g.detach().clone()
        else:
            with torch.no_grad():
                v.grad.copy_(g.detach())
    print("-------copy to self.model.arch_parameters: {:.3f}".format(time.time() - time_stamp))


  def _hessian_vector_product_adv(self, vector, std_loss, input, target, r=1e-2):
    # store original params
    original_params = [p.detach().clone() for p in self.model.parameters()]

    # L^{std}_train (w, alpha)
    input_valid = input
    if not std_loss:
      # L^{adv}_train (w, alpha)
      attack = self.attack_f(self.model)
      adv_input = attack(input, target)
      input_valid = adv_input

    # epsilon = 1e-2 / ||v||
    epsilon = r / max(self._concat(vector).norm(), 1e-8)

    # w = w + epsilon*v
    with torch.no_grad():
      for p, v in zip(self.model.parameters(), vector):
        p.data.add_(v, alpha=epsilon)
    loss = self.model._loss(input_valid, target)
    # \nabla_alpha L_train(w+epsilon*v, alpha)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # w = w - 2*epsilon*v (= w - epsilon*v - epsilon*v)
    with torch.no_grad():
      for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(v, alpha=2 * epsilon)
    loss = self.model._loss(input_valid, target)
    # \nabla_alpha L_train(w-epsilon*v, alpha)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    with torch.no_grad():
      for p, original_p in zip(self.model.parameters(), original_params):
        p.data.copy_(original_p)

    # ( \nabla_alpha L_train(w+epsilon*v, alpha) - \nabla_alpha L_train(w-epsilon*v, alpha) ) / (2*epsilon)
    return [(x - y).div_(2 * epsilon) for x, y in zip(grads_p, grads_n)]

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.to(self.device)