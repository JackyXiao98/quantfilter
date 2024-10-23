import torch
import numpy as np
from tqdm import tqdm
from tqdm._utils import _term_move_up
import torch.autograd as ta

def train(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)
        def closure(scale = 1.0):
            predict = model(input)
            if not isinstance(predict, torch.Tensor):
                predict = predict.logits
            loss = criterion(predict, label)
            scaled_loss = loss*scale
            scaled_loss.backward()
            return loss, predict
        
        if hasattr(optimizer, 'prestep'):
            loss, predict = optimizer.prestep(closure)
        else:
            loss, predict = closure()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        del input
        del label
        del loss
        del predict

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])
def noisy_train(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, noise = 0, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    snr = 0
    for t, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)
        def closure(scale = 1.0):
            predict = model(input)
            if not isinstance(predict, torch.Tensor):
                predict = predict.logits
            loss = criterion(predict, label)
            scaled_loss = loss*scale
            scaled_loss.backward()
            return loss, predict
        
        if hasattr(optimizer, 'prestep'):
            loss, predict = optimizer.prestep(closure)
        else:
            loss, predict = closure()

        train_loss = loss.item()
        _, predicted = predict.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            model, snr = add_noise(model, noise)
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        if t==0 or (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d) | SNR: %-.6f'% (epoch, t+1, train_loss, 100.*correct/total, correct, total, snr))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss, snr])

    return model

def train_nlp(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, input in enumerate(train_dl):
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        input_ids = input['input_ids'].to(device)
        label = input['labels'].to(device)
        attn_mask = input['attention_mask'].to(device)
        output = model(input_ids = input_ids, attention_mask = attn_mask, labels = label)
        loss = output.loss
        predict = output.logits
        # if not isinstance(predict, torch.Tensor):
        #     predict = predict.logits
        # loss = criterion(predict, label)
        loss.backward()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])

@torch.no_grad()
def test(model, test_dl, criterion, log_file, device = 'cpu', epoch = -1, **kwargs):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.logits
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(" ")
    log_file.update([epoch, -1],[100.*correct/total, test_loss/(batch_idx+1)])

@torch.no_grad()
def test_nlp(model, test_dl, criterion, log_file, device = 'cpu', epoch = -1, **kwargs):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_dl):
            input_ids = inputs['input_ids'].to(device)
            targets = inputs['labels'].to(device)
            attn_mask = inputs['attention_mask'].to(device)
            output = model(input_ids = input_ids, attention_mask = attn_mask, labels = targets)
            loss = output.loss
            outputs = output.logits

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(" ")
    log_file.update([epoch, -1],[100.*correct/total, test_loss/(batch_idx+1)])

@torch.no_grad()
def record_G(model, mode = 'coordinate', position = 0):
    G_t = []
    if mode == 'coordinate':
        for name, param in model.named_parameters():
            if param.requires_grad:
                G_t.append(param.grad.view(-1)[position].item())
    elif mode == 'layer':
        i = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if i == position:
                    G_t = param.grad.view(-1).tolist()
                    break
                else:
                    i = i+1
                    past_param = param
        if len(G_t) == 0:
            G_t = past_param.grad.view(-1).tolist()
    return torch.tensor(G_t)

@torch.no_grad()
def add_noise(model, noise_multiplier):
    noise_norm = 0
    grad_norm = 0
    # first = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_norm += (param.grad.view(-1).norm(2))**2
            noise = torch.normal(
                mean=0,
                std=noise_multiplier,
                size=param.size(),
                device=param.device,
                dtype=param.dtype,
            )
            param.grad += noise
            # if first:
            #     print((noise.view(-1).norm(2))**2, noise.numel())
                # first = False
            noise_norm += (noise.view(-1).norm(2))**2
            del noise
    return model, grad_norm.item()/noise_norm.item()

import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def zo_perturb(model, seed, scale):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data.add_(z, alpha = scale)

def zo_perturb_and_create_grad(model, seed, scale, loss_diff):
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.requires_grad:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data.add_(z, alpha = scale)
            if param.grad is None:
                param.grad = z * loss_diff / (2.0*scale)
            else:
                param.grad.add_(z, alpha = loss_diff / (2.0*scale))


def zo_backward(model, criterion, input, label, scale):
    old_seed = torch.seed()
    new_seed = np.random.randint(1000000000)
    model.eval()
    with torch.inference_mode():
        zo_perturb(model, new_seed, scale)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss_diff = criterion(predict, label)
        zo_perturb(model, new_seed, -2*scale)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss_diff -= criterion(predict, label)
        zo_perturb_and_create_grad(model, new_seed, scale, loss_diff)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
    torch.manual_seed(old_seed + new_seed)
    return loss, predict

@torch.inference_mode()
def train_zo(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    # model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)
        def closure(scale = 1.0):
            loss, predict = zo_backward(model, criterion, input, label, scale = 1e-3)
            return loss, predict
        
        if hasattr(optimizer, 'prestep'):
            loss, predict = optimizer.prestep(closure)
        else:
            loss, predict = closure()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])


def train_R(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None, mode = 'coordinate', position = 0):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        input = input.to(device)
        label = label.to(device)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            gammas, _ = optimizer.step()
            # print(gammas)
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                gamma_min = min(gammas)
                gamma_max = max(gammas)
                gamma_avg = sum(gammas)/len(gammas)
                gamma_std = np.std(gammas)
                log_file.update([epoch, t],[100.*correct/total, train_loss, gamma_min, gamma_max, gamma_avg, gamma_std])

def train_3(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        input = input.to(device)
        label = label.to(device)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])

def train_2(model, train_dl, optimizer, criterion, log_file, device = 'cpu', epoch = -1, log_frequency = -1, acc_step = 1, lr_scheduler = None, gamma=0.1):
    model.to(device)
    model.train()
    train_loss = 0
    total = 0
    correct = 0
    # print(" ")
    for t, (input, label) in enumerate(train_dl):
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        # with torch.autocast(device_type="cuda", dtype=torch.float16):
        input = input.to(device)
        label = label.to(device)
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()
        ### the grad is computed at x_t_plus
        for p in model.parameters():
            if p.requires_grad:
                if not hasattr(p,'grad_t_plus'):
                    if hasattr(p,'private_grad'):
                        p.grad_t_plus = p.private_grad; p.private_grad=torch.zeros_like(p.data)
                    else:
                        p.grad_t_plus = p.grad
                else:
                    if hasattr(p,'private_grad'):
                        p.grad_t_plus += p.private_grad; p.private_grad=torch.zeros_like(p.data)
                    else:
                        p.grad_t_plus += p.grad
                del p.grad

        ### change back to x_t
        for p in model.parameters():
            if p.requires_grad:
                p.data=p.x_t.clone()
        predict = model(input)
        if not isinstance(predict, torch.Tensor):
            predict = predict.logits
        loss = criterion(predict, label)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                if not hasattr(p,'grad_t'):
                    if hasattr(p,'private_grad'):
                        p.grad_t = p.private_grad; p.private_grad=torch.zeros_like(p.data)
                    else:
                        p.grad_t = p.grad
                else:
                    if hasattr(p,'private_grad'):
                        p.grad_t += p.private_grad; p.private_grad=torch.zeros_like(p.data)
                    else:
                        p.grad_t += p.grad
                del p.grad
        ########
        
        
        train_loss = loss.item()
        _, predicted = predict.max(1)
        total = total + label.size(0)
        correct = correct + predicted.eq(label).sum().item()

        if ((t + 1) % acc_step == 0) or ((t + 1) == len(train_dl)):
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
            # optimizer.prestep()
            optimizer.zero_grad()

        if (t+1)%(acc_step)== 0 or ((t + 1) == len(train_dl)):
            print('Epoch: %d:%d Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, t+1, train_loss, 100.*correct/total, correct, total))
            if log_frequency>0 and ((t+1)%(acc_step*log_frequency) == 0 or t+1 == len(train_dl)):
                log_file.update([epoch, t],[100.*correct/total, train_loss])
