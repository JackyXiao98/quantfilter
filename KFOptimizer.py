import torch
from torch.optim.optimizer import Optimizer, required
import torch.autograd as ta
from collections import defaultdict
from typing import Callable, List, Optional, Union

class KFOptimizer(Optimizer):
    def __init__(self, params, optimizer:Optimizer, kappa = 0.9, gamma = 1.0):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        if gamma ==0:
            gamma = (1-kappa)/kappa
            self.compute_grad = False
        elif abs(gamma - (1-kappa)/kappa) <1e-3:
            gamma = (1-kappa)/kappa
            self.compute_grad = False
        else:
            self.scaling_factor = (gamma*kappa+kappa-1)/(1-kappa)
            self.compute_grad = True
        defaults = dict(kappa = kappa, gamma=gamma)
        self.optimizer = optimizer
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFOptimizer, self).__setstate__(state)

    def prestep(self, closure=required):
        loss = None
        for group in self.param_groups:
            gamma = group['gamma']
            break
        if self.compute_grad:
            with torch.enable_grad():
                loss = closure() # compute grad
        # totoal_grad = 0
        for group in self.param_groups:
            gamma = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb
                p.data.add_(state['kf_d_t'], alpha = gamma)
                if self.compute_grad:
                    if hasattr(p, 'private_grad'):
                        p.private_grad.mul_(self.scaling_factor)
                    elif p.grad is not None:
                        p.grad.mul_(self.scaling_factor)
                    else:
                        raise RuntimeError("Must have either grad or private_grad!")
        with torch.enable_grad():
            if self.compute_grad:
                closure()
            else:
                loss = closure()
        for group in self.param_groups:
            gamma = group['gamma']
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb back
                p.data.add_(state['kf_d_t'], alpha = -gamma)
                if self.compute_grad:
                    if hasattr(p, 'private_grad'):
                        p.private_grad.div_(self.scaling_factor)
                    elif p.grad is not None:
                        p.grad.div_(self.scaling_factor)
        return loss
            

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        scaling_factor = 0.0
        for group in self.param_groups:
            kappa = group['kappa']
            for p in group['params']:
                has_private_grad = False
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                if self.compute_grad:
                    grad.div_(1+1/self.scaling_factor)
                state = self.state[p]
                if 'kf_d_t' not in state:
                    state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    state['kf_m_t'] = grad.clone().to(p.data)
                state['kf_m_t'].lerp_(grad, weight = kappa)
                if has_private_grad:
                    p.private_grad = state['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = state['kf_m_t'].clone().to(p.data)
                    scaling_factor += p.grad.norm().pow(2)
                state['kf_d_t'] = -p.data.clone().to(p.data)
        if scaling_factor > 0 and not has_private_grad:
            scaling_factor = scaling_factor.sqrt()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.div_(scaling_factor)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['kf_d_t'].add_(p.data, alpha = 1)
        return loss

class KFOptimizer_2(Optimizer):
    def __init__(self, optimizer:Optimizer, kappa = 0.7, gamma = 0.5):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer_2(optimizer, kappa, gamma)
        '''
        self.scaling_factor = 1.0
        self.compute_grad_at_original = False
        if gamma == 0 or abs(gamma - (1-kappa)/kappa) <1e-3:
            gamma = (1-kappa)/kappa
        else:
            self.scaling_factor = (1-kappa)/(gamma*kappa)#(gamma*kappa+kappa-1)/(1-kappa)
            self.compute_grad_at_original = True
        self.gamma = gamma
        self.kappa = kappa
        self.optimizer = optimizer

    @property
    def param_groups(self) -> List[dict]:
        """
        Returns a list containing a dictionary of all parameters managed by the optimizer.
        """
        return self.original_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups: List[dict]):
        """
        Updates the param_groups of the optimizer.
        """
        self.original_optimizer.param_groups = param_groups

    @property
    def state(self) -> defaultdict:
        """
        Returns a dictionary holding current optimization state.
        """
        return self.original_optimizer.state

    @state.setter
    def state(self, state: defaultdict):
        """
        Updates the state of the optimizer.
        """
        self.original_optimizer.state = state

    @property
    def defaults(self) -> dict:
        """
        Returns a dictionary containing default values for optimization.
        """
        return self.original_optimizer.defaults

    @defaults.setter
    def defaults(self, defaults: dict):
        """
        Updates the defaults of the optimizer.
        """
        self.original_optimizer.defaults = defaults

    def backward(self, closure_before_backward=required):
        loss = None
        if self.compute_grad_at_original:
            with torch.enable_grad():
                loss = (1-self.scaling_factor)*closure_before_backward() # compute grad
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb
                p.data = p.data.add(state['kf_d_t'], alpha = self.gamma)
        with torch.enable_grad():
            if self.compute_grad_at_original:
                loss += self.scaling_factor*closure_before_backward()
            else:
                loss = self.scaling_factor*closure_before_backward()
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'kf_d_t' not in state:
                    continue
                # perturb
                p.data = p.data.add(state['kf_d_t'], alpha = -self.gamma)
        return loss
            

    def step(self, closure: Optional[Callable[[], float]] = None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        scaling_factor = 0.0
        for group in self.param_groups:
            for p in group['params']:
                has_private_grad = False
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                state = self.state[p]
                if 'kf_d_t' not in state:
                    state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    state['kf_m_t'] = grad.clone().to(p.data)
                state['kf_m_t'].lerp_(grad, weight = self.kappa)
                if has_private_grad:
                    p.private_grad = state['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = state['kf_m_t'].clone().to(p.data)
                    scaling_factor += p.grad.norm().pow(2)
                state['kf_d_t'] = -p.data.clone().to(p.data)
        if scaling_factor > 0 and not has_private_grad:
            scaling_factor = scaling_factor.sqrt()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.div_(scaling_factor)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['kf_d_t'].add_(p.data, alpha = 1)
        return loss

class KFOptimizer_3(Optimizer):
    def __init__(self, params, optimizer:Optimizer, sigma_H=3e-6, sigma_g=1e-5):
        '''
        # wrapping up the optimizer with
        optimizer = KFOptimizer(model.parameters(), optimizer, sigma_H, sigma_g)
        # before the first step of gradient accumulation:
        if t % acc_step == 0 and hasattr(optimizer, 'prestep'):
            optimizer.prestep()
        '''
        defaults = dict(sigma_H=sigma_H, sigma_g=sigma_g)
        self.optimizer = optimizer
        # if nesterov and (momentum <= 0 or dampening != 0):
            # raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(KFOptimizer_2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(KFOptimizer_2, self).__setstate__(state)

    def hessian_d_product(self):
        """
        evaluate hessian vector product
        """
        loss = 0
        G = []
        X = []
        D = []
        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                if p.requires_grad:
                    G.append(p.grad)
                    # print(p.grad)
                    X.append(p)
                    state = self.state[p]
                    if 'd_t' not in state:
                        state['d_t'] = torch.zeros_like(p.grad).to(p.grad)
                    D.append(state['d_t'])
        Hd = ta.grad(G, X, D, retain_graph=False)
        # print(Hd)
        Hd = list(Hd)
        # print("Hd: ", Hd)
        # blk_idx = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'Hd_t' not in state or state['Hd_t'] is None:
                    state['Hd_t'] = Hd.pop(0)
                else:
                    state['Hd_t'].add_(Hd.pop(0))
        return Hd

    def prestep(self):
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'kf_beta_t' not in state:
                    continue
                beta_t = state['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
                k_1 = (1-k_t)/k_t
                p.data.add_(state['kf_d_t'], alpha = k_1)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                has_private_grad = False
                if p.grad is None:
                    continue
                if hasattr(p, 'private_grad'):
                    grad = p.private_grad
                    has_private_grad = True
                elif p.grad is not None:
                    grad = p.grad
                else:
                    continue
                state = self.state[p]
                if 'kf_beta_t' not in state:
                    state['kf_beta_t'] = 1
                    state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    state['kf_m_t'] = grad.clone().to(p.data)
                beta_t = state['kf_beta_t'] + sigma_H**2
                k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2)
                k_1 = (1-k_t)/k_t
                state['kf_beta_t'] = (1-k_t)*beta_t
                p.data.add_(state['kf_d_t'], alpha = -k_1)
                state['kf_m_t'].lerp_(grad, weight = k_t)
                if has_private_grad:
                    p.private_grad = state['kf_m_t'].clone().to(p.data)
                else:
                    p.grad = state['kf_m_t'].clone().to(p.data)
                state['kf_d_t'] = -p.data.clone().to(p.data)
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            sigma_g = group['sigma_g']
            sigma_H = group['sigma_H']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['kf_d_t'].add_(p.data, alpha = 1)
                # beta_t = state['kf_beta_t'] + sigma_H**2
                # k_t = beta_t/(beta_t + sigma_g**2 - sigma_H**2 )
                # k_1 = (1-k_t)/k_t
                # p.data.add_(state['kf_d_t'], alpha = k_1)
        return loss

