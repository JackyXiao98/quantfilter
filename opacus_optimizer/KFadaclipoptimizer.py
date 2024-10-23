from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required



logger = logging.getLogger(__name__)

class KF_AdaClipDPOptimizer(AdaClipDPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        target_unclipped_quantile: float,
        clipbound_learning_rate: float,
        max_clipbound: float,
        min_clipbound: float,
        unclipped_num_std: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        kappa: float = 0.7,
        gamma: float = 0.5
    ):
        super(AdaClipDPOptimizer).__init__(
            optimizer,
            noise_multiplier = noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            target_unclipped_quantile=target_unclipped_quantile,
            clipbound_learning_rate=clipbound_learning_rate,
            max_clipbound=max_clipbound,
            min_clipbound=min_clipbound,
            unclipped_num_std=unclipped_num_std
        )
        if gamma ==0:
            gamma = (1-kappa)/kappa
            self.kf_compute_grad = False
        elif abs(gamma - (1-kappa)/kappa) <1e-3:
            gamma = (1-kappa)/kappa
            self.kf_compute_grad = False
        else:
            self.scaling_factor = (1-kappa)/(gamma * kappa)#(gamma*kappa+kappa-1)/(1-kappa)
            self.kf_compute_grad_at_original = True
        self.kappa = kappa
        self.gamma = gamma
    
    def _compute_one_closure(self, closure=required):
        loss = None
        has_kf_d_t = True
        for p in self.params:
            state = self.state[p]
            if 'kf_d_t' not in state:
                has_kf_d_t = False
                continue
            # perturb
            p.data.add_(state['kf_d_t'], alpha = self.gamma)
        with torch.enable_grad():
            loss = closure()
        if has_kf_d_t:
            for p in self.params:
                state = self.state[p]
                # perturb back
                p.data.add_(state['kf_d_t'], alpha = -self.gamma)
        return loss
    
    def _compute_two_closure(self, closure=required):
        loss = None
        has_kf_d_t = True
        with torch.enable_grad():
            closure()
        for p in self.params:
            state = self.state[p]
            if 'kf_d_t' not in state:
                has_kf_d_t = False
                continue
            # perturb
            p.data.add_(state['kf_d_t'], alpha = self.gamma)
        # store first set of gradient
        if has_kf_d_t:
            if self.grad_samples is not None and len(self.grad_samples) != 0:
                self.past_grad_samples = self.grad_samples
                for grad in self.past_grad_samples:
                    grad.mul_(1.0 - self.scaling_factor)
                self.grad_samples = None
            with torch.enable_grad():
                loss = closure()
            for p in self.params:
                state = self.state[p]
                # perturb back
                p.data.add_(state['kf_d_t'], alpha = -self.gamma)
            if self.grad_samples is not None and len(self.grad_samples) != 0:
                for grad, past_grad in zip(self.grad_samples, self.past_grad_samples):
                    grad.mul_(self.scaling_factor).add_(past_grad)
                self.past_grad_samples = None
        return loss

    def step(self, closure=required) -> Optional[float]:
        if self.kf_compute_grad_at_original:
            loss = self._compute_two_closure(closure)
        else:
            loss = self._compute_one_closure(closure)

        if self.pre_step():
            for p in self.params:
                grad = p.grad
                state = self.state[p]
                if 'kf_d_t' not in state:
                    state['kf_d_t'] = torch.zeros_like(p.data).to(p.data)
                    state['kf_m_t'] = grad.clone().to(p.data)
                state['kf_m_t'].lerp_(grad, weight = self.kappa)
                p.grad = state['kf_m_t'].clone().to(p.data)
                state['kf_d_t'] = -p.data.clone().to(p.data)
            self.original_optimizer.step()
            for p in self.params:
                self.state[p]['kf_d_t'].add_(p.data, alpha = 1.0)
        return loss