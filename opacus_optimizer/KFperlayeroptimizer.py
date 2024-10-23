from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, List, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from opacus.optimizers.utils import params

from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from .KFoptimizer import KF_DPOptimizer


logger = logging.getLogger(__name__)
logger.disabled = True

class KF_DPPerLayerOptimizer(KF_DPOptimizer):
    def __init__(
            self,
            optimizer: Optimizer,
            *,
            noise_multiplier: float,
            max_grad_norm: List[float],
            expected_batch_size: Optional[int],
            loss_reduction: str = "mean",
            generator=None,
            secure_mode: bool = False,
            kappa = 0.7,
            gamma = 0.5
    ):
        assert len(max_grad_norm) == len(params(optimizer))
        self.max_grad_norms = max_grad_norm
        max_grad_norm = torch.norm(torch.Tensor(self.max_grad_norms), p=2).item()
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            kappa=kappa,
            gamma=gamma
        )

    def clip_and_accumulate(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)
            per_sample_norms = grad_sample.norm(
                2, dim=tuple(range(1, grad_sample.ndim))
            )
            per_sample_clip_factor = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(
                max=1.0
            )
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)