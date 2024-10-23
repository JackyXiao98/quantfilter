from opacus.privacy_engine import PrivacyEngine
from opacus.optimizers import DPOptimizer
import torch
from torch import optim
from opacus_optimizer import get_optimizer_class
from typing import List, Union

class KF_PrivacyEngine(PrivacyEngine):
    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False):
        super().__init__(accountant=accountant, secure_mode=secure_mode)
    
    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
        kalman: bool=False,
        **kwargs,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
            kalman=kalman
        )

        return optim_class(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            **kwargs,
        )