import torch
import opacus
from typing import List, Union
import os

class PrivacyEngineXL(opacus.PrivacyEngine):

    def __init__(
        self,
        module: torch.nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        secure_rng: bool = False,
        grad_norm_type: int = 2,
        batch_first: bool = True,
        target_delta: float = 1e-6,
        loss_reduction: str = "mean",
        noise_type: str="gaussian",
        **misc_settings
    ):

        import warnings
        if secure_rng:
            warnings.warn(
                "Secure RNG was turned on. However it is not yet implemented for the noise distributions of privacy_engine_xl."
            )

        opacus.PrivacyEngine.__init__(
            self,
            module,
            batch_size,
            sample_size,
            alphas,
            noise_multiplier,
            max_grad_norm,
            secure_rng,
            grad_norm_type,
            batch_first,
            target_delta,
            loss_reduction,
            **misc_settings)

        self.noise_type = noise_type

    def _generate_noise(self, max_norm, parameter):
        if self.noise_multiplier > 0:
            mean = 0
            scale_scalar = self.noise_multiplier * max_norm

            scale = torch.full(size=parameter.grad.shape, fill_value=scale_scalar, dtype=torch.float32, device=self.device)

            if self.noise_type == "gaussian":
                dist = torch.distributions.normal.Normal(mean, scale)
            elif self.noise_type == "laplacian":
                dist = torch.distributions.laplace.Laplace(mean, scale)
            elif self.noise_type == "exponential":
                rate = 1 / scale
                dist = torch.distributions.exponential.Exponential(rate)
            else:
                dist = torch.distributions.normal.Normal(mean, scale)

            noise = dist.sample()

            return noise
        return 0.0
