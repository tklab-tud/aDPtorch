import torch
import opacus
from typing import List, Union
import os

def generate_noise(max_norm, parameter, sigma, noise_type, device):
    if sigma > 0:
        mean = 0
        scale_scalar = sigma * max_norm

        scale = torch.full(size=parameter.shape, fill_value=scale_scalar, dtype=torch.float32, device=device)

        if noise_type.lower() in ["normal", "gauss", "gaussian"]:
            dist = torch.distributions.normal.Normal(mean, scale)
        elif noise_type.lower() in ["laplace", "laplacian"]:
            dist = torch.distributions.laplace.Laplace(mean, scale)
        elif noise_type.lower() in ["exponential"]:
            rate = 1 / scale
            dist = torch.distributions.exponential.Exponential(rate)
        else:
            dist = torch.distributions.normal.Normal(mean, scale)

        noise = dist.sample()

        return noise
    return 0.0

def apply_noise(weights, batch_size, sigma, noise_type, device, loss_reduction="mean"):
    for p in weights.values():
        noise = generate_noise(0, p, sigma, noise_type, device)
        if loss_reduction == "mean":
            noise /= batch_size
        p += noise

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
        return generate_noise(max_norm, parameter, self.noise_multiplier, self.noise_type, self.device)
