import torch
import opacus
from typing import List, Union
import os

def generate_noise(max_norm, parameter, noise_multiplier, noise_type, device):
    """
    A noise generation function that can utilize different distributions for noise generation.

    @param max_norm
        The maximum norm of the per-sample gradients. Any gradient with norm
        higher than this will be clipped to this value.
    @param parameter
        The parameter, based on which the dimension of the noise tensor
        will be determined
    @param noise_multiplier
        The ratio of the standard deviation of the Gaussian noise to
        the L2-sensitivity of the function to which the noise is added
    @param noise_type
        Sets the distribution for the noise generation.
        See generate_noise for supported strings.
    @param device
        The device used for calculations and needed for tensor definition.

    @return
        a tensor of noise in the same shape as ``parameter``.
    """
    if noise_multiplier > 0:
        mean = 0
        scale_scalar = noise_multiplier * max_norm

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

# Server side Noise
def apply_noise(weights, batch_size, noise_multiplier, noise_type, device, loss_reduction="mean"):
    """
    A function for applying noise to weights on the (intermediate) server side that utilizes the generate_noise function above.

    @param weights
        The weights to which to apply the noise.
    @param batch_size
        Batch size used for averaging.
    @param noise_multiplier
        The ratio of the standard deviation of the Gaussian noise to
        the L2-sensitivity of the function to which the noise is added
    @param noise_type
        Sets the distribution for the noise generation.
        See generate_noise for supported strings.
    @param device
        The device used for calculations and needed for tensor definition.
    @param loss_reduction
        The method of loss reduction.
        currently supported: mean
    """
    for p in weights.values():
        noise = generate_noise(0, p, noise_multiplier, noise_type, device)
        if loss_reduction == "mean":
            noise /= batch_size
        p += noise

# Client side Noise
class PrivacyEngineXL(opacus.PrivacyEngine):
    """
    A privacy engine that can utilize different distributions for noise generation, based on opacus' privacy engine.
    It gets attached to the optimizer just like the privacy engine from opacus.

    @param module:
        The Pytorch module to which we are attaching the privacy engine
    @param batch_size
        Training batch size. Used in the privacy accountant.
    @param sample_size
        The size of the sample (dataset). Used in the privacy accountant.
    @param alphas
        A list of RDP orders
    @param noise_multiplier
        The ratio of the standard deviation of the Gaussian noise to
        the L2-sensitivity of the function to which the noise is added
    @param max_grad_norm
        The maximum norm of the per-sample gradients. Any gradient with norm
        higher than this will be clipped to this value.
    @param secure_rng
        If on, it will use ``torchcsprng`` for secure random number generation. Comes with
        a significant performance cost, therefore it's recommended that you turn it off when
        just experimenting.
    @param grad_norm_type
        The order of the norm. For instance, 2 represents L-2 norm, while
        1 represents L-1 norm.
    @param batch_first
        Flag to indicate if the input tensor to the corresponding module
        has the first dimension representing the batch. If set to True,
        dimensions on input tensor will be ``[batch_size, ..., ...]``.
    @param target_delta
        The target delta
    @param loss_reduction
        Indicates if the loss reduction (for aggregating the gradients)
        is a sum or a mean operation. Can take values "sum" or "mean"
    @param noise_type
        Sets the distribution for the noise generation.
        See generate_noise for supported strings.
    @param **misc_settings
        Other arguments to the init
    """

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
        """
        Generates a tensor of noise in the same shape as ``parameter``.

        @param max_norm
            The maximum norm of the per-sample gradients. Any gradient with norm
            higher than this will be clipped to this value.
        @param parameter
            The parameter, based on which the dimension of the noise tensor
            will be determined

        @return
            a tensor of noise in the same shape as ``parameter``.
        """
        return generate_noise(max_norm, parameter, self.noise_multiplier, self.noise_type, self.device)
