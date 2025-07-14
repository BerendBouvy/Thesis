import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np

@dataclass
class VAEOutput:
    """
    Dataclass to hold the output of the VAE forward pass.
    
    Attributes:
        z_dist (torch.distributions.MultivariateNormal): Distribution of the latent space.
        z_sample (torch.Tensor): Sampled data from the latent space.
        x_recon (torch.Tensor): Reconstructed data in the original input space.
        loss (torch.Tensor): Total loss computed during the forward pass.
        loss_recon (torch.Tensor): Reconstruction loss.
        loss_kl (torch.Tensor): KL divergence loss.
    """
    z_dist: torch.distributions.MultivariateNormal
    z_sample: torch.Tensor
    x_recon: torch.Tensor
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_kl: torch.Tensor

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) class.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        density (int): Density multiplier for hidden layers.
    """
    
    def __init__(self, input_dim, latent_dim, density: int=1):
        super(VAE, self).__init__()
        
        self.smallest_hid_dim = 2**np.ceil(np.log2(latent_dim))  # Ensure the smallest hidden dimension is a power of 2.
        self.first_hid_dim = max(self.smallest_hid_dim, 2**np.ceil(np.log2(input_dim)))  # Ensure the first hidden dimension is at least as large as input_dim.
        steps = int(np.log2(self.first_hid_dim / self.smallest_hid_dim))
        architecture = [self.first_hid_dim // (2**i) for i in range(steps + 1)]
        dens_architecture = [item for item in architecture for _ in range(density)]
        dens_architecture = [int(dim) for dim in dens_architecture]
        dens_architecture_complete = [input_dim] + dens_architecture + [2*latent_dim]
        encoder = []
        for i in range(len(dens_architecture_complete) - 1):
            encoder.append(nn.Linear(dens_architecture_complete[i], dens_architecture_complete[i + 1]))
            encoder.append(nn.SiLU())

            
        self.encoder = nn.Sequential(*encoder)
        
        self.softplus = nn.Softplus()
        
        decoder = []
        reversed_architecture = dens_architecture_complete[::-1]
        reversed_architecture[0] = latent_dim  # The first layer of the decoder should match the latent dimension.
        for i in range(len(reversed_architecture) - 2):
            decoder.append(nn.Linear(reversed_architecture[i], reversed_architecture[i + 1]))
            decoder.append(nn.SiLU())
            
        decoder.append(nn.Linear(reversed_architecture[-2], reversed_architecture[-1]))
        
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x, eps: float = 1e-8):      
        """
        Encodes the input data into the latent space.
        
        Args:
            x (torch.Tensor): Input data.
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            torch.distributions.MultivariateNormal: Normal distribution of the encoded data.
        """
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)
        
    def reparameterize(self, dist):
        """
        Reparameterizes the encoded data to sample from the latent space.
        
        Args:
            dist (torch.distributions.MultivariateNormal): Normal distribution of the encoded data.

        Returns:
            torch.Tensor: Sampled data from the latent space.
        """
        return dist.rsample()
    
    def decode(self, z):
        """
        Decodes the data from the latent space to the original input space.
        
        Args:
            z (torch.Tensor): Data in the latent space.
        
        Returns:
            torch.Tensor: Reconstructed data in the original input space.
        """
        return self.decoder(z)
    
    def forward(self, x, compute_loss: bool = True):
        """
        Performs a forward pass of the VAE.
        
        Args:
            x (torch.Tensor): Input data.
            compute_loss (bool): Whether to compute the loss or not.
        
        Returns:
            VAEOutput: VAE output dataclass.
        """
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        
        if not compute_loss:
            return VAEOutput(
                z_dist=dist,
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # compute loss terms
        loss_recon = F.mse_loss(recon_x, x, reduction='mean')
        std_normal = torch.distributions.MultivariateNormal(
            torch.zeros_like(z, device=z.device),
            scale_tril=torch.eye(z.shape[-1], device=z.device).unsqueeze(0).expand(z.shape[0], -1, -1),
        )
        loss_kl = torch.distributions.kl.kl_divergence(dist, std_normal).mean()
                
        loss = loss_recon + loss_kl
        
        return VAEOutput(
            z_dist=dist,
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=loss_kl,
        )