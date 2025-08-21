from sklearn.preprocessing import scale
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
    # z_dist: torch.distributions.MultivariateNormal
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
    
    def __init__(self, input_dim, latent_dim, density: int=1, beta: float=1.0):
        super(VAE, self).__init__()
        
        self.beta = beta
        self.latent_dim = latent_dim
        self.hidden_layer_size = 50
        
        encoder = []

        encoder.append(nn.Linear(input_dim, self.hidden_layer_size))
        encoder.append(nn.SiLU())
        for _ in range(density):
            encoder.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            encoder.append(nn.SiLU())
        encoder.append(nn.Linear(self.hidden_layer_size, latent_dim))

            
        self.encoder = nn.Sequential(*encoder)
        
        self.softplus = nn.Softplus()
        
        decoder = []
        decoder.append(nn.Linear(latent_dim, self.hidden_layer_size))
        decoder.append(nn.SiLU())
        for _ in range(density):
            decoder.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            decoder.append(nn.SiLU())
        decoder.append(nn.Linear(self.hidden_layer_size, input_dim))
        
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
        z = self.encoder(x)
        return z
        
        
    
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
        z = self.encode(x)
        recon_x = self.decode(z)

        if not compute_loss:
            return VAEOutput(
                z_sample=z,
                x_recon=recon_x,
                loss=None,
                loss_recon=None,
                loss_kl=None,
            )

        # compute loss terms
        loss_recon = F.mse_loss(recon_x, x, reduction='mean')


        loss = loss_recon

        return VAEOutput(
            z_sample=z,
            x_recon=recon_x,
            loss=loss,
            loss_recon=loss_recon,
            loss_kl=torch.tensor(0.0)  # KL loss is not computed in this version,
        )
        
    def set_beta(self, beta: float):
        """
        Sets the beta value for the VAE.
        
        Args:
            beta (float): New beta value.
        """
        self.beta = beta