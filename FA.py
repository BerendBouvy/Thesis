from sklearn.decomposition import FactorAnalysis
from dataLoader import data_loader
import torch
from train_model import train_model
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class FactorAnalysisWithInverse(FactorAnalysis):
    def inverse_transform(self, Z):
        """
        Reconstructs the input from latent variables.
        
        Args:
            Z (ndarray): Latent representation of shape (n_samples, n_components)
        
        Returns:
            X_recon (ndarray): Reconstructed input of shape (n_samples, n_features)
        """
        return Z @ self.components_ + self.mean_


def train_fa_model(train_loader, n_components=10, n_iter=1000, tol=1e-2):
    """
    Train a Factor Analysis model on the given dataset.
    
    Args:
        train_loader (DataLoader): DataLoader for the training set.
        n_components (int): Number of components for Factor Analysis.
        n_iter (int): Number of iterations for the optimization.
        tol (float): Tolerance for convergence.
    
    Returns:
        model (FactorAnalysis): Trained Factor Analysis model.
        scaler (Normalizer): Normalizer used for scaling the dataset.
    """
    
    
    # Initialize the Factor Analysis model
    model = FactorAnalysisWithInverse(n_components=n_components, max_iter=n_iter, tol=tol)

    # Fit the model on the training data
    for data in train_loader:
        data = data[0].numpy()  # Convert to numpy array
        model.fit(data)
        
    return model

def reconstruction_loss(test_loader, FA_model, VAE_model):
    test_data_tensor = test_loader.dataset.dataset.X[test_loader.dataset.indices]
    test_data_np = test_loader.dataset.dataset.X[test_loader.dataset.indices].numpy()
    FA_latent = FA_model.transform(test_data_np)
    FA_recon = FA_model.inverse_transform(FA_latent)
    FA_recon_mse = ((test_data_np - FA_recon) ** 2).mean(axis=0).mean()

    # VAE_recon_mse = 0
    with torch.no_grad():
        VAE_output = VAE_model.forward(test_data_tensor, compute_loss=True)
        # VAE_latent = VAE_output.z_sample.numpy()
        VAE_latent = VAE_output.z_dist.mean.numpy()
        VAE_recon_mse = VAE_output.loss_recon.item()
        # for data in test_loader:
        #     out = VAE_model(data[0], compute_loss=True)
        #     VAE_recon_mse += out.loss_recon.item()
        # VAE_recon_mse /= len(test_loader)
    return FA_recon_mse, VAE_recon_mse, FA_latent, VAE_latent   

if __name__ == "__main__":
    # path = 'data/sim_10000_50_25_1_0.25/set_2/data.csv'
    # source = 'data/sim_10000_50_25_1_0.25/'
    # path = 'data/sim_10000_50_20_1_0/set_2/data.csv'
    # source = 'data/sim_10000_50_20_1_0'
    path = 'data2/sim_50000_100_20_0.25/set_1/data.csv'
    source = 'data2/sim_50000_100_20_0.25'
    paths = ['set_1']
    writer = SummaryWriter(f'runs/FA/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    train_loader, val_loader, test_loader, scaler = data_loader(path, batch_size=1024, target=True)
    fa_model = train_fa_model(train_loader, n_components=10, n_iter=5000, tol=1e-3)

    output = train_model(source, paths, learning_rate=1e-3, weight_decay=1e-2,num_epochs=50, latent_dim=10, density=1, beta=0, target=True, batch_size=1024, writer=writer)
    VAE_model = output['set_1']['model']
    VAE_model.eval()
    
    # FA MSE
    fa_recon_mse, VAE_recon_mse = reconstruction_loss(val_loader, fa_model, VAE_model)
    
    print(f'FA Reconstruction MSE: {fa_recon_mse}')
    print(f'VAE Reconstruction MSE: {VAE_recon_mse}')