from csv import writer
from datetime import datetime
from modulefinder import test
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# select the VAE version
# from VAE import VAE, VAEOutput
# from VAE2 import VAE, VAEOutput
# from VAE3 import VAE, VAEOutput
from VAE4 import VAE, VAEOutput
# from VAE5 import VAE, VAEOutput
from dataLoader import data_loader
from train import train
from test_model import test
import os
import copy




def train_model(source: str, paths: list[str], learning_rate: float, weight_decay: float, num_epochs: int, latent_dim: int, density: int, beta, target: bool, batch_size: int, writer: SummaryWriter = None, verbose: bool = True):
    """    Train the VAE model on the given dataset.
    Args:
        source (str): Path to the dataset directory.
        path (list[str]): List of paths to the dataset files.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        latent_dim (int): Dimensionality of the latent space.
    Returns:
        model (VAE): Trained VAE model.
        scaler (Normalizer): Normalizer used for scaling the dataset.
        dataloaders (list[DataLoader]): List of DataLoaders for training, validation, and testing.
    """
    early_stopping = True
    
    if type(beta) is float or type(beta) is int:
        beta = [beta] * num_epochs
        
    output = {}
    for path in paths:
        print(f"Training path: {path}")
        writer = SummaryWriter(f'runs/{source}vae_{path}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        # Load the dataset
        train_loader, val_loader, test_loader, scaler = data_loader(os.path.join(source, path, 'data.csv'), batch_size=batch_size, target=target)
        input_dim = train_loader.dataset.dataset.X.shape[1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_dim=input_dim, latent_dim=latent_dim, density=density, beta=beta).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_model = model
        best_loss = float('inf')
        best_epoch = 0

        prev_updates = 0
        for epoch in range(num_epochs):
            model.set_beta(beta[epoch])
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, beta: {model.beta}')
            prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer, device=device, batch_size=batch_size, verbose=verbose)
            test_loss, _, _ = test(model, val_loader, prev_updates, writer=writer, device=device, latent_dim=latent_dim, verbose=verbose)
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        if early_stopping:
            model.load_state_dict(best_model)
            test_loss, test_recon_loss, test_kl_loss = test(model, test_loader, prev_updates, writer=None, device=device, latent_dim=latent_dim, verbose=verbose)

            output[path] = {
                'model': model,
                'test_score': {
                    'loss': test_loss,
                    'recon_loss': test_recon_loss,
                    'kl_loss': test_kl_loss
                },
                'best_epoch': best_epoch,
                'scaler': scaler,
                'dataloaders': [train_loader, val_loader, test_loader]
            }
        else:
            test_loss, test_recon_loss, test_kl_loss = test(model, test_loader, prev_updates, writer=None, device=device, latent_dim=latent_dim, verbose=verbose)

            output[path] = {
                'model': model,
                'test_score': {
                    'loss': test_loss,
                    'recon_loss': test_recon_loss,
                    'kl_loss': test_kl_loss
                },
                'best_epoch': best_epoch,
                'scaler': scaler,
                'dataloaders': [train_loader, val_loader, test_loader]
            }
        
    return output

if __name__ == "__main__":
    source = "data3/sim_10000_1000_50_0.25_0.25_1_0.1"
    paths = os.listdir(source)
    paths = [paths[0]]  # For testing, use only the first path
    learning_rates = 1e-3
    weight_decays = 1e-5
    num_epochs = 300
    latent_dims = 20
    density = 1
    beta = 0 # 1e-3
    batch_size = 1024
    target = True
    verbose = True
    writer = SummaryWriter(f'runs/12/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output = train_model(source, paths, learning_rates, weight_decays, num_epochs, latent_dims, density=density, beta=beta, batch_size=batch_size, target=target, writer=writer, verbose=verbose)
    print(output[paths[0]]['model'])
    print(f"Test Loss: {output[paths[0]]['test_score']['loss']:.4f}, Recon Loss: {output[paths[0]]['test_score']['recon_loss']:.4f}, KL Loss: {output[paths[0]]['test_score']['kl_loss']:.4f}")
    print("Training completed.")
    
    
# if __name__ == "__main__":
#     source = "data2/sim_50000_100_50_0.25"
#     paths = os.listdir(source)
#     paths = paths[:1]  # For testing, use only the first path
#     learning_rates = 1e-3
#     weight_decays = 1e-5
#     num_epochs = 100
#     latent_dims = 20
#     density = 1
#     beta = 1e-3
#     batch_size = 1024
#     target = True
#     writer = SummaryWriter(f'runs/9/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
#     output = train_model(source, paths, learning_rates, weight_decays, num_epochs, latent_dims, density=density, beta=beta, batch_size=batch_size, target=target, writer=writer)
#     print("Training completed.")
# if __name__ == "__main__":
#     # source = 'data/sim_10000_50_25_1_0.25'
#     # paths = os.listdir(source)
#     # learning_rates = 1e-3
#     # weight_decays = 1e-2
#     # num_epochs = 100
#     # latent_dims = 16
#     # density = 1
#     # beta = 0.0
#     # target = False
#     # train_model(source, paths, learning_rates, weight_decays, num_epochs, latent_dims, density=density, beta=beta, target=target)
#     source = "data/sim_50000_100_50_1_0.25/set_1"
#     # source = "data/sim_50000_100_50_1_0/set_1"
#     paths = [""]
#     learning_rates = 1e-3
#     weight_decays = 1e-5
#     num_epochs = 100
#     latent_dims = 20
#     density = 1
#     beta = 1e-2
#     batch_size = 1024
#     target = True
#     train_model(source, paths, learning_rates, weight_decays, num_epochs, latent_dims, density=density, beta=beta, batch_size=batch_size, target=target)
