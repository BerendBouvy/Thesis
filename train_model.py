from csv import writer
from datetime import datetime
from modulefinder import test
import torch
from torch.utils.tensorboard import SummaryWriter
# from VAE2 import VAE, VAEOutput
from VAE import VAE, VAEOutput
from dataLoader import data_loader
from train import train
from test_model import test
import os





def train_model(source: str, paths: list[str], learning_rate: float, weight_decay: float, num_epochs: int, latent_dim: int, density: int, beta: float, target: bool, batch_size: int, writer: SummaryWriter = None):
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
    output = {}
    for path in paths:
        
        # Load the dataset
        train_loader, val_loader, test_loader, scaler = data_loader(os.path.join(source, path, 'data.csv'), batch_size=batch_size, target=target)
        input_dim = train_loader.dataset.dataset.X.shape[1]
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_dim=input_dim, latent_dim=latent_dim, density=density, beta=beta).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        
        prev_updates = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer, device=device, batch_size=batch_size)
            test(model, test_loader, prev_updates, writer=writer, device=device, latent_dim=latent_dim)
        output[path] = {
            'model': model,
            'scaler': scaler,
            'dataloaders': [train_loader, val_loader, test_loader]
        }
    return output

if __name__ == "__main__":
    source = "data2/sim_50000_100_50_0.25"
    paths = os.listdir(source)
    paths = paths[:1]  # For testing, use only the first path
    learning_rates = 1e-3
    weight_decays = 1e-5
    num_epochs = 100
    latent_dims = 20
    density = 1
    beta = 1e-3
    batch_size = 1024
    target = True
    writer = SummaryWriter(f'runs/9/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    output = train_model(source, paths, learning_rates, weight_decays, num_epochs, latent_dims, density=density, beta=beta, batch_size=batch_size, target=target, writer=writer)
    print("Training completed.")
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
