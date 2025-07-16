from csv import writer
from datetime import datetime
from modulefinder import test
import torch
from torch.utils.tensorboard import SummaryWriter
from VAE import VAE, VAEOutput
from dataLoader import data_loader
from train import train
from test_model import test


def train_model():

    # Load the dataset
    train_loader, val_loader, test_loader, scaler = data_loader(batch_size=64)

    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    latent_dim = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(input_dim=50, latent_dim=latent_dim, density=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    
    prev_updates = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer, device=device, batch_size=64)
        test(model, test_loader, prev_updates, writer=writer, device=device, latent_dim=latent_dim)

if __name__ == "__main__":
    train_model()
    