from train_model import train_model
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch 

def test():
    source = 'data2/sim_50000_100_20_0.25'
    paths = ['set_1']
    writer = SummaryWriter(f'runs/beta2/vae_8')
    num_epochs = 100
    beta = np.linspace(0, 1, num_epochs)
    # beta = np.zeros(num_epochs)
    output = train_model(source, paths, learning_rate=1e-3, 
                            weight_decay=1e-2, num_epochs=num_epochs, 
                            latent_dim=5, density=1, beta=beta, 
                            target=True, batch_size=1024, writer=writer)
    print(output['set_1']['model'])
    
if __name__ == "__main__":
    
    test()
    print("Test completed.")