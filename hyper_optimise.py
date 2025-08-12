from train_model import train_model
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def main():
    source = "data2/sim_50000_100_20_0.25"
    paths = os.listdir(source)
    paths = paths[:1]  # For testing, use only the first path
    learning_rates = [1e-3]
    weight_decays = [1e-5]
    num_epochs = [100]
    latent_dims = [20]
    density = [1]
    beta = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    batch_size = [1024]
    target = True
    output = {}
    for lr in learning_rates:
        for wd in weight_decays:
            for ne in num_epochs:
                for ld in latent_dims:
                    for d in density:
                        for b in batch_size:
                            for beta_val in beta:
                                # Create a unique writer for each combination of parameters
                                writer = SummaryWriter(f'runs/11/vae_{lr}_{wd}_{ne}_{ld}_{d}_{b}_{beta_val}')
                                print(f"Training with lr={lr}, wd={wd}, ne={ne}, ld={ld}, d={d}, b={b}, beta={beta_val}")
                                output.update(train_model(source, paths, learning_rate=lr, weight_decay=wd,
                                                            num_epochs=ne, latent_dim=ld, density=d, beta=beta_val,
                                                            target=target, batch_size=b, writer=writer))


    print(output)
    print("Training completed.")
    
if __name__ == "__main__":
    main()