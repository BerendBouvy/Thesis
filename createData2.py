from DataSim2 import DataSim
import os
import numpy as np


def create_data(n_sets, n_samples=100, high_dim=10, latent_dim=9, std_A=1, non_linear_ratio=0.5, cross_ratio=0.5, sparsity=1, s2nr=0.1):
    random_seed = np.random.randint(0, 10000)  # Random seed for reproducibility
    folder_name = f"data2/sim_{n_samples}_{high_dim}_{latent_dim}_{non_linear_ratio}"
    if not os.path.exists(folder_name):
        print(f"Creating folder: {folder_name}")
        os.makedirs(folder_name)
    else:
        print(f"Folder {folder_name} already exists. Skipping creation.")
        
    for i in range(n_sets):
        if not os.path.exists(f"{folder_name}/set_{i+1}"):
            print(f"Creating set {i+1} in folder: {folder_name}")
            os.makedirs(f"{folder_name}/set_{i+1}")
        else:
            print(f"Set {i+1} already exists in folder: {folder_name}. Skipping creation.")

        sim = DataSim(n_samples=n_samples, latent_dim=latent_dim, high_dim=high_dim, std_A=std_A, random_seed=random_seed, non_linear_ratio=non_linear_ratio, cross_ratio=cross_ratio, sparsity=sparsity, s2nr=s2nr)
        sim.writeToFile(f"{folder_name}/set_{i+1}/data")
        random_seed += 1  # Increment the random seed for the next set

            
if __name__ == "__main__":
    # create_data(n_sets=25, n_samples=100, n_features=10, rank_A=9, std_A=1, non_linear_ratio=0.5)
    # create_data(n_sets=25, n_samples=10000, n_features=50, rank_A=25, std_A=1, non_linear_ratio=0.25)
    # create_data(n_sets=25, n_samples=10000, n_features=50, rank_A=20, std_A=1, non_linear_ratio=0)
    # create_data(n_sets=1, n_samples=50000, n_features=100, rank_A=50, std_A=1, non_linear_ratio=0, target=True)
    # create_data(n_sets=1, n_samples=50000, high_dim=100, latent_dim=50, std_A=1, non_linear_ratio=0.25, target=True)
    # create_data(n_sets=25, n_samples=50000, high_dim=100, latent_dim=20, std_A=1, non_linear_ratio=0.25)
    # create_data(n_sets=5, n_samples=10000, high_dim=100, latent_dim=20, std_A=1, non_linear_ratio=0.5, sparsity=1, s2nr=1)
    create_data(n_sets=5, n_samples=10000, latent_dim=20, high_dim=100, std_A=1, non_linear_ratio=0.4, cross_ratio=.4, sparsity=1, s2nr=.1)
    # create_data(n_sets=1, n_samples=10000, latent_dim=4, high_dim=12, std_A=1, non_linear_ratio=0.25, cross_ratio=.25, sparsity=1, s2nr=.1)
    print("Data sets created successfully.")
