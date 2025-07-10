import DataSim
import os
import numpy as np

def create_data(n_sets, n_samples=100, n_features=10, rank_A=9, std_A=1, non_linear_ratio=0.5):
    random_seed = np.random.randint(0, 10000)  # Random seed for reproducibility
    folder_name = f"data/sim_{n_samples}_{n_features}_{rank_A}_{std_A}_{non_linear_ratio}"
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
            
        sim = DataSim.DataSim(n_samples=n_samples, n_features=n_features, rank_A=rank_A, std_A=std_A, random_seed=random_seed, non_linear_ratio=non_linear_ratio)
        sim.writeToFile(f"{folder_name}/set_{i+1}/data")


if __name__ == "__main__":
    # create_data(n_sets=25, n_samples=100, n_features=10, rank_A=9, std_A=1, non_linear_ratio=0.5)
    create_data(n_sets=25, n_samples=10000, n_features=50, rank_A=25, std_A=1, non_linear_ratio=0.25)
    print("Data sets created successfully.")
