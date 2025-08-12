import DataSim
import os
import numpy as np

def merge2files(file1, file2, output_file):
    """
    Merges two files into one and add target values.
    """
    data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    data2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    # add target values
    target1 = np.zeros((data1.shape[0], 1))
    target2 = np.ones((data2.shape[0], 1))

    merged_data = np.hstack((data1, target1))
    merged_data = np.vstack((merged_data, np.hstack((data2, target2))))
    merged_data = merged_data.astype(np.float32)

    np.savetxt(output_file, merged_data, delimiter=",", fmt="%.6f", header=",".join([f"Feature {i}" for i in range(data1.shape[1])])+", Target", comments="")

def create_data(n_sets, n_samples=100, n_features=10, rank_A=9, std_A=1, non_linear_ratio=0.5, target=False):
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
        if not target:
            sim = DataSim.DataSim(n_samples=n_samples, n_features=n_features, rank_A=rank_A, std_A=std_A, random_seed=random_seed, non_linear_ratio=non_linear_ratio)
            sim.writeToFile(f"{folder_name}/set_{i+1}/data")
        else:
            sim1 = DataSim.DataSim(n_samples=n_samples//2, n_features=n_features, rank_A=rank_A, std_A=std_A, random_seed=random_seed, non_linear_ratio=non_linear_ratio)
            sim2 = DataSim.DataSim(n_samples=n_samples//2, n_features=n_features, rank_A=rank_A, std_A=std_A, random_seed=random_seed+1, non_linear_ratio=non_linear_ratio)
            sim1_name = f"{folder_name}/set_{i+1}/data_1"
            sim2_name = f"{folder_name}/set_{i+1}/data_2"
            sim1.writeToFile(sim1_name)
            sim2.writeToFile(sim2_name)
            merge2files(sim1_name + ".csv", sim2_name + ".csv", f"{folder_name}/set_{i+1}/data.csv")
            
if __name__ == "__main__":
    # create_data(n_sets=25, n_samples=100, n_features=10, rank_A=9, std_A=1, non_linear_ratio=0.5)
    # create_data(n_sets=25, n_samples=10000, n_features=50, rank_A=25, std_A=1, non_linear_ratio=0.25)
    # create_data(n_sets=25, n_samples=10000, n_features=50, rank_A=20, std_A=1, non_linear_ratio=0)
    # create_data(n_sets=1, n_samples=50000, n_features=100, rank_A=50, std_A=1, non_linear_ratio=0, target=True)
    create_data(n_sets=1, n_samples=50000, n_features=100, rank_A=50, std_A=1, non_linear_ratio=0.25, target=True)

    print("Data sets created successfully.")
