import numpy as np
import nonLinFunc
import DataSim
import matplotlib.pyplot as plt
from dataLoader import data_loader
import numpy as np
import torch

def test_data_sim():
    sim = DataSim.DataSim(n_samples=100, n_features=10, rank_A=9, std_A=1, random_seed=10, non_linear_ratio=0.5)
    sim.writeToFile("data/test_data_sim")

def test_polynomial():
    x = np.array([1, 2, 3])
    p = 3
    result = nonLinFunc.polynomial(x, p)
    expected = np.array([1, 8, 27])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
    
def test1():
    path = 'data\sim_10000_50_25_1_0.25\set_2\data.csv'
    train_loader, val_loader, test_loader, scaler = data_loader(path, batch_size=64)


if __name__ == "__main__":
    # test_data_sim()
    # test_polynomial()
    test1()
    print("All tests passed!")