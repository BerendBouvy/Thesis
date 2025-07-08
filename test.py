import numpy as np
import nonLinFunc
import DataSim

def test_data_sim():
    sim = DataSim.DataSim(n_samples=100, n_features=30, std_A=10, random_seed=42, non_linear_ratio=0.5)

def test_polynomial():
    x = np.array([1, 2, 3])
    p = 3
    result = nonLinFunc.polynomial(x, p)
    expected = np.array([1, 8, 27])
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"
    
if __name__ == "__main__":
    test_data_sim()
    # test_polynomial()
    print("All tests passed!")