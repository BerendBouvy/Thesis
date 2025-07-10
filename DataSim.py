import numpy as np
import nonLinFunc
import csv


class DataSim:
    def __init__(self, n_samples=100, n_features=1, rank_A=1, std_A=1, random_seed=47, non_linear_ratio=0.5):
        """ Initialize the DataSim object.
        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features (rows and columns of A).
            std_A (float): Standard deviation of the entries in A.
            random_state (int): Seed for reproducibility.
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.rank_A = rank_A
        self.std_A = std_A
        self.non_linear_ratio = non_linear_ratio
        self.A = self.createA()
        self.cov_matrix = self.A @ self.A.T
        self.linear_data = self.createLinearData()
        self.non_linear_data, self.metadata = self.createNonLinearData()

    def createA(self):
        """ Create a random matrix A with specified standard deviation.
        Args:
            n_features (int): Number of features (rows and columns of A).
            random_state (int): Seed for reproducibility.
            std_A (float): Standard deviation of the entries in A.
        Returns:
            np.ndarray: Random matrix A of shape (n_features, n_features).
        """
        return self.rng.normal(size=(self.n_features, self.rank_A), scale=self.std_A)

    def createLinearData(self):
        """ Create linear data based on the random matrix A.
        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features (rows and columns of A).
            std_A (float): Standard deviation of the entries in A.
            random_state (int): Seed for reproducibility.
        Returns:
            np.ndarray: Linear data of shape (n_samples, n_features).
        """
        return self.rng.multivariate_normal(mean=np.zeros(self.n_features), cov=self.cov_matrix, size=self.n_samples)
    
    def createNonLinearData(self):
        
        metadata = {}
        
        list_of_functions = [
            nonLinFunc.polynomial,
            nonLinFunc.exp,
            nonLinFunc.log,
            nonLinFunc.smooth_abs,
            nonLinFunc.tanh,
            nonLinFunc.sigmoid,
            nonLinFunc.sin,
            nonLinFunc.cos
        ]
        name_of_functions = [
            "polynomial",
            "exp",
            "log",
            "smooth_abs",
            "tanh",
            "sigmoid",
            "sin",
            "cos"
        ]
        possible_params = [
            (0, 5),  # polynomial degree
            (-10, 10),  # exp coefficient
            (0.01, 1),  # log epsilon
            (0.01, 1),  # smooth_abs epsilon
            None,  # tanh has no parameters
            None,  # sigmoid has no parameters
            None,  # sin has no parameters
            None   # cos has no parameters
        ]
        num_non_linear = int(self.n_features * self.non_linear_ratio)
        non_linear_indices = self.rng.choice(self.n_features, num_non_linear, replace=False)
        non_linear_data = np.copy(self.linear_data)
        for i in non_linear_indices:
            func = self.rng.choice(list_of_functions)
            if func == nonLinFunc.polynomial:
                p = self.rng.integers(*possible_params[0])
                metadata[int(i)] = name_of_functions[list_of_functions.index(func)] + f" (degree {p})"
                non_linear_data[:, i] = func(non_linear_data[:, i], p)
            elif func == nonLinFunc.exp:
                c = self.rng.uniform(*possible_params[1])
                metadata[int(i)] = name_of_functions[list_of_functions.index(func)] + f" (coefficient {c})"
                non_linear_data[:, i] = func(non_linear_data[:, i], c)
            elif func == nonLinFunc.log:
                eps = self.rng.uniform(*possible_params[2])
                metadata[int(i)] = name_of_functions[list_of_functions.index(func)] + f" (epsilon {eps})"
                non_linear_data[:, i] = func(non_linear_data[:, i], eps)
            elif func == nonLinFunc.smooth_abs:
                eps = self.rng.uniform(*possible_params[3])
                metadata[int(i)] = name_of_functions[list_of_functions.index(func)] + f" (epsilon {eps})"
                non_linear_data[:, i] = func(non_linear_data[:, i], eps)
            else:
                non_linear_data[:, i] = func(non_linear_data[:, i])

        return non_linear_data, metadata
    
    def getData(self):
        return self.non_linear_data, self.metadata
    
    def getLinearData(self):
        return self.linear_data
    
    def getCovMatrix(self):
        return self.cov_matrix
    
    def getA(self):
        return self.A
    
    def getNFeatures(self):
        return self.n_features
    
    def getNSamples(self):
        return self.n_samples
    
    def getRankA(self):
        return self.rank_A
    
    def getStdA(self):
        return self.std_A
    
    def getRandomSeed(self):
        return self.random_seed
    
    def getNonLinearRatio(self):
        return self.non_linear_ratio
    
    def __str__(self):
        return (f"DataSim(n_samples={self.n_samples}, n_features={self.n_features}, "
                f"rank_A={self.rank_A}, std_A={self.std_A}, random_seed={self.random_seed}, "
                f"non_linear_ratio={self.non_linear_ratio})")
        
    def writeToFile(self, filename):
        """ Write the non-linear data and metadata to a file.
        Args:
            filename (str): The name of the file to write to.
        """
        # Save non-linear data to CSV
        np.savetxt(f"{filename}.csv", self.non_linear_data, delimiter=",", fmt="%.6f", header=",".join([f"Feature {i}" for i in range(self.n_features)]), comments="")
        
        # Save metadata to TXT
        with open(f"{filename}_metadata.txt", "w") as metafile:
            metafile.write("Name: "+ self.__str__() + "\n")
            metafile.write(f"seed: {self.random_seed}\n")
            metafile.write(f"cov_matrix:\n")
            for row in self.cov_matrix:
                metafile.write(", ".join(f"{val:.4f}" for val in row) + "\n")
            metafile.write(f"rank_A: {self.rank_A}\n")
            for key, value in self.metadata.items():
                metafile.write(f"Feature {key}: {value}\n")
                