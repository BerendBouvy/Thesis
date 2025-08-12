import numpy as np
import nonLinFunc
import csv


class DataSim:
    def __init__(self, n_samples=100, latent_dim=1, high_dim=1, std_A=1, random_seed=47, non_linear_ratio=0.5):
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
        self.latent_dim = latent_dim
        self.high_dim = high_dim
        self.std_A = std_A
        self.non_linear_ratio = non_linear_ratio
        self.latent_x, self.beta, self.epsilon, self.y = self.latentModel()
        self.A = self.createA()
        self.hd_x = self.latent_x @ self.A
        self.non_linear_data, self.metadata = self.createNonLinearData()

    def latentModel(self):
        latent_x = self.rng.normal(size=(self.n_samples, self.latent_dim), scale=1)
        beta = self.rng.normal(size=(self.latent_dim, 1), scale=1)
        epsilon = self.rng.normal(size=(self.n_samples, 1), scale=1)
        y = latent_x @ beta + epsilon
        return latent_x, beta, epsilon, y

    def createA(self):
        """
        Generates a random matrix A with normally distributed entries.
        Returns:
            numpy.ndarray: A matrix of shape (latent_dim, high_dim) where each entry is drawn 
            from a normal distribution with standard deviation `std_A`.
        Notes:
            - `self.rng` should be a NumPy random generator instance.
            - `self.latent_dim` and `self.high_dim` specify the dimensions of the matrix.
            - `self.std_A` specifies the standard deviation of the normal distribution.
        """
        
        return self.rng.normal(size=(self.latent_dim, self.high_dim), scale=self.std_A)
    
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
            (1, 5),  # polynomial degree
            (-2, 2),  # exp coefficient
            (0.01, 1),  # log epsilon
            (0.01, 1),  # smooth_abs epsilon
            None,  # tanh has no parameters
            None,  # sigmoid has no parameters
            None,  # sin has no parameters
            None   # cos has no parameters
        ]
        num_non_linear = int(self.latent_dim * self.non_linear_ratio)
        non_linear_indices = self.rng.choice(self.latent_dim, num_non_linear, replace=False)
        non_linear_data = np.copy(self.hd_x)
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
    
    def getA(self):
        return self.A
    
    def getNFeatures(self):
        return self.latent_dim
    
    def getNSamples(self):
        return self.n_samples
    
    def getRankA(self):
        return self.high_dim
    
    def getStdA(self):
        return self.std_A
    
    def getRandomSeed(self):
        return self.random_seed
    
    def getNonLinearRatio(self):
        return self.non_linear_ratio
    
    def __str__(self):
        return (f"DataSim(n_samples={self.n_samples}, latent_dim={self.latent_dim}, "
                f"high_dim={self.high_dim}, std_A={self.std_A}, random_seed={self.random_seed}, "
                f"non_linear_ratio={self.non_linear_ratio})")
        
    def writeToFile(self, filename):
        """ Write the non-linear data and metadata to a file.
        Args:
            filename (str): The name of the file to write to.
        """
        # Save non-linear data to CSV
        np.savetxt(f"{filename}.csv", np.column_stack((self.non_linear_data, self.y)), delimiter=",", fmt="%.6f", header=",".join([f"Feature {i}" for i in range(self.high_dim)]) + ",Target", comments="")
        np.savetxt(f"{filename}_latent.csv", np.column_stack((self.latent_x, self.y)), delimiter=",", fmt="%.6f", header=",".join([f"Latent Feature {i}" for i in range(self.latent_dim)]) + ",Target", comments="")
        # Save metadata to TXT
        with open(f"{filename}_metadata.txt", "w") as metafile:
            metafile.write("Name: "+ self.__str__() + "\n")
            metafile.write(f"seed: {self.random_seed}\n")
            metafile.write("Beta:\n")
            np.savetxt(metafile, self.beta.T, delimiter=",", fmt="%.6f", comments="")
            metafile.write("Metadata:\n")
            for key, value in self.metadata.items():
                metafile.write(f"{key}: {value}\n")
        
        np.savetxt(f"{filename}_A.csv", self.A, delimiter=",", fmt="%.6f", header=",".join([f"Feature {i}" for i in range(self.high_dim)]), comments="")
            