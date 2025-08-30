import numpy as np
import nonLinFunc
import nonLinCross
import csv


class DataSim:
    def __init__(self, n_samples=100, latent_dim=1, epsilon_snr=1, high_dim=1, std_A=1, random_seed=47, non_linear_ratio=0.25, cross_ratio=.25, sparsity=1, s2nr=.1):
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
        self.epsilon_snr = epsilon_snr
        self.std_A = std_A
        self.non_linear_ratio = non_linear_ratio
        self.cross_ratio = cross_ratio
        self.latent_x, self.beta, self.epsilon, self.y = self.latentModel()
        self.sparsity = sparsity
        self.s2nr = s2nr
        self.non_linear_features = int(self.high_dim * self.non_linear_ratio)
        self.cross_features = (int(self.high_dim * self.cross_ratio) + 1) // 2 *2

        self.non_linear_indices = np.arange(0, self.non_linear_features)
        self.indices_cross = np.arange(self.non_linear_features, self.non_linear_features + self.cross_features)

        self.A = self.createA()


        self.hd_x = self.latent_x @ self.A
        self.non_linear_data, self.metadata = self.createNonLinearData()
        self.non_linear_data_cross = self.cross()
        self.non_linear_data_noisy = self.addNoise(self.non_linear_data)

    def latentModel(self):
        latent_x = self.rng.normal(size=(self.n_samples, self.latent_dim), scale=1)
        # latent_x += self.rng.normal(size=latent_x.shape, scale=100)
        beta = self.rng.normal(size=(self.latent_dim, 1), scale=10)
        beta_norm2 = np.linalg.norm(beta, ord=2)
        epsilon_var = beta_norm2 / self.epsilon_snr
        epsilon = self.rng.normal(size=(self.n_samples, 1), scale=epsilon_var)
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
        non_lin_features = self.non_linear_features + self.cross_features
        latent_non_lin_features = np.ceil(self.latent_dim * non_lin_features / self.high_dim).astype(int)

        full_A = self.rng.normal(size=(self.latent_dim, self.high_dim), scale=self.std_A)
        sparce_A = np.copy(full_A)
        # Set some entries to zero based on sparsity
        mask = self.rng.uniform(size=full_A.shape) > self.sparsity
        sparce_A[mask] = 1e-10
        sparce_A[latent_non_lin_features:, :non_lin_features] = 0  # Ensure non-linear features are not connected to linear features
        sparce_A[:latent_non_lin_features, non_lin_features:] = 0  # Ensure linear features are not connected to non-linear features
        return sparce_A

    def createNonLinearData(self):
        
        metadata = {}
        
        list_of_functions = [
            nonLinFunc.polynomial,
            # nonLinFunc.exp,
            nonLinFunc.log,
            nonLinFunc.smooth_abs,
            # nonLinFunc.tanh,
            # nonLinFunc.sigmoid,
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
            (2, 5),  # polynomial degree
            (-2, 2),  # exp coefficient
            (0.01, 1),  # log epsilon
            (0.01, 1),  # smooth_abs epsilon
            None,  # tanh has no parameters
            None,  # sigmoid has no parameters
            None,  # sin has no parameters
            None   # cos has no parameters
        ]
        non_linear_data = np.copy(self.hd_x)
        for i in self.non_linear_indices:
            # Normalize the data before applying non-linear functions
            # non_linear_data[:, i] = (non_linear_data[:, i]  - np.mean(non_linear_data[:, i])) / np.std(non_linear_data[:, i])
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
                metadata[int(i)] = name_of_functions[list_of_functions.index(func)]
                non_linear_data[:, i] = func(non_linear_data[:, i])
                

        return non_linear_data, metadata
    
    def cross(self):
        
        #reshape to form pairs
        self.indices_cross = self.indices_cross.reshape(-1, 2)
        non_linear_data_cross = np.copy(self.non_linear_data)
        list_of_functions = [
            nonLinCross.polynomialprod,
            nonLinCross.polynomialsum,
            nonLinCross.exp,
            nonLinCross.log,
            nonLinCross.ratio1,
            nonLinCross.ratio2,
            nonLinCross.sin1,
            nonLinCross.sin2
        ]
        name_of_functions = [
            "polynomialprod",
            "polynomialsum",
            "exp",
            "log",
            "ratio1",
            "ratio2",
            "sin1",
            "sin2"
        ]
        possible_params = [
            (1, 4),  # polynomial degree for x and y
            (1, 4),  # polynomial degree for x and y
            (-2, 2),  # exp coefficient
            (0.01, 1),  # log epsilon
            (0.01, 1),  # ratio1 epsilon
            (0.01, 1),  # ratio2 epsilon
            None,  # sin1 has no parameters
            None   # sin2 has no parameters
        ]
        non_linear_data_cross = np.copy(self.non_linear_data)
        for i, (idx1, idx2) in enumerate(self.indices_cross):
            func = self.rng.choice(list_of_functions)
            # non_linear_data_cross[:, idx1] = (non_linear_data_cross[:, idx1] - np.mean(non_linear_data_cross[:, idx1])) / np.std(non_linear_data_cross[:, idx1])
            # non_linear_data_cross[:, idx2] = (non_linear_data_cross[:, idx2] - np.mean(non_linear_data_cross[:, idx2])) / np.std(non_linear_data_cross[:, idx2])
            if func == nonLinCross.polynomialprod:
                p1, p2 = self.rng.integers(*possible_params[0]), self.rng.integers(*possible_params[0])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (degree {p1}, {p2})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], p1, p2)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.polynomialsum:
                p1, p2 = self.rng.integers(*possible_params[1]), self.rng.integers(*possible_params[1])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (degree {p1}, {p2})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], p1, p2)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.exp:
                c = self.rng.uniform(*possible_params[2])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (coefficient {c})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], c)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.log:
                eps = self.rng.uniform(*possible_params[3])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (epsilon {eps})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], eps)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.ratio1:
                eps = self.rng.uniform(*possible_params[4])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (epsilon {eps})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], eps)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.ratio2:
                eps = self.rng.uniform(*possible_params[5])
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)] + f" (epsilon {eps})"
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2], eps)
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.sin1:
                non_linear_data_cross[:, idx1] = (non_linear_data_cross[:, idx1] - np.mean(non_linear_data_cross[:, idx1])) / np.std(non_linear_data_cross[:, idx1])
                non_linear_data_cross[:, idx2] = (non_linear_data_cross[:, idx2] - np.mean(non_linear_data_cross[:, idx2])) / np.std(non_linear_data_cross[:, idx2])    
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)]
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2])
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
            elif func == nonLinCross.sin2:
                self.metadata[f"{idx1}_{idx2}"] = name_of_functions[list_of_functions.index(func)]
                non_linear_data_cross[:, idx1] = func(non_linear_data_cross[:, idx1], non_linear_data_cross[:, idx2])
                non_linear_data_cross[:, idx2] = self.rng.normal(0, 1, size=non_linear_data_cross[:, idx2].shape)
        return non_linear_data_cross

    def addNoise(self, data):
        s2nr = self.s2nr
        std_dev = np.std(data, axis=0)
        data_std = np.copy(data) / std_dev
        noise = self.rng.normal(0, s2nr, size=data_std.shape)
        noisy_data = data_std + noise
        rescaled_data = noisy_data * std_dev
        return rescaled_data

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
        np.savetxt(f"{filename}.csv", np.column_stack((self.non_linear_data_noisy, self.y)), delimiter=",", fmt="%.12f", header=",".join([f"Feature {i}" for i in range(self.high_dim)]) + ",Target", comments="")
        np.savetxt(f"{filename}_latent.csv", np.column_stack((self.latent_x, self.y)), delimiter=",", fmt="%.12f", header=",".join([f"Latent Feature {i}" for i in range(self.latent_dim)]) + ",Target", comments="")
        # Save metadata to TXT
        with open(f"{filename}_metadata.txt", "w") as metafile:
            metafile.write("Name: "+ self.__str__() + "\n")
            metafile.write(f"seed: {self.random_seed}\n")
            metafile.write("Beta:\n")
            np.savetxt(metafile, self.beta.T, delimiter=",", fmt="%.12f", comments="")
            metafile.write("Metadata:\n")
            for key, value in self.metadata.items():
                metafile.write(f"{key}: {value}\n")
        
        np.savetxt(f"{filename}_A.csv", self.A, delimiter=",", fmt="%.12f", header=",".join([f"Feature {i}" for i in range(self.high_dim)]), comments="")
            