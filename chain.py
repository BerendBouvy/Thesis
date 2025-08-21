import os

import numpy as np
import pandas as pd
from createData2 import create_data
from train_model import train_model
from FA import train_fa_model, reconstruction_loss
from Lin_regression2 import LinRegression

def main(n_sets, n_samples, high_dim, latent_dim, epsilon_var, std_A, non_linear_ratio, cross_ratio, 
            sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
            n_iter, tol):
    # folder = 'data3/sim_10000_1000_50_0.25_0.25_1_0.1/'
    folder = create_data(n_sets=n_sets, n_samples=n_samples, high_dim=high_dim, 
                latent_dim=latent_dim, epsilon_var=epsilon_var, std_A=std_A, non_linear_ratio=non_linear_ratio, 
                cross_ratio=cross_ratio, sparsity=sparsity, s2nr=s2nr)
    print(f"Data sets created successfully in folder: {folder}")
    # latent_dim += latent_dim
    data_sets = os.listdir(folder)
    output = train_model(source=folder, paths=data_sets, learning_rate=learning_rate, 
                            weight_decay=weight_decay, num_epochs=num_epochs, latent_dim=latent_dim, 
                            density=density, beta=beta, target=True, batch_size=batch_size, verbose=verbose)
    for data_set in output.keys():
        print(f"Processing data set: {data_set}")
        path_full = os.path.join(folder, data_set, 'data.csv')
        path_latent = os.path.join(folder, data_set, 'data_latent.csv')
        data_full = np.genfromtxt(path_full, delimiter=',', skip_header=1)
        true_latent = np.genfromtxt(path_latent, delimiter=',', skip_header=1)
        
        train_loader = output[data_set]['dataloaders'][0]
        test_loader = output[data_set]['dataloaders'][2]

        data_full_test_x = data_full[test_loader.dataset.indices, :-1]
        true_latent_test_x = true_latent[test_loader.dataset.indices,:-1]
        
        y_train = true_latent[train_loader.dataset.indices, -1]
        y_test = true_latent[test_loader.dataset.indices,-1]
        
        VAE_model = output[data_set]['model']
        VAE_model.eval()
        
        FA_model = train_fa_model(train_loader, n_components=latent_dim, n_iter=n_iter, tol=tol)
        output[data_set]['FA_model'] = FA_model
        
        _, _, FA_latent_train, VAE_latent_train = reconstruction_loss(train_loader, FA_model, VAE_model)
        FA_recon_mse, VAE_recon_mse, FA_latent, VAE_latent = reconstruction_loss(test_loader, FA_model, VAE_model)

        output[data_set]['FA_recon_mse'] = FA_recon_mse
        output[data_set]['VAE_recon_mse'] = VAE_recon_mse
        output[data_set]['FA_latent'] = FA_latent
        output[data_set]['VAE_latent'] = VAE_latent
        print(f"FA Reconstruction MSE: {FA_recon_mse}")
        print(f"VAE Reconstruction MSE: {VAE_recon_mse}")

        X_dict_train = {
            'full': data_full[train_loader.dataset.indices, :-1],
            'True_latent': true_latent[train_loader.dataset.indices, :-1],
            'FA_latent': FA_latent_train,
            'VAE_latent': VAE_latent_train
        }

        X_dict_test = {
            'full': data_full_test_x,
            'True_latent': true_latent_test_x,
            'FA_latent': FA_latent,
            'VAE_latent': VAE_latent
        }

        lr = LinRegression(X_dict_train, X_dict_test, y_train, y_test, intercept=True)

        scores = lr.get_scores()

        output[data_set]['Lin_regression'] = scores
        df = pd.DataFrame(scores).T
        print(df)

if __name__ == "__main__":
    n_sets = 1
    n_samples = 10000
    high_dim = 200
    latent_dim = 30
    epsilon_var = 1
    std_A = 10
    non_linear_ratio = 0
    cross_ratio = .75
    sparsity = .5
    s2nr = 1
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 200
    density = 2
    beta = 0
    batch_size = 256
    verbose = False
    n_iter = 5000
    tol = 1e-3
    main(n_sets, n_samples, high_dim, latent_dim, epsilon_var, std_A, non_linear_ratio, cross_ratio,
            sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
            n_iter, tol)