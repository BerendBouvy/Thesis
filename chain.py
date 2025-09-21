import os

import numpy as np
import pandas as pd
from createData2 import create_data
from train_model import train_model
# from train_model6 import train_model
from FA import train_fa_model, reconstruction_loss
# from FA6 import train_fa_model, reconstruction_loss
from Lin_regression2 import LinRegression
import time

def main(n_sets, n_samples, high_dim, latent_dim, epsilon_snr, std_A, non_linear_ratio, cross_ratio, 
            sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
            n_iter, tol, folder_name):
    folder = create_data(n_sets=n_sets, n_samples=n_samples, high_dim=high_dim, 
                latent_dim=latent_dim, epsilon_snr=epsilon_snr, std_A=std_A, non_linear_ratio=non_linear_ratio, 
                cross_ratio=cross_ratio, sparsity=sparsity, s2nr=s2nr, folder_name=folder_name)
    print(f"Data sets created successfully in folder: {folder}")
    # folder = 'data61/8_sim_2000_200_20_0.45_0.45_0.7_1'
    data_sets = os.listdir(folder)
    # data_sets.remove('formatted_scores.csv')
    output = train_model(source=folder, paths=data_sets, learning_rate=learning_rate, 
                            weight_decay=weight_decay, num_epochs=num_epochs, latent_dim=latent_dim, 
                            density=density, beta=beta, target=True, batch_size=batch_size, verbose=verbose)
    VAE_best = 0
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
        if scores['VAE_latent']['predicted R^2'] > scores['FA_latent']['predicted R^2'] and \
            scores['VAE_latent']['predicted R^2'] > scores['full']['predicted R^2']:
            print(f"VAE performed best for data set: {data_set}\n because {scores['VAE_latent']['predicted R^2']} > {scores['FA_latent']['predicted R^2']} and {scores['VAE_latent']['predicted R^2']} > {scores['full']['predicted R^2']}")
            VAE_best += 1
        
        
    all_scores = [pd.DataFrame(output[data_set]['Lin_regression']).T for data_set in output.keys()]
    combined_scores = pd.concat(all_scores, axis=0)
    summary_scores = combined_scores.groupby(combined_scores.index).aggregate(['mean', 'std'])
    means = summary_scores.xs('mean', axis=1, level=1)
    stds  = summary_scores.xs('std', axis=1, level=1)
    formatted_scores = means.round(4).astype(str) + " (" + stds.round(4).astype(str) + ")"
    formatted_scores.to_csv(os.path.join(folder, "formatted_scores.csv"))
    print(formatted_scores)
    print(f"VAE performed best in {VAE_best} out of {len(output.keys())} data sets.")

if __name__ == "__main__":
    
    
    
    # n_sets = 5
    # n_samples = 2000
    # high_dim = 200
    # latent_dim = [10, 20, 50, 80]
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = .4
    # cross_ratio = .4
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 250
    # density = 1
    # beta = 0# np.linspace(0, 1, num_epochs)
    # batch_size = 64
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data24"
    # start_time = time.time()
    # for latent in latent_dim:
    #     print(f"Running with latent_dim: {latent}")
    #     main(n_sets, n_samples, high_dim, latent, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
    #          sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #          n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # n_sets = 10
    # n_samples = 2000
    # high_dim = 200
    # latent_dim = [15, 20, 25, 30]
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = .45
    # cross_ratio = .45
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 150
    # density = 1
    # beta = 0
    # batch_size = 64
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data15"
    # start_time = time.time()
    # for latent in latent_dim:
    #     print(f"Running with latent_dim: {latent}")
    #     main(n_sets, n_samples, high_dim, latent, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
    #          sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #          n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    n_sets = 5
    n_samples = 2000
    high_dim = 200
    latent_dim = 20
    epsilon_snr = 1
    std_A = 10
    non_linear_ratio = .45
    cross_ratio = .45
    sparsity = .7
    s2nr = 1
    learning_rate = 1e-3
    weight_decay = 1e-4
    num_epochs = 150
    density = 1
    beta = [np.zeros(num_epochs),
            np.linspace(0, 1e-3, num_epochs),
            np.linspace(0, 1e-2, num_epochs),
            np.linspace(0, 1e-1, num_epochs)
           ]
    batch_size = 64
    verbose = False
    n_iter = 5000
    tol = 1e-3
    folder_name = "data13"
    start_time = time.time()
    for b in beta:
        print(f"Running with beta: {b}")
        main(n_sets, n_samples, high_dim, latent_dim, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
             sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, b, verbose,
             n_iter, tol, folder_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # n_sets = 10
    # n_samples = 2000
    # high_dim = 200
    # latent_dim = 20
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = [0, .125, .25, .375, .5] #[0, .125, .25, .375, .5]
    # cross_ratio = [0, .125, .25, .375, .5]
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 200
    # density = 1
    # beta = 0 # np.linspace(0, 1, num_epochs)
    # batch_size = 64
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data19"
    # start_time = time.time()
    # for i, j in zip(non_linear_ratio, cross_ratio):
    #     print(f"Running with non_linear_ratio: {i}, cross_ratio: {j}")
    #     main(n_sets, n_samples, high_dim, latent_dim, epsilon_snr, std_A, i, j,
    #          sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #          n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    
    # n_sets = 5
    # n_samples = 2000
    # high_dim = 200
    # latent_dim = 20
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = .3
    # cross_ratio = .3
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 300
    # density = 1
    # beta = 0 # np.linspace(0, 5e-1, num_epochs)
    # batch_size = 64
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data9"
    # start_time = time.time()
    # main(n_sets, n_samples, high_dim, latent_dim, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
    #      sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #      n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # n_sets = 5
    # n_samples = 10000
    # high_dim = 200
    # latent_dim = [10, 20, 50, 100]
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = 0.3
    # cross_ratio = 0.3
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 200
    # density = 2
    # beta = 0
    # batch_size = 256
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data8"
    # start_time = time.time()
    # for i in latent_dim:
    #     print(f"Running with latent_dim: {i}")
    #     main(n_sets, n_samples, high_dim, i, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
    #             sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #             n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    # n_sets = 5
    # n_samples = 10000
    # high_dim = 200
    # latent_dim = 20
    # epsilon_snr = 1
    # std_A = 10
    # non_linear_ratio = [0, .125, .25, .375, .5]
    # cross_ratio = [0, .125, .25, .375, .5]
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 250
    # density = 2
    # beta = 0
    # batch_size = 256
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data8"
    # start_time = time.time()
    # for i, j in zip(non_linear_ratio, cross_ratio):
    #     print(f"Running with non_linear_ratio: {i}, cross_ratio: {j}")
    #     main(n_sets, n_samples, high_dim, latent_dim, epsilon_snr, std_A, i, j,
    #             sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #             n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    
    # n_sets = 5
    # n_samples = 10000
    # # high_dim = [100, 200, 500, 1000]
    # # latent_dim = [10, 20, 50, 100]
    # high_dim = [1000]
    # latent_dim = [100]
    # epsilon_snr = 2
    # std_A = 10
    # non_linear_ratio = .3
    # cross_ratio = .3
    # sparsity = .7
    # s2nr = 1
    # learning_rate = 1e-3
    # weight_decay = 1e-4
    # num_epochs = 200
    # density = 2
    # beta = 0
    # batch_size = 256
    # verbose = False
    # n_iter = 5000
    # tol = 1e-3
    # folder_name = "data4"
    # start_time = time.time()
    # for i, j in zip(high_dim, latent_dim):
    #     print(f"Running with high_dim: {i}, latent_dim: {j}")
    #     main(n_sets, n_samples, i, j, epsilon_snr, std_A, non_linear_ratio, cross_ratio,
    #             sparsity, s2nr, learning_rate, weight_decay, num_epochs, batch_size, density, beta, verbose,
    #             n_iter, tol, folder_name)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("End time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))