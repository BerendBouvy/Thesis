import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from Normalizer import Normalizer

# Custom Dataset class
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.iloc[:, :].values, dtype=torch.float32) 
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def data_loader(batch_size=64):
    full_dataset = CSVDataset('data/sim_10000_50_25_1_0.25/set_1/data.csv')

    N = len(full_dataset)
    train_size = int(0.8 * N)
    val_size = (N - train_size) // 2
    test_size = N - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))

    scaler = Normalizer(train_dataset.dataset.X)
    train_dataset.dataset.X = scaler.normalize(train_dataset.dataset.X)
    val_dataset.dataset.X = scaler.normalize(val_dataset.dataset.X)
    test_dataset.dataset.X = scaler.normalize(test_dataset.dataset.X)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scaler

