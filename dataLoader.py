import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from Normalizer import Normalizer
from sklearn.model_selection import train_test_split

# Custom Dataset class
class CSVDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.iloc[:, :].values, dtype=torch.float32) 
        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def data_loader(path, batch_size=64):
    full_dataset = CSVDataset(path)

    N = len(full_dataset)
    train_size = int(0.8 * N)
    val_size = (N - train_size) // 2
    test_size = N - train_size - val_size
    
    indices = list(range(N))
    train_indices, temp_indices = train_test_split(indices, train_size=train_size, shuffle=True, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, shuffle=True, random_state=42)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # scaler = Normalizer(train_dataset.dataset.X)
    scaler = Normalizer(train_dataset.dataset.X[train_dataset.indices])
    train_dataset.dataset.X = scaler.normalize(train_dataset.dataset.X)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scaler

