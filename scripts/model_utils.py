import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder


class NYCTaxiExampleDataset(torch.utils.data.Dataset):
    """
    Training data object for our nyc taxi data
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.X_train = X_train
        self.y_train = y_train
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.X = np.array(self._one_hot_X().toarray())
        self.y = np.array(self.y_train.values)
        self.X_enc_shape = self.X.shape[-1]
        print(f"encoded shape is {self.X_enc_shape}")
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
        
    def _one_hot_X(self):
        return self.one_hot_encoder.fit_transform(self.X_train)

class MLP(nn.Module):
    """
    Multilayer Perceptron for regression

    model structure:
        Sequential(
    (0): Linear(in_features=encoded_shape, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=1, bias=True))
    """

    def __init__(self, encoded_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoded_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
    
    def forward(self, x):
        return self.layers(x)
