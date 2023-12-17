import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define the sliding window function
def create_sliding_window_sequences(input_data, output_data, window_size):
    X = []
    y = []
    for i in range(window_size, len(input_data)):
        X.append(input_data[i-window_size:i])
        y.append(output_data[i])
    return np.array(X), np.array(y)

def create_sequences(X, y, time_steps=36):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Define the LSTM model
class CMEPredictorLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(CMEPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# Evaluate the model
# evaluate the model by predicting on the test set and calculating accuracy, precision, recall, etc.
