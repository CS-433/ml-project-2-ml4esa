import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_sliding_window_sequences(input_data, output_data, window_size):
    '''
    Create a n+1 dimensional array where each row is a sliding window view of array X with length window_size.
    Parameters:
    - input_data: nD array
    - output_data: 1D array
    - window_size: int, length of the sliding window
    Returns:
    - X: n+1 dimensional array
    - y: 1D array
    '''
    X = []
    y = []
    for i in range(window_size, len(input_data)):
        X.append(input_data[i - window_size:i])
        y.append(output_data[i])
    return np.array(X), np.array(y)

def create_sequences(X, y, time_steps=36):
    '''
    Create time series input for the LSTMs
    Parameters:
    - input_data: nD array
    - output_data: 1D array
    - time_steps: int, length of the sliding window
    Returns:
    - X: n+1 dimensional array
    - y: 1D array
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def create_sequences_without_nan(X, y, time_steps=36):
    '''
    For debugging purposes
    '''
    Xs, ys = create_sequences(X, y, time_steps=time_steps)
    time_steps = Xs.shape[1]
    nb_features = Xs.shape[2]
    Xs = Xs[~np.isnan(Xs).any(axis=-1)]
    Xs = Xs.reshape(-1, time_steps, nb_features)
    ys = ys[~np.isnan(ys)]
    return Xs, ys

def calculate_metrics(y_true, y_pred):
    '''
    For a given model, calculates the accuracy, precision, recall, and F1 score
    Input:
    - y_true: true labels
    - y_pred: predicted labels
    '''
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# LSTM model
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

# Interpretability

def plot_series(optimized_input, opti_type='max'):
    '''
    Input:
    - optimized_input: input that maximises a class logit
    - opti_type: 'max' or 'min'
    Output:
    - plot of the timeseries corresponding to the optimized input
    '''
    # Reshape the optimized input to remove the batch dimension if necessary
    if len(optimized_input.shape) > 2:
        optimized_input = optimized_input.reshape((optimized_input.shape[1], optimized_input.shape[2]))

    # Extract the x, y, and z coordinates
    x_coordinates = optimized_input[:, 0]
    y_coordinates = optimized_input[:, 1]
    z_coordinates = optimized_input[:, 2]

    # Generate a time axis based on the number of time steps
    time_steps = range(len(x_coordinates))

    # Create the plot
    plt.figure(figsize=(14, 6))

    # Plot each of the coordinates over time
    plt.plot(time_steps, x_coordinates, label='X Coordinate')
    plt.plot(time_steps, y_coordinates, label='Y Coordinate')
    plt.plot(time_steps, z_coordinates, label='Z Coordinate')

    # Add title and labels
    plt.title(f'Input {opti_type}imally activating the CME predictor model')
    plt.xlabel('Time Steps (by 5 minutes)')
    plt.ylabel('Magnetic Field Coordinate Values')
    plt.legend()

    # Show the plot
    plt.show()

def add_noise(data: np.ndarray, std: float = 0.1):
    '''
    Add noise to create new intances of the minority class
    Input:
    - data: nD array
    - std: standard deviation of the noise
    Output:
    - augmented_data: nD array with the added noise
    '''
    assert data.ndim == 3
    noise_data = np.random.normal(0, std, data.shape)
    augmented_data = data + noise_data
    return augmented_data

def augment_data(data: np.ndarray, labels: np.ndarray, label_to_augment: int = 1, factor: int = None, std: float = 0.1):
    '''
    Data augmentation: create new instances for the minority class by adding noise to existing instances
    Input:
    - data: nD array
    - labels: 1D array
    - label_to_augment: label of the minority class
    - factor: number of new instances to create
    - std: standard deviation of the noise
    '''
    assert data.ndim == 3
    assert labels.ndim == 1

    # Get the indices of the data to augment
    indices = np.where(labels == label_to_augment)
    data_to_augment = data[indices]

    if factor is None:
        factor = (len(data) - len(data_to_augment)) // len(data_to_augment)

    # Create the data to augment by repeating the data
    factored = [data_to_augment for _ in range(factor)]
    factored_data_to_augment = np.concatenate(factored, axis=0)
    noisy_data = add_noise(factored_data_to_augment, std=std)

    # Augment the data
    X_augmented = np.concatenate([data, noisy_data], axis=0)
    y_augmented = np.concatenate([labels, np.ones(len(data_to_augment) * factor)], axis=0)
    return X_augmented, y_augmented
