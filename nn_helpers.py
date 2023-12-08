# data processing
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn for preprocessing
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# tensorflow for model building
import tensorflow as tf
from tensorflow.keras import layers, models

def print_metrics(model_name, y_true, y_pred):
    print(f'Performance of {model_name}:')
    print(f'Accuracy: {round(skm.accuracy_score(y_true, y_pred), 2)}')
    print(f'Precision: {round(skm.precision_score(y_true, y_pred), 2)}')
    print(f'Recall: {round(skm.recall_score(y_true, y_pred), 2)}')
    print(f'F1 score: {round(skm.f1_score(y_true, y_pred), 2)}')

def augmentate(x, y, n):
    '''
    Over representation of 1s (minority class)
    '''
    x_1 = x[y == 1]
    y_1 = y[y == 1]
    x_0 = x[y == -1]
    y_0 = y[y == -1]

    # Augmentating by n the minority class samples
    x_1_augmented = np.repeat(x_1, n, axis=0)
    y_1_augmented = np.repeat(y_1, n)

    # Combine original majority class and augmented minority class
    new_x = np.vstack((x_0, x_1_augmented))
    new_y = np.concatenate((y_0, y_1_augmented))

    # Shuffle
    idx = np.random.permutation(new_x.shape[0])
    new_x = new_x[idx]
    new_y = new_y[idx]

    return new_x, new_y

def create_series(X, n):
    """
    Create a 2D array where each row is a sliding window view of array X with length n.

    Parameters:
    - X: 1D array
    - n: int, length of the sliding window

    Returns:
    - 2D array
    """
    num_windows = len(X) - n + 1
    series = np.zeros((num_windows, n), dtype=X.dtype)

    for i in range(num_windows):
        series[i, :] = X[i:i + n]

    return series

def create_series_dim(X, n):
    """
    Create a 2D array where each row is a sliding window view of array X with length m*n.

    Parameters:
    - X: m-dimensional array
    - n: int, length of the sliding window

    Returns:
    - 2D array
    """
    num_windows = len(X) - n + 1
    series = np.zeros((num_windows, n, X.shape[1]), dtype=X.dtype)

    for i in range(num_windows):
        series[i, :] = X[i:i + n]

    # Flatten the series
    series = series.reshape(num_windows, -1)

    return series

def neural_net(X, y, window_size, epochs=10, batch_size=32, with_class_weights=False, augmentation = 0):
    X1_series = create_series(X, window_size) # each line is a series of length window_size
    y1 = y[window_size-1:].astype('float32') # the label is the last element of the series

    if augmentation > 0:
        X1_series, y1 = augmentate(X1_series, y1, augmentation)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X1_series, y1, test_size=0.2, random_state=42)
    
    # define the neural network
    model_1 = models.Sequential()
    model_1.add(layers.Dense(64, activation='relu', input_shape=(window_size,)))
    model_1.add(layers.Dense(32, activation='relu'))
    model_1.add(layers.Dense(1, activation='sigmoid'))

    # compile the model and fit it to the training data
    model_1.compile(optimizer='adam', loss='binary_crossentropy', 
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])
    
    if not with_class_weights : #default
        model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    else : # mitigate class imbalance with class weights
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights)
    
    # Evaluate the model on the test set
    test_metrics = model_1.evaluate(X_test, y_test)
    test_metrics[4] = test_metrics[4][0] # F1 score is an array of one elemet
    return test_metrics, model_1

def neural_net_comp(X, y, window_sizes, epochs=10, batch_size=32, dim=False, with_class_weights=False):
    '''
    Compares the performance of the neural network with different window sizes
    for each window, we get info on : loss, accuracy, precision, recall, f1 score
    '''
    metrics_array = []
    for window_size in window_sizes:
        if dim:
            test_metrics, _ = neural_net_dim(X, y, window_size, epochs, batch_size, with_class_weights)
        else:
            test_metrics, _ = neural_net(X, y, window_size, with_class_weights)
        metrics_array.append(test_metrics)
    return metrics_array

def plot_metrics(metrics_array, window_sizes):
    '''
    Plots the metrics for each window size
    '''
    metrics_df = pd.DataFrame(metrics_array, columns=['loss', 'accuracy', 'precision', 'recall', 'f1 score'])
    metrics_df['window size'] = window_sizes
    
    # Plot for Precision, Recall, and F1 Score
    metrics_df_melted = pd.melt(metrics_df, id_vars='window size', value_vars=['precision', 'recall'])
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='value', hue='variable', data=metrics_df_melted, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Precision and Recall')
    plt.show()

    # Plot for F1 score
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='f1 score', data=metrics_df, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('F1 score')
    plt.show()

    # Plot for Accuracy
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='accuracy', data=metrics_df, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Accuracy')
    plt.show()

def print_improvement(test_metrics_1, test_metrics_2, improvement_name):
    '''
    Prints the improvement of the model with 'improvement_name'
    '''
    print(f'Performance of the model with {improvement_name}:')
    print(f'Loss: {round(test_metrics_1[0], 2)}')
    print(f'Accuracy: {round(test_metrics_1[1], 2)}')
    print(f'Precision: {round(test_metrics_1[2], 2)}')
    print(f'Recall: {round(test_metrics_1[3], 2)}')
    print(f'F1 score: {round(test_metrics_1[4], 2)}')
    print(f'\nPerformance difference of the model with {improvement_name}:')
    print(f'Loss: {round(test_metrics_1[0] - test_metrics_2[0], 2)}')
    print(f'Accuracy: {round(test_metrics_1[1] - test_metrics_2[1], 2)}')
    print(f'Precision: {round(test_metrics_1[2] - test_metrics_2[2], 2)}')
    print(f'Recall: {round(test_metrics_1[3] - test_metrics_2[3], 2)}')
    print(f'F1 score: {round(test_metrics_1[4] - test_metrics_2[4], 2)}')

def neural_net_dim(X, y, window_size, epochs=10, batch_size=32, with_class_weights=False, augmentation = 0):
    X1_series = create_series_dim(X, window_size) # each line is a series of length window_size*number of features
    y1 = y[window_size-1:].astype('float32') # the label is the last element of the series

    if augmentation > 0:
        X1_series, y1 = augmentate(X1_series, y1, augmentation)

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X1_series, y1, test_size=0.2, random_state=42)
    
    # define the neural network
    model_1 = models.Sequential()
    model_1.add(layers.Dense(128, activation='relu', input_shape=(window_size*X.shape[1],)))
    model_1.add(layers.Dense(64, activation='relu'))
    model_1.add(layers.Dense(32, activation='relu'))
    model_1.add(layers.Dense(1, activation='sigmoid'))

    # compile the model and fit it to the training data
    model_1.compile(optimizer='adam', loss='binary_crossentropy', 
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.F1Score()])
    
    if not with_class_weights : #default
        model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    else : # mitigate class imbalance with class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        model_1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weights)
    
    # Evaluate the model on the test set
    test_metrics = model_1.evaluate(X_test, y_test)
    test_metrics[4] = test_metrics[4][0] # F1 score is an array of one elemet
    return test_metrics, model_1