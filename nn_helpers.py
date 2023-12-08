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

def sliding_window_iterator(df, window_size=8, overlap=7):
    for i in range(0, len(df) - window_size + 1, overlap):
        yield df[i:i + window_size]

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

def neural_net(X, y, window_size, epochs=10, batch_size=32, with_class_weights=False):
    X1_series = create_series(X, window_size) # each line is a series of length window_size
    y1 = y[window_size-1:].astype('float32') # the label is the last element of the series

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
    return test_metrics

def neural_net_comp(X, y, window_sizes, epochs=10, batch_size=32, with_class_weights=False):
    '''
    Compares the performance of the neural network with different window sizes
    for each window, we get info on : loss, accuracy, precision, recall, f1 score
    '''
    metrics_array = []
    for window_size in window_sizes:
        test_metrics = neural_net(X, y, window_size, with_class_weights)
        metrics_array.append(test_metrics)
    return metrics_array

def plot_metrics(metrics_array, window_sizes):
    '''
    Plots the metrics for each window size
    '''
    metrics_df = pd.DataFrame(metrics_array, columns=['loss', 'accuracy', 'precision', 'recall', 'f1 score'])
    metrics_df['window size'] = window_sizes
    
    # Plot for Loss
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='loss', data=metrics_df, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Loss')
    plt.show()
    
    # Plot for Accuracy
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='accuracy', data=metrics_df, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Accuracy')
    plt.show()
    
    # Plot for Precision, Recall, and F1 Score
    metrics_df_melted = pd.melt(metrics_df, id_vars='window size', value_vars=['precision', 'recall', 'f1 score'])
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    sns.lineplot(x='window size', y='value', hue='variable', data=metrics_df_melted, marker='o')
    plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.title('Precision, Recall, and F1 Score')
    plt.show()

def plot_improvement(test_metrics_unbalanced, test_metrics_balanced):
    '''
    Plots the improvement of the model with class weights
    '''
def plot_histogram(data, title):
    sns.set_style('whitegrid')
    sns.set_palette('Set2')
    plt.hist(data, bins=10)
    plt.title(title)
    plt.show()

# Plot Loss Histogram
plot_histogram(metrics_df['loss'], 'Loss')

# Plot Accuracy Histogram
plot_histogram(metrics_df['accuracy'], 'Accuracy')

# Plot Precision, Recall, and F1 Score Histogram
plot_histogram(metrics_df['precision'], 'Precision')
plot_histogram(metrics_df['recall'], 'Recall')
plot_histogram(metrics_df['f1 score'], 'F1 Score')
