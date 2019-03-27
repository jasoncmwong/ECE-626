import tensorflow as tf
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool, freeze_support, cpu_count

#== CONSTANTS ==#
TRAIN_FRAC = 2/3
NUM_FOLDS = 10


def build_mlp(input_dim, num_outputs, learning_rate, learning_momentum, num_layers, num_units):
    """
    Builds an MLP classifier using Stochastic Gradient Descent optimization with a categorical cross entropy loss
    function under the desired input parameters.  Hidden layers use a sigmoidal activation function and output layers
    use a softmax activation function.
    Args:
        input_dim (int): The dimension of an input vector to the network
        num_outputs (int): The number of output classes
        learning_rate (float): The learning rate (step size)
        learning_momentum (float): The learning momentum
        num_layers (int): The number of hidden layers (must be positive)
        num_units (int): The number of units in each hidden layer (must be positive)
    Returns:
        mlp (tf.keras.Sequential): The MLP classifier model
    """
    mlp = tf.keras.Sequential()

    # Initialize MLP with one hidden layer
    mlp.add(tf.keras.layers.Dense(num_units, activation=tf.math.sigmoid, input_shape=[input_dim]))

    # Add more hidden layers
    for i in range(num_layers-1):
        mlp.add(tf.keras.layers.Dense(num_units, activation=tf.math.sigmoid))

    # Add output layer
    mlp.add(tf.keras.layers.Dense(num_outputs, activation=tf.nn.softmax))

    mlp.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=learning_momentum),
                loss='mse',
                metrics=['mse'])
    return mlp


def param_cv(lm, max_units, max_layers, learning_rate, train_data, train_labels):
    """
    Performs k-fold cross validation and calculates performance measures for the input number of units with all possible
    combinations of the other parameters.  Defined in a function to parallelize over learning momentum
    :param lm:
    :param max_units:
    :param max_layers:
    :param learning_rate:
    :param train_data:
    :param train_labels:
    :return:
    """
    # Initialize data frame that holds the possible indices for the training data
    fold_indices = pd.DataFrame(np.arange(train_data.shape[0]))

    # Size of each fold that is added onto the existing training fold
    fold_size = int(np.floor(train_data.shape[0]/(NUM_FOLDS+1)))  # Add 1 to number of folds due to rolling basis
    leftover = train_data.shape[0] % (NUM_FOLDS+1)  # Leftover number of points from not evenly dividing into NUM_FOLDS

    for k in range(1, NUM_FOLDS+1):
        train_indices = fold_indices.iloc[:(k*fold_size+leftover)]
        test_indices = fold_indices.iloc[(k*fold_size+leftover):((k+1)*fold_size+leftover)]


def main():
    # Set random seed for testing consistency
    random.seed(0)
    tf.set_random_seed(0)

    # Import delay embedded glass and laser data sets
    glass_data = pd.read_csv('glass_delay.csv', header=None)
    laser_data = pd.read_csv('laser_delay.csv', header=None)

    # Separate target values from the features
    glass_labels = glass_data.pop(glass_data.shape[1]-1)
    laser_labels = laser_data.pop(laser_data.shape[1]-1)

    # Split into training and test data sets using a chronologically ordered single-split
    glass_split = round(TRAIN_FRAC*glass_data.shape[0])
    glass_train_data = glass_data.iloc[:glass_split, :]
    glass_test_data = glass_data.drop(glass_train_data.index)
    glass_train_labels = glass_labels.iloc[glass_train_data.index]
    glass_test_labels = glass_labels.iloc[glass_test_data.index]

    laser_split = round(TRAIN_FRAC*laser_data.shape[0])
    laser_train_data = laser_data.iloc[:laser_split, :]
    laser_test_data = laser_data.drop(laser_train_data.index)
    laser_train_labels = laser_labels.iloc[laser_train_data.index]
    laser_test_labels = laser_labels.iloc[laser_test_data.index]


    print("Complete\n")


if __name__ == '__main__':
    freeze_support()
    main()