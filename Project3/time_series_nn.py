import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial

#== CONSTANTS ==#

# Validation
TRAIN_FRAC = 2/3  # Fraction of data going to the training set
NUM_FOLDS = 5  # Number of folds used in k-fold cross validation
NUM_MEASURES = 3  # Number of performance measures calculated (RMSE, MASE, NMSE)

# Neural network
NUM_EPOCHS = 10000  # Number of epochs of training for the ANN
ERR_THRESHOLD = 1e-6  # Maximum error change required to stop the training early
PATIENCE = 0  # Number of epochs with no improvement after which the training will be stopped
EARLY_STOPPING = [tf.keras.callbacks.EarlyStopping(monitor='loss',  # Early stopping callback function
                                                   min_delta=ERR_THRESHOLD,
                                                   patience=PATIENCE,
                                                   verbose=True)]

# Hyperparameters
NUM_PARAM = 4  # Number of parameters we are optimizing (# units, # layers, learning rate, learning momentum)
MAX_LAYERS = 1  # Maximum number of hidden layers
MAX_UNITS = 8  # Maximum number of units per hidden layer
#LEARNING_RATE = np.linspace(0, 1, num=51)  # Learning rate
#LEARNING_MOMENTUM = np.linspace(0, 1, num=51)  # Learning momentum
LEARNING_RATE = np.array([0.1])
LEARNING_MOMENTUM = np.array([0.9])


def build_ff_nn(input_dim, learning_rate, learning_momentum, num_layers, num_units):
    """
    Builds an feedforward neural network using non-stochastic gradient descent optimization with momentum.  A mean
    squared error function is used as a loss function.  Hidden layers use a sigmoidal activation function and
    output layers use a softmax activation function.
    Args:
        input_dim (int): The dimension of an input vector to the network
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
    mlp.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

    mlp.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=learning_momentum),
                loss='mse',
                metrics=['mse'])
    return mlp


def param_cv(lm, max_units, max_layers, learning_rate, train_data, train_targets):
    """
    Performs k-fold cross validation and calculates performance measures for the input number of units with all possible
    combinations of the other parameters.  Defined in a function to parallelize over learning momentum

    Args:
        lm (float np.ndarray): Learning momentum values
        max_units (int): Maximum number of units to be used in each hidden layer
        max_layers (int): Maximum number of hidden layers
        learning_rate (float np.ndarray): Array of possible learning rate values
        train_data (pd.DataFrame): Training data
        train_targets (pd.Series): Training targets
    Returns:
         results (float np.ndarray): Array of input parameters and their corresponding results
    """
    # Initialize array to hold CV results for all combinations of parameters
    results = np.zeros((1, (NUM_PARAM + NUM_MEASURES*2)))  # Multiply by 2 to account for mean + standard deviation

    # Initialize data frame that holds the possible indices for the training data
    fold_indices = pd.DataFrame(np.arange(train_data.shape[0]))

    # Size of each fold that is added onto the existing training fold
    fold_size = int(np.floor(train_data.shape[0]/(NUM_FOLDS+1)))  # Add 1 to number of folds due to rolling basis for CV
    leftover = train_data.shape[0] % (NUM_FOLDS+1)  # Leftover number of points from not evenly dividing into NUM_FOLDS

    # Iterate over all possible combinations of hyperparameters
    for i in range(max_layers):
        num_layers = i + 1  # Current number of layers
        for j in range(max_units):
            num_units = j + 1  # Current number of units
            for k in range(len(learning_rate)):
                lr = learning_rate[k]  # Current learning rate

                # Initialize DataFrame to store performance measures for each test fold
                cv_measures = pd.DataFrame(columns=['RMSE',
                                                    'MASE',
                                                    'NMSE'])

                # Perform k-fold cross validation for the current set of hyperparameters
                for fold in range(1, NUM_FOLDS+1):
                    # Get current training and test fold indices
                    train_indices = np.ravel(fold_indices.iloc[:(fold*fold_size+leftover)])
                    test_indices = np.ravel(fold_indices.iloc[(fold*fold_size+leftover):((fold+1)*fold_size+leftover)])

                    # Separate inputs and labels into training and test folds
                    train_fold = train_data.iloc[train_indices]
                    train_fold_targets = train_targets.iloc[train_indices]

                    test_fold = train_data.iloc[test_indices]
                    test_fold_targets = train_targets.iloc[test_indices]

                    # Build and train the MLP
                    mlp = build_ff_nn(len(train_data.keys()), lr, lm, num_layers, num_units)
                    mlp.fit(train_fold.values,
                            train_fold_targets,
                            epochs=NUM_EPOCHS,
                            verbose=False,
                            callbacks=EARLY_STOPPING)

                    # Evaluate and predict test fold data
                    mse = mlp.evaluate(test_fold, test_fold_targets)[0]
                    test_fold_pred = mlp.predict(test_fold)

                    # Calculate performance measures
                    rmse = np.sqrt(mse)
                    
                    std_targets = np.std(test_fold_targets)
                    nmse = mse / (std_targets ** 2)

                    diff_targets = np.zeros((1, len(test_fold)-1))  # Differences between two adjacent targets
                    for l in range(len(diff_targets)):
                        diff_targets[l] = np.abs(test_fold_targets.iloc[l+1] - test_fold_targets.iloc[l])
                    mase = mse / (np.sum(diff_targets)/(len(test_fold)-1))

                    # Add them to the results data frame
                    cv_measures = cv_measures.append(pd.Series(np.concatenate((rmse, nmse, mase), axis=None), index=cv_measures.columns), ignore_index=True)

                    tf.keras.backend.clear_session()  # Destroy current TF graph to prevent cluttering from old models

                # Calculate mean and standard deviation of performance measures from cross-validation
                cv_mean = cv_measures.mean()
                cv_std = cv_measures.std()
                cv_results = [val for pair in zip(cv_mean, cv_std) for val in pair]
                results_vector = np.concatenate((num_units, num_layers, lr, lm, cv_results), axis=None)
                results = np.vstack((results, results_vector))

                # Print progress reports
                print("layers: {} ; units: {} ; lr: {} ; lm: {} completed".format(num_layers, num_units, lr, lm))
    results = np.delete(results, obj=0, axis=0)  # Remove first row that was used to initialize the array
    return results


def training_curve(lr, lm, num_layers, num_units, train_data, train_targets, num_epochs):
    # Build and train the MLP
    mlp = build_ff_nn(len(train_data.keys()), lr, lm, num_layers, num_units)
    history = mlp.fit(train_data.values,
                      train_targets,
                      epochs=num_epochs,
                      verbose=False)
    tf.keras.backend.clear_session()  # Destroy current TF graph to prevent cluttering from old models
    return history.history['loss']


def plot_training_curves(lr, lm, num_layers, max_units, train_data, train_targets, num_epochs):
    matplotlib.rcParams.update({'font.size': 30})

    # Get training curves for constant learning rate, learning momentum, number of layers, but variable number of units
    for i in range(max_units):
        num_units = i + 1
        mse_curve = training_curve(lr, lm, num_layers, num_units, train_data, train_targets, num_epochs)
        plt.plot(range(num_epochs), mse_curve, label='{} unit'.format(num_units))
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title(r'Training Error Curves for $\eta$=0.1 and $\alpha$=0.9')
    plt.legend()
    plt.show()


def main():
    # Set random seed for testing consistency
    random.seed(0)
    tf.set_random_seed(0)

    # Import delay embedded glass and laser data sets
    glass_data = pd.read_csv('glass_delay.csv', header=None)
    laser_data = pd.read_csv('laser_delay.csv', header=None)

    # Separate target values from the features
    glass_targets = glass_data.pop(glass_data.shape[1]-1)
    laser_targets = laser_data.pop(laser_data.shape[1]-1)

    # Split into training and test data sets using a chronologically ordered single-split
    glass_split = round(TRAIN_FRAC*glass_data.shape[0])
    glass_train_data = glass_data.iloc[:glass_split, :]
    glass_test_data = glass_data.drop(glass_train_data.index)
    glass_train_targets = glass_targets.iloc[glass_train_data.index]
    glass_test_targets = glass_targets.iloc[glass_test_data.index]

    #plot_training_curves(0.9, 0.1, 1, 8, glass_train_data, glass_train_targets, 10000)

    laser_split = round(TRAIN_FRAC*laser_data.shape[0])
    laser_train_data = laser_data.iloc[:laser_split, :]
    laser_test_data = laser_data.drop(laser_train_data.index)
    laser_train_targets = laser_targets.iloc[laser_train_data.index]
    laser_test_targets = laser_targets.iloc[laser_test_data.index]

    plot_training_curves(0.9, 0.1, 1, 8, laser_train_data, laser_train_targets, 10000)

    # Perform k-fold cross validation for all parameter combinations, parallelizing over the learning momentum
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(param_cv,
                                   max_units=MAX_UNITS,
                                   max_layers=MAX_LAYERS,
                                   learning_rate=LEARNING_RATE,
                                   train_data=glass_train_data,
                                   train_targets=glass_train_targets),
                           LEARNING_MOMENTUM)

    # Convert array of results to a DataFrame
    glass_ff_results = pd.DataFrame(data=np.vstack(results),
                                    columns=['Number of units',
                                             'Number of layers',
                                             'Learning rate',
                                             'Learning momentum',
                                             'RMSE Mean',
                                             'RMSE StD',
                                             'MASE Mean',
                                             'MASE StD',
                                             'NRMSE Mean',
                                             'NRMSE StD'])

    # Save results to a csv file
    glass_ff_results.to_csv('C:/Users/jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/glass_ff_cv_results.csv', index=False)

    print("Complete\n")


if __name__ == '__main__':
    freeze_support()
    main()
