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
NUM_EPOCHS = 1000  # Number of epochs of training for the ANN
ERR_THRESHOLD = 1e-6  # Maximum error change required to stop the training early
PATIENCE = 0  # Number of epochs with no improvement after which the training will be stopped
EARLY_STOPPING = [tf.keras.callbacks.EarlyStopping(monitor='loss',  # Early stopping callback function
                                                   min_delta=ERR_THRESHOLD,
                                                   patience=PATIENCE,
                                                   verbose=True)]

# Hyperparameters
NUM_PARAM = 4  # Number of parameters we are optimizing (# units, # layers, learning rate, learning momentum)
MAX_LAYERS = 2  # Maximum number of hidden layers
MAX_UNITS = 10  # Maximum number of units per hidden layer
LEARNING_RATE = np.linspace(0, 1, num=51)  # Learning rate
LEARNING_MOMENTUM = np.linspace(0, 1, num=51)  # Learning momentum


def build_ff_nn(input_dim, learning_rate, learning_momentum, num_layers, num_units):
    """
    Builds a feedforward neural network using non-stochastic gradient descent optimization with momentum.  A mean
    squared error function is used as a loss function.  Hidden layers use a sigmoidal activation function and
    output layers use a linear activation function.
    Args:
        input_dim (int): The dimension of an input vector to the network
        learning_rate (float): The learning rate (step size)
        learning_momentum (float): The learning momentum
        num_layers (int): The number of hidden layers (must be positive)
        num_units (int): The number of units in each hidden layer (must be positive)
    Returns:
        mlp (tf.keras.Sequential): The MLP regression model
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


def param_cv(lm, max_units, max_layers, learning_rate, train_data, train_targets, target_mean, target_std):
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
        target_mean (float np.ndarray): Mean of the targets used in normalization of the data set
        target_std (float np.ndarray): Standard deviation of the targets used in normalization of the data set
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

                    # Predict test fold data
                    test_fold_pred = mlp.predict(test_fold)

                    # Inverse normalize the test predictions and targets
                    test_fold_pred = target_std*test_fold_pred + target_mean
                    test_fold_targets = target_std*test_fold_targets + target_mean

                    # Calculate performance measures
                    (rmse, nmse, mase) = calc_performance(test_fold_targets, test_fold_pred)

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


def calc_performance(targets, pred_outputs):
    """
    Calculates the root mean squared error (RMSE), normalized mean squared error (NMSE), and mean absolute scaled error (MASE)

    Args:
        targets (float np.ndarray): The true targets for each input vector
        pred_outputs (pd.Series): The predicted outputs for each input vector
    Returns:
         results (float np.ndarray): Array of input parameters and their corresponding results
    """
    mse = np.sum((np.ravel(targets) - np.ravel(pred_outputs)) ** 2) / len(targets)

    rmse = np.sqrt(mse)

    std_targets = np.std(targets)
    nmse = mse / (std_targets ** 2)

    diff_targets = np.zeros(len(targets)-1)  # Differences between two adjacent targets
    for l in range(len(diff_targets)):
        diff_targets[l] = np.abs(targets.iloc[l + 1] - targets.iloc[l])
    mase = mse / (np.sum(diff_targets) / (len(targets) - 1))
    return (rmse, nmse, mase)


def training_curve(lr, lm, num_layers, num_units, train_data, train_targets, num_epochs):
    """
    Trains an RNN with the given parameters to observe how the mean squared error changes over time.

    Args:
        lr (float): Learning rate
        lm (float): Learning momentum
        num_layers (int): Number of hidden layers
        num_units (int): Number of units in each hidden layer
        train_data (pd.DataFrame): Training data
        train_targets (pd.Series): Training targets
        num_epochs (int): Number of epochs to train for
    Returns:
         mse_curve (float np.ndarray): Mean squared error as a function of epochs
    """
    # Build and train the MLP
    mlp = build_ff_nn(len(train_data.keys()), lr, lm, num_layers, num_units)
    history = mlp.fit(train_data.values,
                      train_targets,
                      epochs=num_epochs,
                      verbose=False)
    mse_curve = history.history['loss']
    tf.keras.backend.clear_session()  # Destroy current TF graph to prevent cluttering from old models
    return mse_curve


def plot_training_curves(lr, lm, num_layers, max_units, train_data, train_targets, num_epochs):
    """
    Trains multiple RNNs over a various number of hidden units and plots their training error curves.

    Args:
        lr (float): Learning rate
        lm (float): Learning momentum
        num_layers (int): Number of hidden layers
        max_units (int): Maximum number of units for each hidden layer
        train_data (pd.DataFrame): Training data
        train_targets (pd.Series): Training targets
        num_epochs (int): Number of epochs to train for
    Returns:
         mse_curve (float np.ndarray): Mean squared error as a function of epochs
    """
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
    matplotlib.rcParams.update({'font.size': 30})

    # Set random seed for testing consistency
    random.seed(0)
    tf.set_random_seed(0)

    #== GLASS ==#
    # Import delay embedded glass data set
    glass_data = pd.read_csv('glass_delay.csv', header=None)

    # Normalize glass data set
    glass_mean = glass_data.mean()
    glass_std = glass_data.std()
    glass_data = (glass_data - glass_mean) / glass_std

    # Separate target values from the features
    glass_targets = glass_data.pop(glass_data.shape[1]-1)

    # Split into training and test data sets using a chronologically ordered single-split
    glass_split = round(TRAIN_FRAC*glass_data.shape[0])
    glass_train_data = glass_data.iloc[:glass_split, :]
    glass_test_data = glass_data.drop(glass_train_data.index)
    glass_train_targets = glass_targets.iloc[glass_train_data.index]
    glass_test_targets = glass_targets.iloc[glass_test_data.index]

    # Perform k-fold cross validation for all parameter combinations, parallelizing over the learning momentum
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(param_cv,
                                   max_units=MAX_UNITS,
                                   max_layers=MAX_LAYERS,
                                   learning_rate=LEARNING_RATE,
                                   train_data=glass_train_data,
                                   train_targets=glass_train_targets,
                                   target_mean=glass_mean[len(glass_mean)-1],
                                   target_std=glass_std[len(glass_std)-1]),
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
                                             'NMSE Mean',
                                             'NMSE StD'])

    # Save glass results to a csv file
    glass_ff_results.to_csv('C:/Users/jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/glass_ff_cv_results.csv', index=False)

    rmse_ind = glass_ff_results['RMSE Mean'].idxmin(axis=1)
    mase_ind = glass_ff_results['MASE Mean'].idxmin(axis=1)
    nmse_ind = glass_ff_results['NMSE Mean'].idxmin(axis=1)

    # Find optimal combination of parameters
    perform_ind = np.array([rmse_ind, mase_ind, nmse_ind])
    temp = 0
    for i in range(len(perform_ind)):
        num_agree = np.sum(perform_ind == perform_ind[i])
        if (num_agree > temp):
            temp = num_agree
            opt_index = perform_ind[i]
            if (num_agree == len(perform_ind)):  # All measures are minimized with the same network configuration
                break

    glass_opt_units = int(glass_ff_results['Number of units'].iloc[opt_index])
    glass_opt_layers = int(glass_ff_results['Number of layers'].iloc[opt_index])
    glass_opt_lr = glass_ff_results['Learning rate'].iloc[opt_index]
    glass_opt_lm = glass_ff_results['Learning momentum'].iloc[opt_index]

    # Build and train the optimal FF NN with the optimal parameters
    opt_ff_nn = build_ff_nn(len(glass_train_data.keys()), glass_opt_lr, glass_opt_lm, glass_opt_layers, glass_opt_units)
    opt_ff_nn.fit(glass_train_data.values, glass_train_targets, epochs=NUM_EPOCHS)

    # Evaluate performance of the model
    glass_test_pred = opt_ff_nn.predict(glass_test_data.values)

    # Inverse normalize the test predictions and targets
    glass_test_pred = glass_std[len(glass_std)-1]*glass_test_pred + glass_mean[len(glass_mean)-1]
    glass_test_targets = glass_std[len(glass_std)-1]*glass_test_targets + glass_mean[len(glass_mean)-1]

    # Calculate performance measures for the out-of-sample test set
    (glass_rmse, glass_nmse, glass_mase) = calc_performance(glass_test_targets, glass_test_pred)
    glass_test_vector = np.concatenate((glass_opt_units, glass_opt_layers, glass_opt_lr, glass_opt_lm, glass_rmse, glass_nmse, glass_mase), axis=None)
    glass_test_results = pd.DataFrame(data=[glass_test_vector],
                                      columns=['num_units',
                                               'num_layers',
                                               'learning_rate',
                                               'learning_momentum',
                                               'RMSE',
                                               'NMSE',
                                               'MASE'])
    glass_test_results.to_csv('C:/Users/jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/glass_ff_test_results.csv', index=False)

    # Plot predicted outputs overlayed with the targets
    time = range(len(glass_test_pred))
    plt.figure(figsize=(18.5, 10.5))
    plt.plot(time, glass_test_pred, label='Predicted')
    plt.plot(time, glass_test_targets, label='Actual')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Mackey-Glass Test Results for FFNN')
    plt.legend()
    plt.savefig('C:/Users/Jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/glass_ff_test_results.svg')

    #== LASER ==#
    # Import delay embedded laser data set
    laser_data = pd.read_csv('laser_delay.csv', header=None)

    # Normalize laser data set
    laser_mean = laser_data.mean()
    laser_std = laser_data.std()
    laser_data = (laser_data - laser_mean) / laser_std

    # Separate target values from the features
    laser_targets = laser_data.pop(laser_data.shape[1]-1)

    # Split into training and test data sets using a chronologically ordered single-split
    laser_split = round(TRAIN_FRAC*laser_data.shape[0])
    laser_train_data = laser_data.iloc[:laser_split, :]
    laser_test_data = laser_data.drop(laser_train_data.index)
    laser_train_targets = laser_targets.iloc[laser_train_data.index]
    laser_test_targets = laser_targets.iloc[laser_test_data.index]


    # Perform k-fold cross validation for all parameter combinations, parallelizing over the learning momentum
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(param_cv,
                                   max_units=MAX_UNITS,
                                   max_layers=MAX_LAYERS,
                                   learning_rate=LEARNING_RATE,
                                   train_data=laser_train_data,
                                   train_targets=laser_train_targets,
                                   target_mean=laser_mean[len(laser_mean)-1],
                                   target_std=laser_std[len(laser_std)-1]),
                           LEARNING_MOMENTUM)

    # Convert array of results to a DataFrame
    laser_ff_results = pd.DataFrame(data=np.vstack(results),
                                    columns=['Number of units',
                                             'Number of layers',
                                             'Learning rate',
                                             'Learning momentum',
                                             'RMSE Mean',
                                             'RMSE StD',
                                             'MASE Mean',
                                             'MASE StD',
                                             'NMSE Mean',
                                             'NMSE StD'])

    # Save laser results to a csv file
    laser_ff_results.to_csv('C:/Users/jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/laser_ff_cv_results.csv', index=False)

    rmse_ind = laser_ff_results['RMSE Mean'].idxmin(axis=1)
    mase_ind = laser_ff_results['MASE Mean'].idxmin(axis=1)
    nmse_ind = laser_ff_results['NMSE Mean'].idxmin(axis=1)

    # Find optimal combination of parameters
    perform_ind = np.array([rmse_ind, mase_ind, nmse_ind])
    temp = 0
    for i in range(len(perform_ind)):
        num_agree = np.sum(perform_ind == perform_ind[i])
        if (num_agree > temp):
            temp = num_agree
            opt_index = perform_ind[i]
            if (num_agree == len(perform_ind)):  # All measures are minimized with the same network configuration
                break

    laser_opt_units = int(laser_ff_results['Number of units'].iloc[opt_index])
    laser_opt_layers = int(laser_ff_results['Number of layers'].iloc[opt_index])
    laser_opt_lr = laser_ff_results['Learning rate'].iloc[opt_index]
    laser_opt_lm = laser_ff_results['Learning momentum'].iloc[opt_index]

    # Build and train the optimal FF NN with the optimal parameters
    opt_ff_nn = build_ff_nn(len(laser_train_data.keys()), laser_opt_lr, laser_opt_lm, laser_opt_layers, laser_opt_units)
    opt_ff_nn.fit(laser_train_data.values, laser_train_targets, epochs=NUM_EPOCHS)

    # Evaluate performance of the model
    laser_test_pred = opt_ff_nn.predict(laser_test_data.values)

    # Inverse normalize the test predictions and targets
    laser_test_pred = laser_std[len(laser_std)-1]*laser_test_pred + laser_mean[len(laser_mean)-1]
    laser_test_targets = laser_std[len(laser_std)-1]*laser_test_targets + laser_mean[len(laser_mean)-1]

    # Calculate performance measures for the out-of-sample test set
    (laser_rmse, laser_nmse, laser_mase) = calc_performance(laser_test_targets, laser_test_pred)
    laser_test_vector = np.concatenate((laser_opt_units, laser_opt_layers, laser_opt_lr, laser_opt_lm, laser_rmse, laser_nmse, laser_mase), axis=None)
    laser_test_results = pd.DataFrame(data=[laser_test_vector],
                                      columns=['num_units',
                                               'num_layers',
                                               'learning_rate',
                                               'learning_momentum',
                                               'RMSE',
                                               'NMSE',
                                               'MASE'])
    laser_test_results.to_csv('C:/Users/jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/laser_ff_test_results.csv', index=False)

    # Plot predicted outputs overlayed with the targets
    time = range(len(laser_test_pred))
    plt.figure(figsize=(18.5, 10.5))
    plt.plot(time, laser_test_pred, label='Predicted')
    plt.plot(time, laser_test_targets, label='Actual')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.title('Santa Fe Laser Test Results for FFNN')
    plt.legend()
    plt.savefig('C:/Users/Jason/Dropbox/University/Grad School/Winter Term/ECE 626/Project 3/laser_ff_test_results.svg')
    plt.show()
    print("Complete\n")


if __name__ == '__main__':
    freeze_support()
    main()
