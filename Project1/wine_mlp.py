import tensorflow as tf
import numpy as np
import pandas as pd
import random
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial

# Define constants used throughout program
TRAIN_FRAC = 0.9  # Fraction of data going to the training set
NUM_FOLDS = 10  # Number of folds used in k-fold cross validation
NUM_EPOCHS = 1000  # Number of epochs of training for the MLP
NUM_PARAM = 4  # Number of parameters we are optimizing (# units, # layers, learning rate, learning momentum)
NUM_MEASURES = 8  # Number of performance measures calculated (loss, classification accuracy, TPR & FPR for all classes)
ERR_THRESHOLD = 1e-6  # Maximum error change required to stop the training early
PATIENCE = 0  # Number of epochs with no improvement after which the training will be stopped


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


def calc_performance(targets, pred_labels, num_classes):
    """
    Calculates the true positive and false positive rates for all classes

    Args:
        targets (int np.ndarray): The true class labels for each input vector
        pred_labels (pd.Series): The predicted class labels for each input vector
        num_classes (int): The total number of output classes
    Returns:
        tpr (float np.ndarray): The true positive rates for each class
        fpr(float np.ndarray): The false positive rates for each class
    """
    # Initialize arrays to store rates
    tpr = np.zeros(num_classes)
    fpr = np.zeros(num_classes)

    # Convert one-hot encoded targets back to categories
    targets = targets.idxmax(axis=1) - 1

    class_acc = np.sum(targets == pred_labels) / len(targets)

    # Iterate over all possible classes and calculate the rates
    for class_no in range(num_classes):
        num_tp = np.sum([i & j for i, j in zip(targets == class_no, pred_labels == class_no)])
        num_p = np.sum(targets == class_no)
        tpr[class_no] = float(num_tp / num_p)

        num_fp = np.sum([i & j for i, j in zip(targets != class_no, pred_labels == class_no)])
        num_n = len(targets) - num_p
        fpr[class_no] = float(num_fp / num_n)
    return class_acc, tpr, fpr

def param_cv(lm, max_units, max_layers, learning_rate, train_data, train_labels, fold_indices, num_outputs):
    """
    Performs k-fold cross validation and calculate performance measures for the input number of units with all possible
    combinations of the other parameters.  Defined in a function to parallelize over learning momentum

    Args:
        lm (float np.ndarray): Learning momentum values
        max_units (int): Maximum number of units to be used in each hidden layer
        max_layers (int): Maximum number of hidden layers
        learning_rate (float np.ndarray): Array of possible learning rate values
        train_data (pd.DataFrame): Training data
        train_labels (pd.Series): Training labels
        fold_indices (int np.ndarray): Array of indices for each fold (row k gives the indices for fold k)
        num_outputs (int): Number of outputs (classes)
    Returns:
         results (float np.ndarray): Array of input parameters and their corresponding results
    """
    # Initialize array to hold CV results for all combinations of parameters
    results = np.zeros((1, (NUM_PARAM + NUM_MEASURES*2)))

    # Define early stopping callback function
    early_stopping = [tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                       min_delta=ERR_THRESHOLD,
                                                       patience=PATIENCE,
                                                       verbose=True)]

    # Iterate over all possible combinations of hyper-parameters
    for i in range(max_layers):
        num_layers = i + 1
        for j in range(max_units):
            num_units = j + 1
            for k in range(len(learning_rate)):
                lr = learning_rate[k]

                # Initialize DataFrame to store performance measures for each test fold
                cv_measures = pd.DataFrame(columns=['Loss',
                                                    'Accuracy',
                                                    'TPR-0',
                                                    'TPR-1',
                                                    'TPR-2',
                                                    'FPR-0',
                                                    'FPR-1',
                                                    'FPR-2'])

                # Perform k-fold cross validation for the current set of hyper-parameters
                for fold_no in range(NUM_FOLDS):
                    # Separate into training folds and test fold
                    train_fold_indices = np.ravel([fold_indices[m, :] for m in range(NUM_FOLDS) if m not in [fold_no]])
                    test_fold_indices = fold_indices[fold_no, :]
                    train_folds = train_data.iloc[train_fold_indices]
                    train_fold_labels = train_labels.iloc[train_fold_indices]
                    test_fold = train_data.iloc[test_fold_indices]
                    test_fold_labels = train_labels.iloc[test_fold_indices]

                    # Build and train the MLP
                    mlp = build_mlp(len(train_data.keys()), num_outputs, lr, lm, num_layers, num_units)
                    mlp.fit(train_folds.values,
                            train_fold_labels,
                            epochs=NUM_EPOCHS,
                            verbose=False,
                            callbacks=early_stopping)

                    # Evaluate and classify test fold data
                    loss = mlp.evaluate(test_fold, test_fold_labels)[0]
                    test_fold_pred = np.argmax(mlp.predict(test_fold), axis=1)  # Choose class with highest probability

                    # Calculate performance measures
                    (class_acc, tpr, fpr) = calc_performance(test_fold_labels, test_fold_pred, num_outputs)
                    cv_measures = cv_measures.append(pd.Series(np.concatenate((loss, class_acc, tpr, fpr),axis=None),
                                                               index=cv_measures.columns),
                                                     ignore_index=True)
                    tf.keras.backend.clear_session()  # Destroy current TF graph to prevent cluttering from old models

                # Calculate desired performance measures
                cv_mean = cv_measures.mean()
                cv_std = cv_measures.std()
                cv_results = [val for pair in zip(cv_mean, cv_std) for val in pair]
                result_vector = np.concatenate((num_units, num_layers, lr, lm, cv_results), axis=None)
                results = np.vstack((results, result_vector))
                print("layers: {} ; units: {} ; lr: {} ; lm: {} completed".format(num_layers, num_units, lr, lm))
    results = np.delete(results, obj=0, axis=0)  # Remove first row that was used to initialize the array
    return results

def main():
    # Set random seed for testing consistency
    random.seed(0)
    tf.set_random_seed(0)

    var_names = ['Origin',  # Attribute names of wine data set
                 'Alcohol',
                 'Malic acid',
                 'Ash',
                 'Alcalinity of ash',
                 'Magnesium',
                 'Total phenols',
                 'Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity',
                 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']

    # Import wine data set
    wine_data = pd.read_csv('wine.data', header=None, names=var_names)

    # Get number of outputs
    num_outputs = max(wine_data['Origin'])

    # One hot encode the targets
    labels = pd.get_dummies(wine_data['Origin'])
    wine_data = wine_data.drop('Origin', axis=1)

    # Split into training and test data sets
    train_data = wine_data.sample(frac=TRAIN_FRAC, random_state=0)
    test_data = wine_data.drop(train_data.index)

    # Separate the target values from the features
    train_labels = labels.iloc[train_data.index]
    test_labels = labels.iloc[test_data.index]
    #print(test_labels)  # Print test labels to ensure that the classes are distributed ~evenly

    # Get mean and standard deviation of training data to perform normalization
    norm_mean = train_data.mean()
    norm_std = train_data.std()
    train_data = (train_data - norm_mean) / norm_std

    # Split training data into k folds
    num_samples_fold = int(len(train_data)/NUM_FOLDS)
    fold_indices = np.zeros((NUM_FOLDS, num_samples_fold))  # Initialize array to store indices for each fold
    train_indices = pd.DataFrame(np.arange(len(train_data)))  # All input vector numbers by index
    for k in range(NUM_FOLDS):
        fold = train_indices.sample(n=num_samples_fold, random_state=0)
        train_indices = train_indices.drop(fold.index)
        fold_indices[k, :] = np.ravel(fold)  # Each row holds the corresponding indices for one fold

    # Set hyper-parameters to iterate over
    max_layers = 2  # Maximum of 2 hidden layers
    max_units = 10  # Maximum of 10 hidden units per layer
    learning_rate = np.linspace(0, 1, num=51)  # 0 to 1; step size of 0.02
    learning_momentum = np.linspace(0, 1, num=51)  # 0 to 1; step size of 0.02

    # Perform k-fold cross validation for all parameter combinations, parallelizing over the learning momentum
    with Pool(cpu_count()) as pool:  # Number of processes = number of CPU cores
        results = pool.map(partial(param_cv,
                                   max_units=max_units,
                                   max_layers=max_layers,
                                   learning_rate=learning_rate,
                                   train_data=train_data,
                                   train_labels=train_labels,
                                   fold_indices=fold_indices,
                                   num_outputs=num_outputs),
                           learning_momentum)

    # Convert array of results to a DataFrame
    # Accuracy - number of correct classifications over the number of input vectors
    # TPR - true positive rate
    # FPR - false positive rate
    exp_results = pd.DataFrame(data=np.vstack(results),
                               columns=['Number of units',
                                        'Number of layers',
                                        'Learning rate',
                                        'Learning momentum',
                                        'Loss Mean',
                                        'Loss StD',
                                        'Accuracy Mean',
                                        'Accuracy StD',
                                        'TPR-0 Mean',
                                        'TPR-0 StD',
                                        'TPR-1 Mean',
                                        'TPR-1 StD',
                                        'TPR-2 Mean',
                                        'TPR-2 StD',
                                        'FPR-0 Mean',
                                        'FPR-0 StD',
                                        'FPR-1 Mean',
                                        'FPR-1 StD',
                                        'FPR-2 Mean',
                                        'FPR-2 StD'])

    # Save results to a csv file
    exp_results.to_csv('cv_mlp_results.csv', index=False)

    # Find optimal combination of parameters
    max_accuracy = exp_results['Accuracy Mean'].max()

    indices = [i for i, x in enumerate(exp_results['Accuracy Mean']) if x == max_accuracy]  # Get MLPs with best acc.
    best_mlps = exp_results.iloc[indices]
    opt_index = best_mlps['Loss Mean'].idxmin(axis=1)  # Argmin of loss among best MLPs
    opt_units = int(exp_results['Number of units'].iloc[opt_index])
    opt_layers = int(exp_results['Number of layers'].iloc[opt_index])
    opt_lr = exp_results['Learning rate'].iloc[opt_index]
    opt_lm = exp_results['Learning momentum'].iloc[opt_index]

    # Build and train MLP with the appropriate parameters
    opt_mlp = build_mlp(len(test_data.keys()), num_outputs, opt_lr, opt_lm, opt_layers, opt_units)
    opt_mlp.fit(train_data.values, train_labels, epochs=NUM_EPOCHS)

    # Normalize testing data
    test_data = (test_data - norm_mean) / norm_std

    # Evaluate performance of the model
    loss = opt_mlp.evaluate(test_data, test_labels)[0]
    test_pred = np.argmax(opt_mlp.predict(test_data.values), axis=1)

    # Calculate performance measures
    (class_acc, tpr, fpr) = calc_performance(test_labels, test_pred, num_outputs)
    test_vector = np.concatenate((opt_units, opt_layers, opt_lr, opt_lm, loss, class_acc, tpr, fpr), axis=None)
    test_results = pd.DataFrame(data=[test_vector],
                                columns=['num_units',
                                         'num_layers',
                                         'learning_rate',
                                         'learning_momentum',
                                         'Loss',
                                         'Accuracy',
                                         'TPR-0',
                                         'TPR-1',
                                         'TPR-2',
                                         'FPR-0',
                                         'FPR-1',
                                         'FPR-2'])
    test_results.to_csv('test_mlp_results.csv', index=False)
    print("Complete")


if __name__ == '__main__':
    freeze_support()
    main()