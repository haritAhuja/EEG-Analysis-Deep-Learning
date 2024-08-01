import numpy as np
from sklearn.preprocessing import StandardScaler

def randomize(dataset, labels, seed=123):
    np.random.seed(seed)
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def standardize_data(train_data, test_data):
    scalers = {}
    for i in range(train_data.shape[1]):
        scalers[i] = StandardScaler()
        train_data[:, i, :] = scalers[i].fit_transform(train_data[:, i, :])

    for i in range(test_data.shape[1]):
        test_data[:, i, :] = scalers[i].transform(test_data[:, i, :])

    return train_data, test_data