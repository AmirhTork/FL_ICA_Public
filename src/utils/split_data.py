from utils.dataset_utils import NumpyDataset

def split_data(X, y, n_clients):
    n = len(X)
    per = n // n_clients
    datasets = []

    for i in range(n_clients):
        s = i * per
        e = (i+1)*per if i != n_clients - 1 else n
        datasets.append(NumpyDataset(X[s:e], y[s:e]))

    return datasets
 
