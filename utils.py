"""
Various utility functions.
"""

import torch
import argparse


def compute_multiclass_accuracy(output, target):
    preds = output.argmax(1).type_as(target)
    correct = preds.eq(target).double()
    correct = correct.sum()
    return correct / len(target)


def spectral_normalization(adj_matrix):
    """Normalize the adjacency matrix."""
    # normalize the adjacency matrix
    max_eig = torch.linalg.eigh(torch.tensor(adj_matrix))[0][-1]
    adj_matrix = adj_matrix / max_eig
    return adj_matrix.numpy()


def parse_boolean(value):
    """Parse boolean values passed as argument"""
    value = value.lower()
    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False
    return False

def parse_args():
    """ Parse arguments """
    parse = argparse.ArgumentParser()

    ## Run details
    parse.add_argument("--nEpochs", help="epochs", type=int, default=100)
    parse.add_argument("--batchSize", help="batch size", type=int, default=100000)
    parse.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parse.add_argument("--directed", help="directed or undirected topoNN", type=parse_boolean, default=True)
    parse.add_argument("--directed_dataset", help="directed or undirected dataset", type=parse_boolean, default=True)
    parse.add_argument("--upper_laplacian", help="whether to use upper laplacian or not", type=parse_boolean, default=False)
    parse.add_argument("--n_layers", help="topoNN layers", type=int, default=3)
    parse.add_argument("--n_hid_conv", help="hidden size of convolutional layers", type=int, default=64)
    parse.add_argument("--n_hid_mlp", help="hidden size of readout MLP", type=int, default=64)
    parse.add_argument("--conv_order", help="order of convolutional filters", type=int, default=1)
    parse.add_argument("--snr", help="SNR in dB for the dataset", type=int, default=40)
    parse.add_argument("--spike", help="Spike intensity for the dataset", type=float, default=1.0)
    parse.add_argument("--iterations", help="iterations for each experiment", type=int, default=1)
    parse.add_argument("--write", help="write output to csv file", type=parse_boolean, default=True)

    args = parse.parse_args()
    return args
