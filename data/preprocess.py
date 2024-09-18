"""
Preprocess the synthetic dataset to obtain the adjacency and incidence
matrixes needed by the DirSNN and SNN.
"""

import sys 
sys.path.append('../')

from compute_adj import compute_lower_adj, compute_lower_adj_undirected, compute_incidences
import torch
from data_generation import CellDataset
from utils import parse_args

args = parse_args()

directed_dataset = args.directed_dataset
directed = args.directed
snr = args.snr

if directed_dataset:
    dset = torch.load(f"synth/cell_sbm_dataset_directed_{snr}.pt")

    inc_0_1, adj_0_up = compute_incidences(dset.G, directed_dataset=True)

    n_samples = len(dset.samples)
    x_1 = torch.zeros([n_samples, dset.samples[0][0].shape[0]]) 
    y = []
    for i, sample in enumerate(dset.samples):
        x_1[i] = sample[0].squeeze()
        y.append(sample[1])
    y = torch.tensor(y)

    if directed:
        adj_low_100_1, adj_low_101_1, adj_low_110_1, adj_low_111_1 = compute_lower_adj(dset.G)

        data = (x_1, adj_low_100_1, adj_low_101_1, adj_low_110_1, adj_low_111_1)
        torch.save(data, f"synth/data_dirmodel_dirdata_{snr}.pt")
        torch.save((inc_0_1, adj_0_up, x_1), f"synth/inc_dirdata_{snr}.pt")
        torch.save(y, f"synth/y_dirdata_{snr}.pt")

    else:
        adj_1_down = compute_lower_adj_undirected(dset.G)
        data = (x_1, adj_1_down)
        torch.save(data, f"synth/data_undirmodel_dirdata_{snr}.pt")
        torch.save((inc_0_1, adj_0_up, x_1), f"synth/inc_dirdata_{snr}.pt")
        torch.save(y, f"synth/y_dirdata_{snr}.pt")

else:
    
    dset = torch.load(f"synth/cell_sbm_dataset_undirected_{snr}.pt")
    inc_0_1, adj_0_up = compute_incidences(dset.G, directed_dataset=False)

    adj_1_down = compute_lower_adj_undirected(dset.G)
    n_samples = len(dset.samples)
    x_1 = torch.zeros([n_samples, dset.samples[0][0].shape[0]]) 
    y = []
    for i, sample in enumerate(dset.samples):
        x_1[i] = sample[0].squeeze()
        y.append(sample[1])
    y = torch.tensor(y)


    if directed:
        adj_low_100_1, adj_low_101_1, adj_low_110_1, adj_low_111_1 = adj_1_down, adj_1_down, adj_1_down, adj_1_down
        data = (x_1, adj_low_100_1, adj_low_101_1, adj_low_110_1, adj_low_111_1)
        torch.save(data, f"synth/data_dirmodel_undirdata_{snr}.pt")
        torch.save((inc_0_1, adj_0_up, x_1), f"synth/inc_undirdata_{snr}.pt")
        torch.save(y, f"synth/y_undirdata_{snr}.pt")

    else:
        data = (x_1, adj_1_down)
        torch.save(data, f"synth/data_undirmodel_undirdata_{snr}.pt")
        torch.save((inc_0_1, adj_0_up, x_1), f"synth/inc_undirdata_{snr}.pt")
        torch.save(y, f"synth/y_undirdata_{snr}.pt")



