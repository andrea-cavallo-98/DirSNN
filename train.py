"""
Train and evaluate DirSNN and SNN on the synthetic dataset
"""

import torch
from sklearn.model_selection import train_test_split 
from model import DirSNNClassifier
from utils import compute_multiclass_accuracy, parse_args, spectral_normalization
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

args = parse_args()

directed = args.directed
directed_dataset = args.directed_dataset
snr = args.snr
spike = args.spike

dset_string = "dirdata" if directed_dataset else "undirdata"
model_string = "dirmodel" if directed else "undirmodel"

data = torch.load(f"data/synth/data_{model_string}_{dset_string}_{snr}.pt")
y = torch.load(f"data/synth/y_{dset_string}_{snr}.pt")


if directed:
    (x_1, adj_low_100, adj_low_101, adj_low_110, adj_low_111) = data
    adj_low_100 = torch.FloatTensor(spectral_normalization(adj_low_100))
    adj_low_101 = torch.FloatTensor(spectral_normalization(adj_low_101))
    adj_low_110 = torch.FloatTensor(spectral_normalization(adj_low_110))
    adj_low_111 = torch.FloatTensor(spectral_normalization(adj_low_111))
    adjs = (adj_low_100, adj_low_101, adj_low_110, adj_low_111)

else: # same model, but different adjacencies
    (x_1, adj_1_down) = data
    adj_1_down = torch.FloatTensor(spectral_normalization(adj_1_down))
    adjs = [adj_1_down]

    
# Normalize data
scaler = StandardScaler()

x_1 = torch.FloatTensor(scaler.fit_transform(x_1))

x_1_train, x_1_val, y_train, y_val = train_test_split(x_1, y, test_size=0.2, random_state=42, stratify=y)
x_1_val, x_1_test, y_val, y_test = train_test_split(x_1_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
n_classes = 11
n_train = x_1_train.shape[0]

for it in range(args.iterations):

    scconv = DirSNNClassifier(edge_channels=1, n_layers=args.n_layers, 
            n_classes=n_classes, n_hid_conv=args.n_hid_conv, n_hid_mlp=args.n_hid_mlp,
            conv_order=args.conv_order, n_adjs=len(adjs), update_func="leaky_relu")

    optimizer = torch.optim.Adam(scconv.parameters(), lr=args.lr, weight_decay=0.001)

    loss = torch.nn.NLLLoss()
    Best_Valid_acc = 0

    batchSize = args.batchSize
    nTrainBatches = int(np.ceil(n_train / batchSize))

    for epoch in range(args.nEpochs):
        tot_train_loss = []
        tot_train_acc = []
        train_perm_idx = torch.randperm(n_train)

        for batch in range(nTrainBatches):

            thisBatchIndices = torch.LongTensor(train_perm_idx[batch * batchSize : (batch + 1) * batchSize])
            x_1_train_batch = x_1_train[thisBatchIndices]
            yTrainBatch = y_train[thisBatchIndices]

            scconv.zero_grad()
            yHatTrain = scconv(x_1_train_batch.unsqueeze(2), adjs)
            
            lossValueTrain = loss(yHatTrain.squeeze(), yTrainBatch)
            accTrain = compute_multiclass_accuracy(yHatTrain.squeeze(), yTrainBatch)
            pred = yHatTrain.squeeze().argmax(1)
            lossValueTrain.backward()
            optimizer.step()
            tot_train_loss.append(lossValueTrain.detach())
            tot_train_acc.append(accTrain)

        with torch.no_grad():
            yHatVal = scconv(x_1_val.unsqueeze(2), adjs)
            
            lossValueVal = loss(yHatVal.squeeze(), y_val)
            accVal = compute_multiclass_accuracy(yHatVal.squeeze(), y_val)

            if accVal > Best_Valid_acc:
                best_model = deepcopy(scconv)
                Best_Valid_acc = accVal

            print(f"Epoch {epoch} train loss {lossValueTrain.mean().item()}" +
                f" train acc {accTrain.mean().item()} val loss {lossValueVal.detach().item()} val acc {accVal.item()}")

    # TEST
    yHatTest = best_model(x_1_test.unsqueeze(2), adjs)

    lossTest = loss(yHatTest.squeeze(), y_test)
    accTest = compute_multiclass_accuracy(yHatTest.squeeze(), y_test)

    print(f"Loss test: {lossTest.detach().item()} acc test: {accTest.item()}")
