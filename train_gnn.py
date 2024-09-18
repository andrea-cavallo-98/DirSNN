"""
Train and evaluate DirGNN and GCN on the synthetic dataset
"""
import torch
from sklearn.model_selection import train_test_split 
from utils import compute_multiclass_accuracy, parse_args
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import numpy as np
from gnns import GNN

args = parse_args()

directed = args.directed
directed_dataset = args.directed_dataset
snr = args.snr
spike = args.spike

dset_string = "dirdata" if directed_dataset else "undirdata"

inc_0_1, adj_0_up, x_1 = torch.load(f"data/synth/inc_{dset_string}_{snr}.pt")
y = torch.load(f"data/synth/y_{dset_string}_{snr}.pt")

if directed_dataset:
    if directed:
        edge_index = torch.LongTensor(np.stack(adj_0_up.nonzero()))
    else:
        # Make adj_0_up symmetric -> otherwise it is using random directions
        adj_0_up += adj_0_up.T
        np.fill_diagonal(adj_0_up, 0)
        edge_index = torch.LongTensor(np.stack(adj_0_up.nonzero()))

else:
    edge_index = torch.LongTensor(np.stack(adj_0_up.nonzero()))

inc_0_1 = torch.FloatTensor(inc_0_1)

## Generate node signals by projecting edge signals on nodes
x_0 = torch.matmul(inc_0_1, x_1.T).T

# Normalize data
scaler = StandardScaler()

x_0 = torch.FloatTensor(scaler.fit_transform(x_0))

x_0_train, x_0_val, y_train, y_val = train_test_split(x_0, y, test_size=0.2, random_state=42, stratify=y)
x_0_val, x_0_test, y_val, y_test = train_test_split(x_0_val, y_val, test_size=0.5, random_state=42, stratify=y_val)
n_classes = 11
n_train = x_0_train.shape[0]

for it in range(args.iterations):

    if directed:
        gnn = GNN(num_features=1, num_classes=n_classes, hidden_dim=args.n_hid_conv,
                    num_layers=args.n_layers, dropout=0, conv_type="dir-gcn", jumping_knowledge=False,
                    normalize=False, alpha=1 / 2, learn_alpha=True)
    else:
        gnn = GNN(num_features=1, num_classes=n_classes, hidden_dim=args.n_hid_conv,
                    num_layers=args.n_layers, dropout=0, conv_type="gcn", jumping_knowledge=False,
                    normalize=False, alpha=1 / 2, learn_alpha=True)

    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=0.001)
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
            x_0_train_batch = x_0_train[thisBatchIndices]
            yTrainBatch = y_train[thisBatchIndices]

            gnn.zero_grad()
            yHatTrain = gnn(x_0_train_batch.unsqueeze(2), edge_index)
            lossValueTrain = loss(yHatTrain.squeeze(), yTrainBatch)
            accTrain = compute_multiclass_accuracy(yHatTrain.squeeze(), yTrainBatch)
            pred = yHatTrain.squeeze().argmax(1)
            lossValueTrain.backward()
            optimizer.step()
            tot_train_loss.append(lossValueTrain.detach())
            tot_train_acc.append(accTrain)

        with torch.no_grad():
            yHatVal = gnn(x_0_val.unsqueeze(2), edge_index)
            lossValueVal = loss(yHatVal.squeeze(), y_val)
            accVal = compute_multiclass_accuracy(yHatVal.squeeze(), y_val)

            if accVal > Best_Valid_acc:
                best_model = deepcopy(gnn)
                Best_Valid_acc = accVal

            print(f"Epoch {epoch} train loss {lossValueTrain.mean().item()}" +
                f" train acc {accTrain.mean().item()} val loss {lossValueVal.detach().item()} val acc {accVal.item()}")

    # TEST
    yHatTest = best_model(x_0_test.unsqueeze(2), edge_index)
    lossTest = loss(yHatTest.squeeze(), y_test)
    accTest = compute_multiclass_accuracy(yHatTest.squeeze(), y_test)

    print(f"Loss test: {lossTest.detach().item()} acc test: {accTest.item()}")

