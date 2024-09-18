"""
Replicate the experiment on expressivity
"""
import torch
from model import DirSNNClassifier
from utils import compute_multiclass_accuracy, parse_args
import networkx as nx
from compute_adj import compute_lower_adj, compute_upper_adj
from tqdm import tqdm
from gnns import GNN

args = parse_args()
nEpochs = 1000

# Create the graphs in Fig. 8 and compute their node and edge adjacency matrixes

G1 = nx.DiGraph()
for i in range(7):
    G1.add_node(i)

G1.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6),
                  (0,2),(1,3),(2,4),(3,5),(4,6),(5,0),(6,1),(6,0)])

# For Dir-SNN
adj_low_100_1, adj_low_101_1, adj_low_110_1, adj_low_111_1 = compute_lower_adj(G1)
_, _, _, _, adj_up_120_1, _ = compute_upper_adj(G1)

# For Dir-GNN
x_0_train_1 = torch.FloatTensor([1]*G1.number_of_nodes()).reshape([-1,1]).unsqueeze(0)
edge_index_1 = torch.stack([torch.tensor([e[0], e[1]]) for e in G1.edges()]).T


G2 = nx.DiGraph()
for i in range(7):
    G2.add_node(i)
G2.add_edges_from([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,0),
                  (0,3),(3,6),(6,2),(2,5),(5,1),(1,4),(4,0)])

# For Dir-SNN
adj_low_100_2, adj_low_101_2, adj_low_110_2, adj_low_111_2 = compute_lower_adj(G2)
_, _, _, _, adj_up_120_2, _ = compute_upper_adj(G2)

# For Dir-GNN
x_0_train_2 = torch.FloatTensor([1]*G2.number_of_nodes()).reshape([-1,1]).unsqueeze(0)
edge_index_2 = torch.stack([torch.tensor([e[0], e[1]]) for e in G2.edges()]).T

###
# Dir-SNN training and evaluation
###

adjs_1 = (torch.FloatTensor(adj_low_100_1), torch.FloatTensor(adj_low_101_1), 
            torch.FloatTensor(adj_low_110_1), torch.FloatTensor(adj_low_111_1), torch.FloatTensor(adj_up_120_1))
adjs_2 = (torch.FloatTensor(adj_low_100_2), torch.FloatTensor(adj_low_101_2), 
            torch.FloatTensor(adj_low_110_2), torch.FloatTensor(adj_low_111_2), torch.FloatTensor(adj_up_120_2))

x_1 = torch.ones([1, adj_low_100_1.shape[0]])
n_classes = 2

scconv = DirSNNClassifier(edge_channels=1, n_layers=1, 
                n_classes=n_classes, n_hid_conv=16, n_hid_mlp=1,
                conv_order=1, n_adjs=len(adjs_1), update_func="leaky_relu")

optimizer = torch.optim.Adam(scconv.parameters(), lr=args.lr)

loss = torch.nn.NLLLoss()
Best_Valid_acc = 0

print("Training and evaluating Dir-SNN")
for epoch in tqdm(range(nEpochs)):

    scconv.zero_grad()
    yHatTrain1 = scconv(x_1.unsqueeze(2), adjs_1)
    yHatTrain2 = scconv(x_1.unsqueeze(2), adjs_2)
    lossValueTrain = loss(yHatTrain1.squeeze(), torch.tensor(0)) + loss(yHatTrain2.squeeze(), torch.tensor(1))
    lossValueTrain.backward()
    optimizer.step()

# TEST
yHatTest1 = scconv(x_1.unsqueeze(2), adjs_1)
yHatTest2 = scconv(x_1.unsqueeze(2), adjs_2)
    
accTest = compute_multiclass_accuracy(torch.stack([yHatTest1.squeeze(), yHatTest2.squeeze()]), torch.tensor([0,1]))

print(f"Acc test: {accTest.item()}")


###
# Dir-GNN
###

gnn = GNN(num_features=1, num_classes=n_classes, hidden_dim=args.n_hid_conv,
            num_layers=args.n_layers, dropout=0, conv_type="dir-gcn", jumping_knowledge=False,
            normalize=False, alpha=1 / 2, learn_alpha=True)
optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr)

loss = torch.nn.NLLLoss()
Best_Valid_acc = 0

print("Training and evaluating Dir-GNN")
for epoch in tqdm(range(nEpochs)):

    gnn.zero_grad()
    yHatTrain1 = gnn(x_0_train_1, edge_index_1)
    yHatTrain2 = gnn(x_0_train_2, edge_index_2)
    
    lossValueTrain = loss(yHatTrain1.squeeze(), torch.tensor(0)) + loss(yHatTrain2.squeeze(), torch.tensor(1))
    lossValueTrain.backward()
    optimizer.step()


# TEST
yHatTest1 = gnn(x_0_train_1, edge_index_1)
yHatTest2 = gnn(x_0_train_2, edge_index_2)
accTest = compute_multiclass_accuracy(torch.stack([yHatTest1.squeeze(), yHatTest2.squeeze()]), torch.tensor([0,1]))

print(f"Acc test: {accTest.item()}")

