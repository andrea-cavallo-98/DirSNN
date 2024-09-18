"""
DirSNN and DirSNNClassifier implementation
"""
import torch
from layers import DirSNNLayer


class DirSNNClassifier(torch.nn.Module):
    """Readout classifier on top of DirSNN
        Parameters
    ----------
    n_classes : int
        Number of classes for classification task
    edge_channels: int
        Dimension of edge features
    n_layers: int
        Number of layers.
    n_hid_conv: int
        Dimension of features of hidden layers on edges.
    n_hid_mlp : int
        Hidden dimension of readout MLP
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    n_adjs: int
        Number of adjacency matrixes used.
    aggr_norm: bool
        Whether to normalize the aggregation.
    update_func: str
        Update function for the simplicial complex convolution.
    """

    def __init__(self, n_classes, edge_channels, n_layers, n_hid_conv = 32, 
                n_hid_mlp = 32, conv_order=1, 
                n_adjs=1,aggr_norm=False, update_func="relu"):
        super().__init__()
        self.n_classes = n_classes
        self.n_hid_conv = n_hid_conv
        self.n_hid_mlp = n_hid_mlp

        self.scconv = DirSNN(edge_channels, n_layers=n_layers, n_hid=n_hid_conv, 
                             conv_order=conv_order, n_adjs=n_adjs, 
                             aggr_norm=aggr_norm, update_func=update_func)
        
        self.readout = torch.nn.Sequential(
            torch.nn.Linear(self.n_hid_conv, self.n_hid_mlp),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.n_hid_mlp, self.n_classes)
        )
        self.readout.apply(self.weights_init)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)


    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)


    def forward(self, x_1, adjs):
        
        x_1 = self.scconv(x_1, adjs)

        # maxpool on edge features and log softmax for class likelihoods
        x = self.readout(x_1.max(1)[0]) 
        y = self.log_softmax(x)
        return y




class DirSNN(torch.nn.Module):
    """DirSNN implementation [1].

    Parameters
    ----------
    edge_channels: int
        Dimension of edge features
    n_layers: int
        Number of layers.
    n_hid: int
        Dimension of features of hidden layers on edges.
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    n_adjs: int
        Number of adjacency matrixes used.
    aggr_norm: bool
        Whether to normalize the aggregation.
    update_func: str
        Update function for the simplicial complex convolution.
    """

    def __init__(
        self,
        edge_channels, 
        n_layers=2, 
        n_hid=1,
        conv_order=1,
        n_adjs=1,
        aggr_norm=False,
        update_func=None,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_1 = torch.nn.Linear(edge_channels, n_hid)

        self.layers = torch.nn.ModuleList(
            DirSNNLayer(
                in_channels_1=n_hid,
                out_channels_1=n_hid,
                conv_order=conv_order,
                n_adjs=n_adjs,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_1, adjs):
        """Forward computation.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (batch_size, n_edges, n_features).

        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors.
            Each entry shape = (n_edges,n_edges).

        Returns
        -------
        x_1 : torch.Tensor
            Output edge representations.
            Shape = (batch_size, n_edges, output_size).
        """
        in_x_1 = self.in_linear_1(x_1)

        # Forward through DirSNN layers
        x_1 = in_x_1
        for layer in self.layers:
            x_1 = layer(x_1, adjs)

        return x_1






