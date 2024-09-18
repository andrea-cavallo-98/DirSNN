"""
Layer of DirSNN. 
Adapted from https://github.com/pyt-team/TopoModelX/blob/main/topomodelx/nn/simplicial/sccnn_layer.py
"""

import torch
from torch.nn.parameter import Parameter


class DirSNNLayer(torch.nn.Module):
    r"""Layer of a Directed Simplicial Neural Network.

    Parameters
    ----------

    in_channels_1 : int
        Dimensions of input features on edges
    out_channels_1 : int
        Dimensions of output features on edges
    conv_order : int
        Convolution order of the simplicial filter
    n_adjs : int
        Number of adjacency matrixes used
    aggr_norm : bool = False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : str, default = None
        Activation function used in aggregation layers.
    initialization : str, default = "xavier_normal"
        Weight initialization method.
    """

    def __init__(
        self,
        in_channels_1,
        out_channels_1,
        conv_order,
        n_adjs = 1,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        self.in_channels_1 = in_channels_1
        self.out_channels_1 = out_channels_1

        self.conv_order = conv_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization
        self.n_adjs = n_adjs

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                conv_order * n_adjs + 1,
            )
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization."""
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor, shape = (batch_size,n_target_cells, out_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "leaky_relu":
            return torch.nn.functional.leaky_relu(x)
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse, shape = (n_simplices,n_simplices)
            Convolution operator e.g., the adjacency matrix, or the Hodge Laplacians.
        conv_order : int
            The order of the convolution.
        x : torch.Tensor, shape = (batch_size,n_simplices,num_channels)
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor. x[:, :, k] = (conv_operator@....@conv_operator) @ x.
        """
        batch_size, num_simplices, num_channels = x.shape
        X = torch.empty(size=(batch_size, num_simplices, num_channels, conv_order))

        if self.aggr_norm:
            X[:, :, :, 0] = torch.matmul(conv_operator, x)
            X[:, :, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, :, 0])
            for k in range(1, conv_order):
                X[:, :, :, k] = torch.matmul(conv_operator, X[:, :, :, k - 1])
                X[:, :, :, k] = self.aggr_norm_func(conv_operator, X[:, :, :, k])
        else:
            X[:, :, :, 0] = torch.matmul(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, :, k] = torch.matmul(conv_operator, X[:, :, :, k - 1])
        return X

    def forward(self, x_1, laplacian_all):
        r"""Forward computation.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (batch_size,n_edges,in_channels_1),
            Edge features

        laplacian_all: tuple of tensors, len = n_adjs
            Tuple of adjacency tensors

        Returns
        -------
        y_1 : torch.Tensor
            Output features on edges.
        """

        num_edges = x_1.shape[1]
        identity_1 = torch.eye(num_edges)

        # convolution in the edge space

        x_1_all = []

        x_1_all.append(torch.unsqueeze(identity_1 @ x_1, 3))
        for adj in laplacian_all:
            x_1_all.append(self.chebyshev_conv(adj, self.conv_order, x_1))

        assert len(x_1_all) == self.n_adjs + 1

        x_1_all = torch.cat(x_1_all, dim=3)
        y_1 = torch.einsum("bnik,iok->bno", x_1_all, self.weight_1)

        if self.update_func is None:
            return y_1

        return self.update(y_1)