"""
Generate synthetic datasets. 
Code is adapted from https://github.com/SimoneFiorellino/AirTNN/
"""

import torch
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import random
from tqdm import tqdm

from data_utils_sbm import white_noise, k_hop_adjacency_matrix
from cell_utils import get_incidences 
import sys 
sys.path.append('../')
from utils import parse_args


class CellDataset(Dataset):
    """A dataset of stochastic block model graphs."""
    torch.manual_seed(42)

    def _create_adj_matrix(self, community_size, p_intra, p_inter):
        """Create a stochastic block model adjacency matrix."""
        # creating the SBM graph
        sizes = [self.n_nodes_for_each_community for _ in range(community_size)]
        p = np.full((community_size, community_size), p_inter)
        np.fill_diagonal(p, p_intra)
        G = nx.stochastic_block_model(sizes, p, seed=1418, directed=self.directed)

        # detecting communities
        communities = {i: [] for i in range(community_size)}
        for node in G.nodes():
            community = G.nodes[node]["block"]
            communities[community].append(node)

        return communities, G

    def _spectral_normalization(self, adj_matrix):
        """Normalize the adjacency matrix."""
        # normalize the adjacency matrix
        max_eig = torch.linalg.eigh(adj_matrix)[0][-1]
        adj_matrix = adj_matrix / max_eig
        return adj_matrix
    
    def _fill_diagonal(self, adj_matrix, n=1):
        """Fill the diagonal of the adjacency matrix."""
        return adj_matrix.fill_diagonal_(n)
    
    def _create_edge_signal(self):
        """Create an edge signal."""
        std = 2*torch.sqrt(torch.Tensor([self.len_edge_list]))
        edge_component = torch.randn(self.len_edge_list, 1) / std
        return edge_component

    def _truncated_normal(self, mean, std):
        """Create a truncated normal distribution."""
        x = torch.randn(1) * std + mean
        while x > -std and x < std:
            x = torch.randn(1) * std + mean
        return x

    def _add_spike(self, signal, source, delta):
        """Add a spike to the signal. The is defined as a truncated normal distro."""
        signal[source] += self._truncated_normal(0, delta)
        return signal

    def _sample_source_and_label(self, communities):
        """Sample a source node and a label."""
        # sample a source node
        source_community = np.random.randint(len(communities))
        source_node = np.random.choice(communities[source_community])
        
        return source_node, source_community

    def _create_edge_subsets(self, edge_list, communities):
        # create dictionary to map nodes to their corresponding community
        node_to_community = {}
        for community, nodes in communities.items():
            for node in nodes:
                node_to_community[node] = community

        # create subsets of edges within each community
        community_subsets = []
        for community, nodes in communities.items():
            edges = [(u, v) for u, v in edge_list if u in nodes and v in nodes]
            community_subsets.append(edges)

        # create subset of edges between communities
        between_community_edges = [(u, v) for u, v in edge_list if node_to_community[u] != node_to_community[v]]

        # check if all edges are used
        all_edges = set(edge_list)
        used_edges = set()
        for subset in community_subsets + [between_community_edges]:
            used_edges.update(subset)
        assert used_edges == all_edges, "Not all edges were used"

        # return list of subsets
        return community_subsets + [between_community_edges]

    def _diffused_signal(self, signal, k, snr_db):
        """Diffuse the signal."""
        if k == 0:
            return signal + white_noise(signal, snr_db)
        diff_signal = self.S[k,:,:].reshape(self.len_edge_list,self.len_edge_list) @ signal
        return diff_signal + white_noise(signal, snr_db)

    def _get_shift_matrix(self):
        S = k_hop_adjacency_matrix(self.edge_adj, self.k_diffusion)
        return S
    
    def _get_source_edges(self, edge_list, edge_subsets, n_community, fixes_source_idxs, fixed=True):
        if fixed:
            return fixes_source_idxs[n_community]
        else:
            return edge_list.index(random.choice(edge_subsets[n_community]))
            
    def _sample_k_from(self, max, scale=2.5, shape=(1,), dtype=torch.float32):
        # scale 2.5 
        # Create a Student's t-distribution with 3 degrees of freedom
        t_dist = torch.distributions.StudentT(df=10, scale=scale)

        # Generate a sample from the distribution
        t_sample = abs(t_dist.sample())

        while t_sample>=max:
            t_sample = abs(t_dist.sample())

        return t_sample.long()

    def _support_of_matrix(self, matrix, diagonal=True):
        """Return the support of a matrix. Diagonal is not included."""
        if diagonal:
            matrix.fill_diagonal_(0)
        return matrix.ne(0).float()
    
    def _symmetric_normalization(self, matrix):
        """Return the symmetric normalization of a matrix."""
        matrix.fill_diagonal_(0)
        degrees = matrix.sum(dim=1)
        degrees[degrees == 0] = 1
        degrees = degrees.pow(-0.5)
        degrees = degrees.view(-1, 1)
        return degrees * matrix * degrees.T
    
    def _stochastic_normalization(self, matrix):
        """Return the stochastic normalization of a matrix."""
        matrix.fill_diagonal_(0)
        degrees = matrix.sum(dim=1)
        degrees[degrees == 0] = 1
        degrees = degrees.pow(-1)
        degrees = degrees.view(-1, 1)
        return degrees * matrix

    def _matrix_normalization(self, matrix, ntype='identity'):
        """Return the stochastic normalization of a matrix."""
        if ntype == 'identity':
            return matrix
        elif ntype == 'symmetric':
            return self._symmetric_normalization(matrix)
        elif ntype == 'stochastic':
            return self._stochastic_normalization(matrix)
        elif ntype == 'spectral':
            return self._spectral_normalization(matrix)  

    def __init__(self, n_nodes, n_community, p_intra, p_inter, num_samples, k_diffusion, spike, snr_db, n_spikes, 
                 ntype='identity',
                 directed=False):

        """Initialize the dataset."""
        self.k_diffusion = k_diffusion
        self.n_nodes = n_nodes
        self.num_samples = num_samples
        self.n_nodes_for_each_community = n_nodes // n_community
        self.directed = directed
        # ---------------------
        # Generate the SBM graph
        # ---------------------
        self.communities, self.G = self._create_adj_matrix(n_community, p_intra, p_inter)
        
        node_list = torch.tensor([i for i in range(len(list(self.G.nodes)))])
        edge_list = [(i, j) for i, j in self.G.edges] # list of edges in the graph (edges are not duplicated for undirected graphs)
        self.len_edge_list = len(edge_list)
        # ---------------------
        # Compute the upper and lower laplacians
        # ---------------------
        self.lower_incidence = get_incidences(node_list, edge_list)
        print("Number of edges: ", self.len_edge_list)
        self.lower_laplacian = self.lower_incidence.T @ self.lower_incidence
    
        # normalize the laplacians
        if self.directed:
            line_graph = torch.tensor(nx.adjacency_matrix(nx.line_graph(self.G)).todense())
            edge_adj = self._support_of_matrix(line_graph)
        else:
            edge_adj = self._support_of_matrix(self.lower_laplacian)
        edge_adj = self._fill_diagonal(edge_adj)
        self.edge_adj = self._matrix_normalization(edge_adj, 'spectral')
        self.lower_laplacian = self._spectral_normalization(self.lower_laplacian)

        # ---------------------
        # Generate the S matrix
        # ---------------------
        self.S = self._get_shift_matrix()

        # convert the laplacians to sparse matrices
        # compute the support -> normalization -> to sparse
        self.sparse_lower_laplacian = self._matrix_normalization(self._support_of_matrix(self.lower_laplacian), ntype).to_sparse()

        # check if the S matrix has Nan values
        if torch.isnan(self.S).any():
            print("Nan values in the S matrix")

        # ---------------------
        # Source edges
        # ---------------------
        edge_subsets = self._create_edge_subsets(edge_list, self.communities)
        # random choice of one edge from each subset
        fixed_sources = [random.choice(edge_subsets[i]) for i in range(len(edge_subsets))]

        # print the number of edge for each community
        for i in range(len(edge_subsets)):
            print(f"Number of edges in community {i}: {len(edge_subsets[i])}")

        # return the indices of the source edges in the edge_list
        fixes_source_idxs = [edge_list.index(fixed_sources[i]) for i in range(len(edge_subsets))]
        self.samples = []

        # mean of edges in each community
        mean_edges = self.len_edge_list // len(edge_subsets)
        print("Mean edges: ", mean_edges)
        # mean_edges / 100 * percentage
        n_spikes = int(mean_edges * n_spikes)
        print("Number of spikes: ", n_spikes)

        # ---------------------
        # Generate the samples
        # ---------------------
        diffusion_hist = []
        for i in tqdm(range(num_samples)):
            # choose the diffusion order
            if self.k_diffusion == 0:
                k = 0
            else:
                k = self._sample_k_from(max=self.k_diffusion)
                diffusion_hist.append(k)
            # choose the community
            n_community = torch.randint(0, len(edge_subsets), (1,))
            xp = self._create_edge_signal()
            # add the spikes
            for i in range(n_spikes):
                source = self._get_source_edges(edge_list, edge_subsets, n_community, fixes_source_idxs, fixed=False)
                xp = self._add_spike(xp, source, spike)

            # diffusion and noise
            xp = self._diffused_signal(xp, k, snr_db)
            self.samples.append((xp, n_community))
        # print some stats about the diffusion order
        print("Diffusion order stats: ")
        print(f"Mean: {sum(diffusion_hist) / len(diffusion_hist)}")
        print(f"Max: {max(diffusion_hist)}")
        print(f"Min: {min(diffusion_hist)}")
         
    def __len__(self):
        """Return the length of the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """Return the sample and label at the given index."""
        sample, label = self.samples[idx]
        return sample, label

# test main function
if __name__ == "__main__":

    args = parse_args()

    directed = args.directed_dataset
    snr = args.snr

    dataset = CellDataset(
        n_nodes = 70,
        n_community = 10, 
        p_intra = 0.9, 
        p_inter = 0.01, 
        num_samples = 1000, 
        k_diffusion = 100,
        snr_db = snr,
        spike = 1,
        n_spikes = 0.8,
        directed = directed
    )

    # Save the dataset
    if directed:
        torch.save(dataset, f'synth/cell_sbm_dataset_directed_{snr}.pt')
    else:
        torch.save(dataset, f'synth/cell_sbm_dataset_undirected_{snr}.pt')
    