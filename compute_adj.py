"""
Computations of lower and upper (directed) adjacency matrixes
and incidence matrixes
"""
import networkx as nx
import numpy as np

def compute_lower_adj(G):
    """
    Compute directed edge lower adjacencies for a digraph G.    
    Parameters
    ----------
    G : nx.digraph

    Returns
    -------
    adj_low_100, adj_low_101, adj_low_110, adj_low_111 : np.array of shape (n_edges, n_edges)
        Directed edge lower adjacency matrixes
    """
    adj_low_101 = nx.adjacency_matrix(nx.line_graph(G)).todense().astype(float)
    adj_low_110 = adj_low_101.T

    edge_index = np.stack([np.array([e[0],e[1]]).T for e in G.edges()])
    adj_low_100 = (edge_index[:,0].reshape((-1,1)) == edge_index[:,0].reshape((1,-1))).astype(float)
    adj_low_111 = (edge_index[:,1].reshape((-1,1)) == edge_index[:,1].reshape((1,-1))).astype(float)
    return adj_low_100, adj_low_101, adj_low_110, adj_low_111

def compute_upper_adj(G):
    """
    Compute directed edge upper adjacencies for a digraph G.    
    Parameters
    ----------
    G : nx.digraph

    Returns
    -------
    adj_up_101, adj_up_102, adj_up_112, adj_up_110, adj_up_120, adj_up_121 : np.array of shape (n_edges, n_edges)
        Directed edge upper adjacency matrixes
    """

    edge_list = G.edges()
    n_edges = len(edge_list)
    all_triangles = sorted(nx.simple_cycles(G.to_undirected(), length_bound=3))
    dir_triangles = []
    # Check if triangles are in the correct order
    for t in all_triangles:
        if (t[0],t[1]) in edge_list and (t[0],t[2]) in edge_list and (t[1],t[2]) in edge_list:
            dir_triangles.append(t)
        elif (t[0],t[2]) in edge_list and (t[0],t[1]) in edge_list and (t[2],t[1]) in edge_list:
            dir_triangles.append([t[0], t[2], t[1]])
        elif (t[1],t[0]) in edge_list and (t[1],t[2]) in edge_list and (t[0],t[2]) in edge_list:
            dir_triangles.append([t[1], t[0], t[2]])
        elif (t[1],t[2]) in edge_list and (t[1],t[0]) in edge_list and (t[2],t[0]) in edge_list:
            dir_triangles.append([t[1], t[2], t[0]])
        elif (t[2],t[1]) in edge_list and (t[2],t[0]) in edge_list and (t[1],t[0]) in edge_list:
            dir_triangles.append([t[2], t[1], t[0]])
        elif (t[2],t[0]) in edge_list and (t[2],t[1]) in edge_list and (t[0],t[1]) in edge_list:
            dir_triangles.append([t[2], t[0], t[1]])

    adj_up_101, adj_up_102, adj_up_112 = np.zeros([n_edges, n_edges]), np.zeros([n_edges, n_edges]), np.zeros([n_edges, n_edges])
    adj_up_110, adj_up_120, adj_up_121 = np.zeros([n_edges, n_edges]), np.zeros([n_edges, n_edges]), np.zeros([n_edges, n_edges])

    edges_to_id = {e:i for i,e in enumerate(edge_list)}

    for t in dir_triangles:
        adj_up_101[edges_to_id[(t[1],t[2])], edges_to_id[(t[0],t[2])]] = 1
        adj_up_102[edges_to_id[(t[1],t[2])], edges_to_id[(t[0],t[1])]] = 1
        adj_up_112[edges_to_id[(t[0],t[2])], edges_to_id[(t[0],t[1])]] = 1
        adj_up_110[edges_to_id[(t[0],t[2])], edges_to_id[(t[1],t[2])]] = 1
        adj_up_120[edges_to_id[(t[0],t[1])], edges_to_id[(t[1],t[2])]] = 1
        adj_up_121[edges_to_id[(t[0],t[1])], edges_to_id[(t[0],t[2])]] = 1

    return adj_up_101, adj_up_102, adj_up_112, adj_up_110, adj_up_120, adj_up_121


def compute_lower_adj_undirected(G):
    """
    Compute undirected edge lower adjacencies for a digraph G.    
    Parameters
    ----------
    G : nx.digraph

    Returns
    -------
    adj_1_down : np.array of shape (n_edges, n_edges)
        Undirected edge lower adjacency matrix
    """

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    inc_0_1 = np.zeros([n_nodes, n_edges])

    for edge_id, edge in enumerate(G.edges()): 
        inc_0_1[edge[0], edge_id] = 1 
        inc_0_1[edge[1], edge_id] = 1 

    adj_1_down = np.matmul(inc_0_1.T, inc_0_1)
    np.fill_diagonal(adj_1_down, 0)

    return adj_1_down

def compute_incidences(G, directed_dataset=True):
    """
    Compute undirected edge lower adjacencies for a digraph G.    
    Parameters
    ----------
    G : nx.digraph
    directed_dataset : 
        Whether the underlying graph is directed or not

    Returns
    -------
    inc_0_1 : np.array of shape (n_nodes, n_edges)
        Incidence matrix of nodes to edges
    adj_0_up : np.array of shape (n_edges, n_edges)
        Node adjacency matrix (symmetric if directed_dataset = False)
    """

    inc_0_1 = np.zeros([G.number_of_nodes(), G.number_of_edges()])
    adj_0_up = np.zeros([G.number_of_nodes(), G.number_of_nodes()])
    
    for edge_id, edge in enumerate(G.edges()): 
        adj_0_up[edge[0], edge[1]] = 1
        if directed_dataset:
            inc_0_1[edge[0], edge_id] = -1 # outgoing edge
        else:
            adj_0_up[edge[1], edge[0]] = 1 # symmetric edges
            inc_0_1[edge[0], edge_id] = 1 
        inc_0_1[edge[1], edge_id] = 1 # incoming edge

    return inc_0_1, adj_0_up
