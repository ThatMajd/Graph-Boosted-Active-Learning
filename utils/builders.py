from utils.metrics import Similarity
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch

class GraphBuilder:
    def __init__(self, metric: Similarity):
        self.metric = metric

    def connect(self, A: np.ndarray, qunatile: float = 0.5):
        """Function to get the edges of `N` nodes, based on affinity matrix.

        Args:
            A (ndarray): Affinity matrix of size `NxN`, where `N` is the number of nodes.
            threshold (float, optional): nodes with affinity less than the threshold will be connected. Defaults to 1.
            quantile (float, optional): 

        Returns:
            ndarray: indices of connected nodes `2xN`
        """
        unique_distances = A[np.triu_indices_from(A, k=1)]
        
        threshold = np.quantile(unique_distances, q=qunatile)
        return np.vstack(np.where(A < threshold))

    def build(self, X: np.ndarray, y: np.ndarray = None, qunatile: float = 0.5, pytorch=False):
        """build graph from tabular data.

        Args:
            X (ndarray): data matrix.
            y (ndarray, optional): labels of the data.
            threshold (float, optional): Threshold for affinity. Defaults to .1 of max distance.

        Returns:
            (Graph, ndarray): graph `G` which is an object of networkx.Graph and its edges `E` in GNN format (2xN).
        """
        A = self.metric(X)
        if threshold is None:
            threshold = .1 * A.max()
        
        E = self.connect(A, qunatile)
  
        if pytorch:
            return Data(x=torch.tensor(X), y=torch.tensor(y), edge_index=torch.tensor(E))
        
        G = nx.Graph()
        G.add_nodes_from([(idx, dict(embedding=tuple(x), label=(y[idx] if y is not None else None))) for idx, x in enumerate(X)])
        G.add_edges_from(zip(*E))
        
        return G

    def __call__(self, X, y, qunatile: float = 0.5, pytorch: bool = False):
        return self.build(X, y, qunatile, pytorch=pytorch)