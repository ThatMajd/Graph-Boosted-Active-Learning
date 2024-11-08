from utils.metrics import Similarity
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch

class GraphBuilder:
    def __init__(self, metric: Similarity):
        self.metric = metric

    def connect(self, A: np.ndarray, threshold: float = 1):
        """Function to get the edges of `N` nodes, based on affinity matrix.

        Args:
            A (ndarray): Affinity matrix of size `NxN`, where `N` is the number of nodes.
            threshold (float, optional): nodes with affinity less than the threshold will be connected. Defaults to 1.

        Returns:
            ndarray: indices of connected nodes `2xN`
        """
        return np.vstack(np.where(A < threshold))

    def build(self, X: np.ndarray, y: np.ndarray = None, threshold: float = None, pytorch=False):
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
        
        E = self.connect(A, threshold)
  
        if pytorch:
            return Data(x=torch.tensor(X), y=torch.tensor(y), edge_index=torch.tensor(E))
        
        G = nx.Graph()
        G.add_nodes_from([(idx, dict(embedding=tuple(x), label=(y[idx] if y is not None else None))) for idx, x in enumerate(X)])
        G.add_edges_from(zip(*E))
        
        return G

    def __call__(self, X, y, threshold: float = None, pytorch: bool = False):
        return self.build(X, y, threshold, pytorch=pytorch)