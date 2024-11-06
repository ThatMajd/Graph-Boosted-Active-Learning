import torch
from torch_geometric.data import Data, Dataset
from utils.Similarity import Similarity
from torch_geometric.utils import to_networkx


class GraphBuilder:
    def __init__(self, dataset: Dataset, metric: str, threshold: float = 0.8):
        self.similarity = Similarity(metric)
        self.threshold = threshold

        X = dataset[:][0]
        
        self.graph = self._build_graph(X)
        self.nx_graph = to_networkx(self.graph, to_undirected=True)

    def get_graph(self):
        return self.graph
    
    def get_nx_graph(self):
        return self.nx_graph

    def _build_graph(self, X):
        # Compute the similarity (affine) matrix
        affine_matrix = self.similarity(X)
        E = torch.vstack(torch.where(affine_matrix < self.threshold))

        # Create PyTorch Geometric Data object
        data = Data(x=X, edge_index=E)
        return data

