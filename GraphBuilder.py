import torch_geometric
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from Similarity import Similarity

class GraphBuilder:
    def __init__(self, dataset: Dataset, metric: str, threshold: float = 0.8):
        self.similarity = Similarity(metric)
        self.threshold = threshold

        X = dataset[:][0]
        
        self.graph = self._build_graph(X)

    def get_graph(self):
        return self.graph

    def _build_graph(self, X):
        # Compute the similarity (affine) matrix
        affine_matrix = self.similarity(X)

        # Apply threshold to the similarity matrix
        edge_mask = affine_matrix > self.threshold
        edge_index, edge_attr = dense_to_sparse(edge_mask.float())

        # Create PyTorch Geometric Data object
        data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr)

        return data

