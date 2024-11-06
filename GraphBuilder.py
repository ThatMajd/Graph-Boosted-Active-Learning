import torch
from torch_geometric.data import Data, Dataset
from utils.Similarity import Similarity
from torch_geometric.utils import to_networkx


class GraphBuilder:
    def __init__(self, metric: str, threshold: float = 0.8):
        self.similarity = Similarity(metric)
        self.threshold = threshold

        self.graph, self.nx_graph = None, None

    def get_graph(self):
        if self.graph:
            return self.graph
        else:
            raise Exception("You need to build the graph first")
    
    def get_nx_graph(self):
        if self.nx_graph:
            return self.nx_graph
        else:
            raise Exception("You need to build the graph first")

    def __call__(self, X, y=None):
        # Compute the similarity (affine) matrix
        affine_matrix = self.similarity(X)
        E = torch.vstack(torch.where(affine_matrix < self.threshold))

        # Create PyTorch Geometric Data object
        data = Data(x=X, edge_index=E)

        self.graph = data
        self.nx_graph = to_networkx(self.graph, to_undirected=True, node_attrs=["x"])

        return self.graph, self.nx_graph

