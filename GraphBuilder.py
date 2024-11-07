import torch
import numpy as np
import networkx as nx
from utils.Similarity import Similarity
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx


class GraphBuilder:
	def __init__(self, similarity_metric: str, threshold: float = 0.8):
		self.similarity = Similarity(similarity_metric)
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
		E = np.vstack(np.where(affine_matrix < self.threshold))

		# Create PyTorch Geometric Data object
		data = Data(x=X, edge_index=torch.Tensor(E))

		self.graph = data

		# print("graph being built")
		self.nx_graph = nx.Graph()
		self.nx_graph.add_nodes_from(range(len(X)))
		self.nx_graph.add_edges_from(zip(*E))
		# print("graph is finished")

		# self.nx_graph = to_networkx(self.graph, to_undirected=True, node_attrs=["x"])

		return self.graph, self.nx_graph

