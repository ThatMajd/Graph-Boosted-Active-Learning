from utils.metrics import Similarity
import networkx as nx
import numpy as np

class GraphBuilder:
	def __init__(self, metric: Similarity, ):
		self.metric = metric

	def connect(self, A: np.ndarray, threshold: float = 1):
		"""Function to get the edges of `N` nodes, based on affinity matrix.

		Args:
			A (ndarray): Affinity matrix of size `NxN`, where `N` is the number of nodes.
			threshold (float, optional): nodes with affinity less than the threshold will be connected. Defaults to 1.

		Returns:
			ndarray: incides of connected nodes `2xN`
		"""
		return np.vstack(np.where(A < threshold))

	def build(self, X: np.ndarray, y: np.ndarray = None, threshold: float = None):
		"""build graph from tabular data.

		Args:
			X (ndarray): data matrix.
			y (ndarray, optional): labels of the data.
			threshold (float, optional): _description_. Defaults to .1 of max distance.

		Returns:
			(Graph, ndarray): graph `G` which is an object of networkx.Graph and it's edges `E` which are in GNN format (2xN).
		"""
		A = self.metric(X)
		if threshold is None:
			threshold = .1 * A.max()
		
		E = self.connect(A, threshold)



		G = nx.Graph()
		G.add_nodes_from([(idx, dict(embedding=tuple(x), label=(y[idx] if y is not None else None))) for idx, x in enumerate(X)])
		G.add_edges_from(zip(*E))
		
		return G, E

	def __call__(self, X, y, threshold: float = None):
		return self.build(X, y, threshold)