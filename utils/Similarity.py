import torch
import numpy as np
import numpy.linalg as nla
from sklearn.metrics import pairwise_distances

class Similarity:
	def __init__(self, metric='cosine'):
		"""
		Similarity class

		Args:
			metric (str, optional): the similarity metric that would be used, Note that 'cosine' metric which is the default is being calculated as `1 - CosineSimilarity`. Defaults to 'cosine'.
		"""
		__m = ('cosine', 'euclidean')
		assert metric in __m, f'metric should be one of the following: {", ".join(__m)}'

		self.metric = metric

	def __call__(self, X):
		return torch.Tensor(pairwise_distances(X, X, metric=self.metric)).fill_diagonal_(float('inf'))
