import torch
import numpy as np
import numpy.linalg as nla
from sklearn.metrics import pairwise_distances

class Similarity:
	def __init__(self, metric='cosine'):
		m = ['cosine', 'euclidean']
		assert metric in m, f'metric should be one of the following: {", ".join(m)}'

		self.metric = metric

	def __call__(self, X):
		return torch.tensor(-pairwise_distances(X, X, metric=self.metric))
