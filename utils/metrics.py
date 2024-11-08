from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from scipy.stats import entropy
import networkx as nx
import numpy as np
import warnings
from typing import Iterable

class Similarity:
	def __init__(self, metric='cosine'):
		__m = ('cosine', 'euclidean')
		assert metric in __m, f'metric should be one of the following: {", ".join(__m)}'

		self.metric = metric

	def calc(self, X):
		A = pairwise_distances(X, X, metric=self.metric)
		np.fill_diagonal(A, float('inf'))
		return A
	
	def __call__(self, X):
		return self.calc(X)


class Uncertainty:
	def __init__(self, uc_type, **kwargs):
		self.nx_flag = hasattr(nx, uc_type) and callable(eval(f'nx.{uc_type}'))
		self.__m = {
			'entropy': self.__entropy, 
			'entropy_e': self.__entropy_e, 
			'density_kmean': self.__density_kmean,
			'nx': self.__nx,
		}
		
		if not self.nx_flag:
			assert uc_type in self.__m, f'type should be one of the following: {", ".join(self.__m)}'
		self.uc_type = uc_type
		self.kwargs = kwargs

	def __entropy(self, X, **kwargs):
		if any(X.reshape(-1) <= 0):
			warnings.warn(f'X values should be bigger than 0')
		return dict(zip(range(len(X)), entropy(X, axis=-1)))
	
	def __entropy_e(self, X, **kwargs):
		model = kwargs.get('model')
		if model is None:
			model = self.kwargs.get('model')
		assert model is not None, 'model should not be None! call help for info.'
		
		X = model(X)
		return self.__entropy(X)
	
	def __density_kmean(self, X, **kwargs):
		"""
		applying the density score function in the paper `Active Learning for Graph Embedding`
		:math:`$\phi_{density}(v_i) = \\frac{1}{1 + ED(Emb_{v_i}, CC_{v_i})}$`

		Args:
			X (torch.Tensor): the input data of shape [n_samples, data_dim].
			keepdims (bool): flag whether the original shape of the data wants to be preserved or not.

		Returns:
			density_scores (dict): a dictionary of the scores such that the keys are the node id and value is the score.
		"""
		n_clusters = kwargs.get('n_clusters')
		if n_clusters is None:
			n_clusters = self.kwargs.get('n_clusters')
		assert n_clusters is not None, 'n_clusters should be passed as an argument! call help for info.'

		kmeans = KMeans(n_clusters=n_clusters).fit(X)
		density_scores = kmeans.transform(X).min(axis=-1, keepdims=kwargs.get('keepdims', False))
		density_scores = 1 / (1 + density_scores)
		return dict(zip(range(len(X)), density_scores))
	
	def __nx(self, G: nx.Graph):
		ret = eval(f'nx.{self.uc_type}')(G)
		return ret if isinstance(ret, dict) else dict(ret)

	def calc(self, X: nx.Graph | np.ndarray, **kwargs):
		if self.nx_flag:
			return self.__m['nx'](X)
		return self.__m[self.uc_type](X, **kwargs)

	def __call__(self, X, **kwargs):
		return self.calc(X, **kwargs)
	
	def help(self):
		print("""
Args:
	X (_type_): the dataset embeddings.
	kwargs: cases:
		- _type=='entropy' no special requirements.
		- _type=='entropy_e' then kwargs should contain `model` parameter.
		- _type=='density_kmean', kwargs should contain `n_clusters` parameter.

Returns:
	_type_: _description_
		""")


class UCAggregator:
	def __init__(self, *uncertainties: Iterable[Uncertainty], aggr: str ='sum', **kwargs):
		self.ucs = uncertainties

		self.__aggrs = {
			'sum': sum,
		}
		assert aggr in self.__aggrs, f'type should be one of the following: {", ".join(self.__aggrs)}'

		self.__aggr = aggr
		self.coef = kwargs.get('coef', np.ones(len(uncertainties)))

	def __get_r(self, X, **kwargs):
		__r = {}
		for uc in self.ucs:
			if hasattr(uc, 'nx_flag') and uc.nx_flag:
				G = kwargs.get('G')
				assert G is not None, 'graph G should be passed as an argument!'
				__r[uc.uc_type] = uc.calc(G)
				continue

			__r[uc.uc_type] = uc.calc(X, **kwargs)
		
		return __r
	
	def __check_assert(self, n_coef, n_ucs):
		assert n_coef == n_ucs, f'number of coefficients and uncertainty measures is different ({n_coef} != {n_ucs})'
	
	def __aggregate(self, r: dict, **kwargs):
		f = self.__aggrs[self.__aggr]
		dicts = list(r.values())
		index = dicts[0].keys()

		coef = kwargs.get('coef', self.coef)
		self.__check_assert(len(coef), len(dicts))

		scores = map(f, zip(*[c * np.array(list(e.values())) for c, e in zip(coef, dicts)]))
		ret = dict(zip(index, scores))
		return ret

	def calc(self, X, **kwargs):
		r = self.__get_r(X, **kwargs)

		lens = list(map(len, r.values()))
		assert min(lens) == max(lens), 'probably the graph has different number of nodes from X!'
		
		return self.__aggregate(r, **kwargs)
	
	def __call__(self, X, **kwargs):
		return self.calc(X, **kwargs)


















