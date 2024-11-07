from sklearn.cluster import KMeans
from scipy.stats import entropy
import networkx as nx
import warnings

class Uncertainty:
	def __init__(self, _type: str):
		"""

		Args:
			type (str): type of uncertainty, should be one of the following: 'entropy', 'entropy_e', 'density_kmeans', and any supported networkx centrality function.
			Note that type='entropy' calculates the entropy on the passed arg X, while type='entropy_e' requires parameter model since it calculates the output of the model on X then calculating entropy.
		"""
		self.__nx_flag = hasattr(nx, _type) and callable(eval(f'nx.{_type}'))
		self.__m = {
			'entropy': self.__entropy, 
			'entropy_e': self.__entropy_e, 
			'density_kmean': self.__density_kmean,
			'nx': self.__nx,
		}
		
		if not self.__nx_flag:
			assert _type in self.__m, f'type should be one of the following: {", ".join(self.__m)}'

		self._type = _type

	def __call__(self, X, **kwargs):
		if self.__nx_flag:
			# print(kwargs)
			return self.__m['nx'](X, **kwargs)
		return self.__m[self._type](X, **kwargs)
	
	def __entropy(self, X, **kwargs):
		if any(X.reshape(-1) <= 0):
			warnings.warn(f'X values should be bigger than 0')

		return dict(zip(range(len(X)), entropy(X, axis=-1)))

	def __entropy_e(self, X, **kwargs):
		# ENT = (X * torch.log2(X)).sum(dim=-1)
		# ENT = ((ENT - ENT.min()) / (ENT.max() - ENT.min())).numpy()
		model = kwargs.get('model')
		assert model is not None, 'model should not be None! call help for info.'

		return self.__entropy(model.predict_proba(X))
		# return dict(zip(range(len(X)), entropy(model.predict_proba(X), axis=-1)))
	
	def __density_kmean(self, X, *args, **kwargs):
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
		assert n_clusters is not None, 'n_clusters should be passed as an argument! call help for info.'

		kmeans = KMeans(n_clusters=n_clusters).fit(X)
		density_scores = kmeans.transform(X).min(axis=-1, keepdims=kwargs.get('keepdims', False))
		density_scores = 1 / (1 + density_scores)
		return dict(zip(range(len(X)), density_scores))
	
	def __nx(self, G, **kwargs):
		ret = eval(f'nx.{self._type}')(G)
		if not isinstance(ret, dict):
			ret = dict(zip(range(len(G.nodes)), list(ret)))
		return ret
	
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
			




