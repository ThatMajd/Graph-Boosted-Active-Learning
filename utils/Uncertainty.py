from scipy.stats import entropy
import networkx as nx

class Uncertainty:
	def __init__(self, _type: str):
		"""

		Args:
			type (str): type of uncertainty, should be one of the following: ('entropy', 'entropy_e', 'density')
			such that type='entropy' calculates the entropy on the passed arg X, while type='entropy_e' 
		"""
		self.__nx_flag = hasattr(nx, _type) and callable(eval(f'nx.{_type}'))
		__m = ('entropy', 'entropy_e', 'density')
		
		if not self.__nx_flag:
			assert _type in __m, f'type should be one of the following: {", ".join(__m)}'

		self._type = _type

	def __call__(self, X, **kwargs):
		if self.__nx_flag:
			return self.__nx(X, **kwargs)
		return eval(f'__{self._type}')(X, **kwargs)
	
	def __entropy(self, X, **kwargs):
		return dict(zip(range(len(X)), entropy(X, axis=-1)))

	def __entropy_e(self, X, **kwargs):
		# ENT = (X * torch.log2(X)).sum(dim=-1)
		# ENT = ((ENT - ENT.min()) / (ENT.max() - ENT.min())).numpy()
		model = kwargs.get('model')
		if model is None:
			raise Exception('model should not be None')
		return self.__entropy(model.predict_proba(X))
		# return dict(zip(range(len(X)), entropy(model.predict_proba(X), axis=-1)))
	
	def __density(self, X, *args, **kwargs):
		"""
		applying the density score function in the paper `Active Learning for Graph Embedding`
		:math:`$\phi_{density}(v_i) = \\frac{1}{1 + ED(Emb_{v_i}, CC_{v_i})}$`

		Args:
			X (torch.Tensor): the input data of shape [n_samples, data_dim].
			keepdims (bool): flag whether the original shape of the data wants to be preserved or not.

		Returns:
			density_scores (dict): a dictionary of the scores such that the keys are the node id and value is the score.
		"""
		self.kmeans = self.kmeans.fit(X)
		density_scores = self.kmeans.transform(X).min(axis=-1, keepdims=kwargs.get('keepdims', True))
		density_scores = 1 / (1 + density_scores)
		return dict(zip(range(len(X)), density_scores))
	
	def __nx(self, G, **kwargs):
		return eval(f'nx.{self._type}')(G)





