from scipy.stats import entropy

class Uncertainty:
	def __init__(self, type):
		m = ['entropy', 'density']
		assert type in m, f'type should be one of the following: {", ".join(m)}'

		self.type = type

	def __call__(self, X, model):
		return eval(f'__{self.type}')(X, model)

	def __entropy(self, X, model):
		# ENT = (X * torch.log2(X)).sum(dim=-1)
		# ENT = ((ENT - ENT.min()) / (ENT.max() - ENT.min())).numpy()
		return dict(zip(range(len(X)), entropy(model.predict_proba(X), axis=-1)))
	
	def __density(self, X, model):
		pass

