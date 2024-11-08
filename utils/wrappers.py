from torch import nn
from torch import optim

class ModelWrapper:
	def __init__(self, model):
		self.model = model
		self.nn_flag = isinstance(self.model, nn.Module)

	def fit(self, X, y, **kwargs):
		if not self.nn_flag:
			self.model = self.model.fit(X, y)
			return

		criterion = kwargs.get('criterion', nn.CrossEntropyLoss)()
		optimizer = kwargs.get('optim', optim.Adam)(self.model.parameters(), **kwargs)

		o = self.predict(X)
		loss = criterion(o, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()		

	def __call__(self, X):
		if self.nn_flag:
			return self.model(X)
		
		return self.model.predict_proba(X)
	
	def predict(self, X):
		if self.nn_flag:
			return self.model(X).argmax(dim=-1)
		
		return self.model.predict(X)

