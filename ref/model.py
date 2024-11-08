from torch import nn, optim

class Model:
	def __init__(self, model, **kwargs):
		self.model = model
		self.nn_flag = isinstance(self.model, nn.Module)
		self.lr = kwargs.get('lr', .001)

	def fit(self, X, y):

		if not self.nn_flag:
			self.model = self.model.fit(X, y)

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

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
	




		