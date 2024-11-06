from torch import nn, optim

class Model:
	def __init__(self, model, **kwargs):
		self.model = model

	def fit(self, X, y):

		if not isinstance(self.model, nn.Module):
			return

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=lr)

		x = self.embed_gnn(gnn_model)
		o = gnn_model.predict(x)
		loss = criterion(o, self.D_labels[self.gnn_labeled_index])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()		

		model = self.classifier_class()
		model = model.fit(self.train_samples, self.train_labels)
		
		return model, gnn_model
		