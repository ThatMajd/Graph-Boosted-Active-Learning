import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class SimpleGNN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
		super(SimpleGNN, self).__init__()
		
		# Encoder
		self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
		# self.encoder_conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='sum')
		self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)
		
		# Decoder
		# self.decoder_conv1 = GCNConv(hidden_dim, hidden_dim)
		# self.decoder_conv2 = GCNConv(hidden_dim, output_dim)
		self.decoder = Classifier(hidden_dim, out_dim=output_dim)
		
		# Dropout probability
		self.dropout_prob = dropout_prob

	def embed(self, data):
		x, edge_index = data.x, data.edge_index
		x = x.float()
		
		# Encoder forward pass with dropout
		x = self.encoder_conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout_prob, training=self.training)
		
		x = self.encoder_conv2(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout_prob, training=self.training)
		return x

	def forward(self, data):
		x = self.embed(data)
		x = self.decoder(x)
		return x
		# return F.softmax(x, dim=1)

# class GNN(nn.Module):
# 	def __init__(self, in_dim, embed_dim, out_dim):
# 		super(GNN, self).__init__()
# 		self.gnn = gnn.SAGEConv(in_dim, embed_dim, aggr='mean')
# 		self.classifier = Classifier(embed_dim, out_dim)

# 	def forward(self, x, edge_index):
# 		x = self.embed(x, edge_index)
# 		return self.predict(x)
	
# 	def embed(self, x, edge_index):
# 		return self.gnn(x, edge_index)
	
# 	def predict(self, x):
# 		return self.classifier(x)

class Classifier(nn.Module):
	def __init__(self, in_dim=2, out_dim=3):
		super(Classifier, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(in_dim, 64),
			nn.ReLU(),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, out_dim),
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		return self.model(x)
	
	def predict(self, x):
		return self.softmax(self.model(x))



