import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.7):
        super(SimpleGNN, self).__init__()
        
        # Encoder
        self.encoder_conv1 = GCNConv(input_dim, hidden_dim)
        self.encoder_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Decoder
        self.decoder_conv1 = GCNConv(hidden_dim, hidden_dim)
        self.decoder_conv2 = GCNConv(hidden_dim, output_dim)
        
        # Dropout probability
        self.dropout_prob = dropout_prob

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        
        # Encoder forward pass with dropout
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        x = self.encoder_conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        # Decoder forward pass with dropout
        x = self.decoder_conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)
        
        x = self.decoder_conv2(x, edge_index)
        return F.softmax(x, dim=1)