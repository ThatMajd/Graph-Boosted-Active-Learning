import torch
from torch import nn
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import entropy
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import networkx as nx
import numpy.linalg as nla
from sklearn.metrics.pairwise import pairwise_distances
import torch_geometric.nn as gnn
from torch import optim 
import torch.nn.functional as F

class GAL:
	def __init__(self,
			  	dataset,
				selection_criterion,
				iterations,
				budget_per_iter,
				train_limit,
				# graph_maker,
				thresh,
				sim_metric,
				classifier_class,
				gnn_class,
				gnn_emb_dim=5,
				out_dim=4):
		
		self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
		self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
		self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset['available_pool_labels']
		
		self.iterations = iterations
		self.budget_per_iter = budget_per_iter if budget_per_iter is not None else 10
		self.train_limit = train_limit
		self.selection_criterion = selection_criterion
		self.classifier_class = classifier_class
		self.gnn_class = gnn_class
		# self.graph_maker = graph_maker
		self.thresh = thresh if thresh is not None else .8

		# self.sim_metric = lambda x: pairwise_distances(x, x)
		self.sim_metric = GAL.cosine_sim_metric
		# self.sim_metric = sim_metric

		# prep for GNN
		self.D_samples = np.concatenate([self.train_samples, self.available_pool_samples], axis=0)
		self.D_labels = np.concatenate([self.train_labels, self.available_pool_labels], axis=0)
		self.gnn_labeled_index = list(range(len(self.train_samples)))
		self.E_gnn = self.construct_edges(self.sim_metric(self.D_samples))
		self.E_gnn = torch.tensor(self.E_gnn)
		self.D_samples = torch.Tensor(self.D_samples)
		self.D_labels = torch.tensor(self.D_labels)
		self.gnn_emb_dim = gnn_emb_dim
		self.out_dim = out_dim


		# assert callable(self.classifier_class), 'classifier_class should be callable.'
		# assert hasattr(self.classifier_class(), 'fit') and callable(self.classifier_class().fit), 'Object returned from classifier_class() should contain method fit(x_train, y_train).'

		# self.tensor_flag = isinstance(gnn, nn.Module)

	def cosine_sim_metric(X):
		X = X / nla.norm(X, axis=-1).reshape(-1, 1)
		cos_sim_mat = (X @ X.T)
		np.fill_diagonal(cos_sim_mat, 0)
		cos_sim_mat = np.absolute(cos_sim_mat)
		return cos_sim_mat

	def sim_mat(self, X):
		return self.sim_metric(X)
	
	def construct_edges(self, A):
		return np.vstack(np.where(A > self.thresh))

	def construct_graph(self, A, nodes):
		E = self.construct_edges(A)

		G = nx.Graph()
		G.add_nodes_from(range(len(nodes)))
		G.add_edges_from(zip(*E))
		return G
	
	def embed_gnn(self, gnn_model):
		return gnn_model.embed(self.D_samples, self.E_gnn)[self.gnn_labeled_index]
	
	def heterogeneous_edge_influence(graph, node_labels, label_diversity_weight=0.5):
		"""
		Calculate the influence score for each node based on connections to labeled nodes and label diversity.
		
		:param graph: The networkx graph.
		:param node_labels: Dictionary of node IDs and their labels.
		:param label_diversity_weight: The weight to assign to label diversity in the final score.
		
		:return: A dictionary of influence scores for each node.
		"""
		influence_scores = defaultdict(float)
		
		for node in graph.nodes:
			if node in node_labels:
				continue  # Skip labeled nodes, focus on unlabeled ones
			
			# Get neighbors and their labels
			neighbors = list(graph.neighbors(node))
			labeled_neighbors = [neighbor for neighbor in neighbors if neighbor in node_labels]
			
			# Count how many labeled nodes it's connected to
			num_labeled_neighbors = len(labeled_neighbors)
			
			# Measure label diversity
			label_count = defaultdict(int)
			for neighbor in labeled_neighbors:
				label = node_labels[neighbor]
				label_count[label] += 1
			
			# Calculate diversity as the number of unique labels connected to this node
			label_diversity = len(label_count)
			
			# Combine influence score based on connection count and label diversity
			influence_score = (1 - label_diversity_weight) * num_labeled_neighbors + label_diversity_weight * label_diversity
			
			influence_scores[node] = influence_score
		
		return influence_scores
	
	def entropy(self, X, model):
		if not isinstance(X, torch.Tensor):
			X = torch.Tensor(X)
		ENT = (X * torch.log2(X)).sum(dim=-1)
		ENT = ((ENT - ENT.min()) / (ENT.max() - ENT.min())).numpy()
		return dict(zip(range(len(X)), entropy(model.predict_proba(X))))
	
	def sum_dicts(self, a, b):
		s = {}
		for k in a.keys():
			s[k] = a[k] + b.get(k, 0) 
		return s
	
	def uncertainty_score(self, G, model):
		try:
			return self.sum_dicts(nx.pagerank(G), self.entropy(self.available_pool_samples, model))
		except:
			print('error uncertainty metric')
			return nx.eigenvector_centrality(G)
		# return nx.pagerank(G)
	
	def select_points(self, G, model):
		R = self.uncertainty_score(G, model)
		return sorted(R, key=lambda x: R[x], reverse=True)[:self.budget_per_iter]
	
	
	def label_update(self, new_selected_samples):
		"""
		Update the indices such that the new selected samples are added to the train set and removed from the available pool
		"""
		self.gnn_labeled_index.extend([e + len(self.train_samples) for e in new_selected_samples])

		self.train_samples = np.vstack((self.train_samples, self.available_pool_samples[new_selected_samples]))
		self.train_labels = np.hstack((self.train_labels, self.available_pool_labels[new_selected_samples]))
		# shuffle the train set to avoid bias caused by the order of the samples
		idx = np.random.permutation(len(self.train_labels))
		self.train_samples = self.train_samples[idx]
		self.train_labels = self.train_labels[idx]
		self.available_pool_samples = np.delete(self.available_pool_samples, new_selected_samples, axis=0)
		self.available_pool_labels = np.delete(self.available_pool_labels, new_selected_samples)

	def _train_model(self, lr=.001):
		gnn_model = self.gnn_class(self.D_samples[0].shape[0], self.gnn_emb_dim, self.out_dim)
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(gnn_model.parameters(), lr=lr)

		x = self.embed_gnn(gnn_model)
		o = gnn_model.predict(x)
		loss = criterion(o, self.D_labels[self.gnn_labeled_index])

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()		

		model = self.classifier_class()
		model = model.fit(self.train_samples, self.train_labels)
		
		return model, gnn_model

	def run_pipeline(self):
		accuracy_scores = []
		E_test = self.construct_edges(self.sim_mat(self.test_samples))
		E_test = torch.tensor(E_test)

		for iteration in range(self.iterations):
			assert len(self.train_samples) < self.train_limit, f'The train set is larger than {self.train_limit} samples'
			print_str = f'Iteration {iteration + 1}/{self.iterations}'
			print(print_str)
			model, gnn_model = self._train_model()


			A = self.sim_mat(self.train_samples)
			G = self.construct_graph(A, self.available_pool_samples)
			U_idx = self.select_points(G, model)
			self.label_update(U_idx)
			# L_gnn = self.embed_gnn(gnn_model)
			gal = GALClassifier(gnn_model, model, Classifier(2 * self.out_dim, self.out_dim))

			accuracy = self._evaluate_model(gal, E_test)
			accuracy_scores.append(accuracy)
			print(f'Accuracy: {accuracy}')
			print('-' * len(print_str))
		return accuracy_scores
	
	def _evaluate_model(self, trained_model, E_test):
		"""
		Evaluate the model
		:param trained_model: the trained model
		:return: the accuracy of the model on the test set
		"""
		preds = trained_model(torch.Tensor(self.test_samples), E_test).argmax(dim=-1)
		return round(np.mean(preds.numpy() == self.test_labels), 3)


class GALClassifier:
	def __init__(self, gnn, model, classifier):
		self.classifier = classifier
		self.gnn = gnn
		self.model = model

	def __call__(self, x, edge_index):
		gnn_pred = self.gnn(x, edge_index)
		if isinstance(self.model, nn.Module):
			model_pred = self.model(x)
		else:
			model_pred = torch.Tensor(self.model.predict_proba(x))
		
		# x = gnn_pred + model_pred
		x = torch.maximum(gnn_pred, model_pred)
		return F.softmax(x, dim=-1)
		# x = torch.cat([gnn_pred, model_pred], axis=1).type(torch.float)
		# return self.classifier(x)


class ActiveLearningPipeline:
	def __init__(self,
				 dataset,
				 selection_criterion,
				 iterations,
				 budget_per_iter,
				 train_limit):
		self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
		self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
		self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset['available_pool_labels']
		self.iterations = iterations
		self.budget_per_iter = budget_per_iter
		self.train_limit = train_limit
		self.selection_criterion = selection_criterion
		

	def run_pipeline(self):
		"""
		Run the active learning pipeline
		"""
		accuracy_scores = []
		for iteration in range(self.iterations):
			if len(self.train_samples) > self.train_limit:
				raise ValueError(f'The train set is larger than {self.train_limit} samples')
			print(f'Iteration {iteration + 1}/{self.iterations}')
			trained_model = self._train_model()
			if self.selection_criterion == 'random':
				new_selected_samples = self._random_sampling()
			else:
				new_selected_samples = self._custom_sampling(trained_model)
			self._update_indices(new_selected_samples)
			accuracy = self._evaluate_model(trained_model)
			accuracy_scores.append(accuracy)
			print(f'Accuracy: {accuracy}')
			print('----------------------------------------')
		return accuracy_scores


	def _train_model(self):
		"""
		Train the model
		"""
		model = LogisticRegression()
		return model.fit(self.train_samples, self.train_labels)

	def _random_sampling(self):
		return np.random.choice(range(self.available_pool_samples.shape[0]), self.budget_per_iter)

	def _custom_sampling(self, trained_model):
		# TODO: Implement the custom samplingR
		
		# # entropy
		probabilities = trained_model.predict_proba(self.available_pool_samples)
		uncertainties = entropy(probabilities, axis=1)
		select_indices = np.argpartition(uncertainties, -len(self.available_pool_labels) // 3)[-len(self.available_pool_labels) // 3:]
		# select_indices = np.argpartition(uncertainties, -self.budget_per_iter)[-self.budget_per_iter:]
		# return select_indices
		
		# Diversity 
		def top_k(arr, k):
			return np.argpartition(arr, -k)[-k:]
		
		sub_ava = self.available_pool_samples[select_indices]
		dists = pairwise_distances(sub_ava, self.train_samples)

		min_distances = dists.min(axis=1)

		top_indices = top_k(min_distances, min(self.budget_per_iter, len(sub_ava)))
		return top_indices

	def _update_indices(self, new_selected_samples):
		"""
		Update the indices such that the new selected samples are added to the train set and removed from the available pool
		"""
		self.train_samples = np.vstack((self.train_samples, self.available_pool_samples[new_selected_samples]))
		self.train_labels = np.hstack((self.train_labels, self.available_pool_labels[new_selected_samples]))
		# shuffle the train set to avoid bias caused by the order of the samples
		idx = np.random.permutation(len(self.train_labels))
		self.train_samples = self.train_samples[idx]
		self.train_labels = self.train_labels[idx]
		self.available_pool_samples = np.delete(self.available_pool_samples, new_selected_samples, axis=0)
		self.available_pool_labels = np.delete(self.available_pool_labels, new_selected_samples)

	def _evaluate_model(self, trained_model):
		"""
		Evaluate the model
		:param trained_model: the trained model
		:return: the accuracy of the model on the test set
		"""
		preds = trained_model.predict(self.test_samples)
		return round(np.mean(preds == self.test_labels), 3)

def generate_plot(accuracy_scores_dict):
	"""
	Generates a plot
	"""
	plt.figure(figsize=(15, 5))
	for criterion, accuracy_scores in accuracy_scores_dict.items():
		plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)
	plt.xlabel('Iterations')
	plt.ylabel('Accuracy')
	plt.xticks(range(1, len(accuracy_scores) + 1))
	plt.legend()
	plt.show()

class GNN(nn.Module):
	def __init__(self, in_dim, embed_dim, out_dim):
		super(GNN, self).__init__()
		self.gnn = gnn.SAGEConv(in_dim, embed_dim, aggr='mean')
		self.classifier = Classifier(embed_dim, out_dim)

	def forward(self, x, edge_index):
		x = self.embed(x, edge_index)
		return self.predict(x)
	
	def embed(self, x, edge_index):
		return self.gnn(x, edge_index)
	
	def predict(self, x):
		return self.classifier(x)

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



