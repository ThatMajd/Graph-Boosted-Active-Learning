from utils.metrics import Similarity, Uncertainty, UCAggregator
from utils.builders import GraphBuilder
from utils.wrappers import ModelWrapper
from utils.selection import Selector
import matplotlib.pyplot as plt
from tqdm import trange
import networkx as nx
import numpy as np
from utils.gnn_models import SimpleGNN
import torch
import wandb


class GAL:
	def __init__(self,
				 dataset,
				 classifier,
				 budget_per_iter: int,
				 uncertainty_measures,
				 **kwargs):
		"""_summary_

		Args:
			dataset (_type_): _description_
			classifier (_type_): _description_
			budget_per_iter (int): _description_
			labels (Itrable[int] | int): either the set of labels or the number of labels, if labels is int then the labels are assumed to be range(labels).
			coef (Itrable[bool | float | int]): the coefficients of the uncertainties, if itrable of bools then it element that's True means that the cofficient the corresponds to this element is increasing beta RV., False means decreasing (Similar to AL4GE).
		"""

		self.wandb = kwargs.get("use_wandb", False)

		self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
		self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
		self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset[
			'available_pool_labels']

		self.sim_metric = kwargs.get('sim_metric', 'euclidean')
		self.similarity = Similarity(self.sim_metric)
		self.graph_builder = GraphBuilder(self.similarity)

		if kwargs.get('AL4GE'):
			self.uc_aggr = UCAggregator(
				Uncertainty('entropy_e'),
				Uncertainty('density_kmean'),
				Uncertainty('pagerank'),
			)
			self.nx_flag = True
			self.selector = Selector(budget_per_iter, self.uc_aggr, AL4GE=True)
		else:
			self.uc_aggr = UCAggregator(*[Uncertainty(e) for e in uncertainty_measures], coef=kwargs.get('coef'))
			self.nx_flag = any(e.nx_flag for e in self.uc_aggr.ucs)
			self.selector = Selector(budget_per_iter, self.uc_aggr, coef=kwargs.get('coef'))

		self.iterations = kwargs.get('iterations', 10)
		self.budget_per_iter = budget_per_iter
		self.classifier = ModelWrapper(classifier)
		self.n_clusters = kwargs.get('n_clusters', 2)
		self.quantile = kwargs.get('quantile', .2)

		self.labels = kwargs.get('labels')

		assert self.iterations * self.budget_per_iter <= len(dataset["available_pool_labels"]), f"Not enough samples in pool ({self.iterations * self.budget_per_iter} > {len(dataset['available_pool_labels'])})"

		self.use_gnn = kwargs.get("use_gnn", False)
		if self.use_gnn:
			input_dim = kwargs.get("input_dim", 3)
			hidden_dim = kwargs.get("hidden_dim", 16)
			output_dim = kwargs.get("output_dim", 4)

			self.epochs = kwargs.get("gnn_epochs", 5)
			self.gnn_model = SimpleGNN(input_dim, hidden_dim, output_dim)
			self.optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
			self.criterion = torch.nn.CrossEntropyLoss()

			self.eval_graph = self.graph_builder(
				self.test_samples,
				self.test_labels,
				self.quantile,
				pytorch=True
			)
			self.gnn_unlabeled_idx = np.arange(len(self.available_pool_labels))
			self.gnn_labeled_idx = np.arange(len(self.train_labels))
			self.init_labeled_size = len(self.train_samples)
			self.train_graph_include_test = kwargs.get('train_graph_include_test', False)
			self.train_graph = self.create_train_graph()

	def create_train_graph(self, pytorch=True):
		train_x, train_y = self.train_samples, self.train_labels
		pool_x, pool_y = self.available_pool_samples, self.available_pool_labels

		if self.train_graph_include_test:
			test_x, test_y = self.test_samples, self.test_labels

			data_x = np.concatenate([train_x, pool_x, test_x])
			data_y = np.concatenate([train_y, pool_y, test_y])
		else:
			data_x = np.concatenate([train_x, pool_x])
			data_y = np.concatenate([train_y, pool_y])

		# Create a pytorch Graph
		train_graph = self.graph_builder(data_x, data_y, self.quantile, pytorch=pytorch)
		mask_len = len(self.train_samples) + len(self.available_pool_samples) + (len(self.test_samples) if self.train_graph_include_test else 0)
		pool_mask = np.array(
			[False] * (mask_len))
		pool_mask[self.gnn_unlabeled_idx + self.init_labeled_size] = True
		train_graph.pool_mask = pool_mask

		return train_graph

	def update_indices(self, selection_indices):
		if self.use_gnn:
			self.gnn_labeled_idx = np.concatenate(
				(self.gnn_labeled_idx, [self.gnn_unlabeled_idx[e] + self.init_labeled_size for e in selection_indices]))
			self.gnn_unlabeled_idx = np.delete(self.gnn_unlabeled_idx, selection_indices)

			
		self.train_samples = np.vstack((self.train_samples, self.available_pool_samples[selection_indices]))
		self.train_labels = np.hstack((self.train_labels, self.available_pool_labels[selection_indices]))
		# Shuffle the train set to avoid bias caused by the order of the samples
		idx = np.random.permutation(len(self.train_labels))
		self.train_samples = self.train_samples[idx]
		self.train_labels = self.train_labels[idx]
		self.available_pool_samples = np.delete(self.available_pool_samples, selection_indices, axis=0)
		self.available_pool_labels = np.delete(self.available_pool_labels, selection_indices)

	def _evaluate_model(self, trained_model):
		"""
		Evaluate the model
		:param trained_model: the trained model
		:return: the accuracy of the model on the test set
		"""
		preds = trained_model.predict(self.test_samples)
		return round(np.mean(preds == self.test_labels), 3)

	def _train_gnn(self):
		mask_len = len(self.train_samples) + len(self.available_pool_samples) + (len(self.test_samples) if self.train_graph_include_test else 0)
		train_mask = np.array(
			[False] * (mask_len))
		pool_mask = np.array(
			[False] * (mask_len))

		train_mask[self.gnn_labeled_idx] = True
		pool_mask[self.gnn_unlabeled_idx + self.init_labeled_size] = True

		self.train_graph.train_mask = train_mask
		self.train_graph.pool_mask = pool_mask

		# optimizer = self.optimizer
		# criterion = self.criterion

		self.gnn_model.train()

		for epoch in range(self.epochs):
			out = self.gnn_model(self.train_graph)
			loss = self.criterion(out[self.train_graph.train_mask],
								  self.train_graph.y[self.train_graph.train_mask])  # Use only train mask nodes

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			_, preds = out[self.train_graph.train_mask].max(dim=1)  # Get the index of the max log-probability
			correct = preds.eq(self.train_graph.y[self.train_graph.train_mask]).sum().item()
			accuracy = correct / self.train_graph.train_mask.sum().item()  # Compute accuracy as a ratio

		# Print training loss and accuracy
		# if epoch + 1 == self.epochs:
		#     print(f'[GNN] - Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')

		return accuracy

	def _evaluate_gnn(self):
		if self.train_graph_include_test:
			mask_len = len(self.train_samples) + len(self.available_pool_samples) + len(self.test_samples)
			eval_graph = self.train_graph
		else:
			mask_len = len(self.test_samples)
			eval_graph = self.eval_graph

		test_mask = np.array(
			[False] * (mask_len))
		test_mask[-len(self.test_samples)::] = True

		self.gnn_model.eval()

		with torch.no_grad():
			out = self.gnn_model(eval_graph)
		out = out[test_mask]

		# Calculate accuracy on test data
		_, preds = out.max(dim=1)  # Get the predicted classes
		correct = preds.eq(torch.Tensor(self.test_labels)).sum().item()
		accuracy = correct / self.test_labels.shape[0]  # Total number of test nodes

		# print(f"[GNN] - Test Accuracy: {accuracy:.4f}")
		return out, accuracy

	def run(self, **kwargs):
		"""

		Returns:
			accuracy_scores: {'aggr': []} if use_gnn else {'aggr': [], 'LR': [], 'GNN': []}
		"""
		accuracy_scores = {'aggr': []}
		if self.use_gnn:
			accuracy_scores = {'aggr': [], 'LR': [], 'GNN': []}

		iterations_progress = trange(self.iterations)

		n_clusters = kwargs.get('n_clusters', self.n_clusters)
		graph_flag = kwargs.get('plot', False)

		for iter_idx in iterations_progress:
			if len(self.available_pool_labels) == 0:
				break
			self.classifier.fit(self.train_samples, self.train_labels)

			if self.use_gnn:
				gnn_train_acc = self._train_gnn()

			nx_G = self.graph_builder.build(
				self.available_pool_samples,
				self.available_pool_labels,
				self.quantile)
			if graph_flag:
				pos = dict(zip(range(len(self.available_pool_samples)), self.available_pool_samples[:, [0, 1]]))
				nx.draw(nx_G, pos=pos, with_labels=True)
				plt.show()
			selection_indices = self.selector.select(self.available_pool_samples,
													 iter_idx + 1,
													 n_clusters=n_clusters,
													 G=nx_G,
													 GNN=self.gnn_model if self.use_gnn else None,
													 GNN_graph=self.train_graph if self.use_gnn else None,
													 model=self.classifier,
													 coef=kwargs.get('coef'),
													 labels=kwargs.get('labels', self.labels))
			self.update_indices(selection_indices)
			# accuracy = self._evaluate_model(self.classifier)

			cls_out = self.classifier(self.test_samples)
			LR_acc = self._evaluate_model(self.classifier)

			if self.use_gnn:
				gnn_out, gnn_test_acc = self._evaluate_gnn()

				assert cls_out.shape == gnn_out.shape

				# Aggregartion Function
				final_preds = np.maximum(cls_out, gnn_out.numpy())

				labels = np.argmax(final_preds, axis=1)

				accuracy = (labels == self.test_labels).sum() / cls_out.shape[0]
				acc_LR = self._evaluate_model(self.classifier)

				accuracy_scores['aggr'].append(accuracy)
				accuracy_scores['GNN'].append(gnn_test_acc)
				accuracy_scores['LR'].append(acc_LR)

				LOG = {"GAL_Iteration": iter_idx, "GAL_Accuracy": accuracy, "GAL_LR test acc": LR_acc, "GNN Train Acc": gnn_train_acc,
					   "GNN Test Acc": gnn_test_acc}
				iterations_progress.set_postfix(LOG)
				


			else:
				accuracy = self._evaluate_model(self.classifier)
				accuracy_scores['aggr'].append(accuracy)

				LOG = {"GAL_Iteration": iter_idx, "GAL_Accuracy": accuracy, "GAL_LR test acc": LR_acc}
				iterations_progress.set_postfix(LOG)


		return accuracy_scores
