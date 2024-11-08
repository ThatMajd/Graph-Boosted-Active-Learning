from utils.metrics import Similarity, Uncertainty, UCAggregator
from utils.builders import GraphBuilder
from utils.wrappers import ModelWrapper
from utils.selection import Selector
import matplotlib.pyplot as plt
from tqdm import trange
import networkx as nx
import numpy as np


class GAL:
	def __init__(self,
			  	 dataset,
				 classifier,
				 budget_per_iter: int,
				 iterations: int = 10,
				 threshold: float = 1,
				 sim_metric: str = 'euclidean',
				 *uncertainty_measures,
				 **kwargs):
		
		self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
		self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
		self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset['available_pool_labels']

		self.similarity = Similarity(sim_metric)
		self.graph_builder = GraphBuilder(self.similarity)

		if kwargs.get('AL4GE'):
			self.ucs = UCAggregator(
				Uncertainty('entropy_e'),
				Uncertainty('density_kmean'),
				Uncertainty('pagerank'),
			)
			self.nx_flag = True
			self.selector = Selector(budget_per_iter, self.ucs, AL4GE=True)
		else:
			self.ucs = UCAggregator(Uncertainty(e) for e in uncertainty_measures)
			self.nx_flag = any(e.nx_flag for e in self.ucs.ucs)
			self.selector = Selector(budget_per_iter, self.ucs)

		self.iterations = iterations
		self.budget_per_iter = budget_per_iter
		self.classifier = ModelWrapper(classifier)
		self.n_clusters = kwargs.get('n_clusters', 2)
		self.threshold = threshold

	def update_indices(self, selection_indices):
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

	def run(self, **kwargs):
		accuracy_scores = []
		iterations_progress = trange(self.iterations)

		n_clusters = kwargs.get('n_clusters', self.n_clusters)
		graph_flag = kwargs.get('plot', False)

		for iter in iterations_progress:
			self.classifier.fit(self.train_samples, self.train_labels)
			G, _ = self.graph_builder.build(
					self.available_pool_samples,
					self.available_pool_labels,
					self.threshold)
			if graph_flag:
				pos = dict(zip(range(len(self.available_pool_samples)), self.available_pool_samples[:, [0, 1]]))
				nx.draw(G, pos=pos, with_labels=True)
				plt.show()
			selection_indices = self.selector.select(self.available_pool_samples, 
											iter+1, 
											n_clusters=n_clusters,
											G=G,
											model=self.classifier)
			self.update_indices(selection_indices)
			accuracy = self._evaluate_model(self.classifier)
			accuracy_scores.append(accuracy)

			iterations_progress.set_postfix({"Accuracy": accuracy})
		
		return accuracy_scores


