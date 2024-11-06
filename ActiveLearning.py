import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import trange
import argparse

from selectioncriterion import SelectionCriterion



class ActiveLearningPipeline:
	def __init__(self,
				 dataset,
				 classifying_model,
				 selection_criterion,
				 iterations,
				 budget_per_iter,
				 ):
		
		# Parse data correctly
		self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
		self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
		self.unlabeled_set, self.unlabeled_labels = dataset['available_pool_samples'], dataset['available_pool_labels']


		self.iterations = iterations
		self.budget_per_iter = budget_per_iter
		self.selection_criterion = selection_criterion

		if classifying_model == "LogisticRegression":
			self.cls_model = LogisticRegression()
		else:
			print("Call Mahmod")

		
		

	def run_pipeline(self):
		"""
		Run the active learning pipeline
		"""
		accuracy_scores = []
		iterations_progress = trange(self.iterations)

		for iteration in iterations_progress:			
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
		return self.cls_model.fit(self.train_samples, self.train_labels)


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