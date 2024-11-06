from utils.Uncertainty import Uncertainty
from GraphBuilder import GraphBuilder
from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import torch

class SelectionCriterion:
	def __init__(self,
				 *criterions,
				 budget_per_iter,
				 weighted=False,
				 similarity_metric='cosine',
				 threshold=.8,
				 model=None,
				 **kwargs):

		self.budget_per_iter = budget_per_iter
		self.weighted = weighted

		self.graph_builder = GraphBuilder(similarity_metric=similarity_metric, threshold=threshold)

		# select params
		self.model = model
		self.G = kwargs.get('G')

		self.uncertainty_dicts = dict()

		for crit_type in criterions:
			print(crit_type)
			self.uncertainty_dicts[crit_type] = Uncertainty(crit_type)

	def __nx_attr(self, func):
		return hasattr(nx, func) and callable(eval(f'nx.{func}'))

	def select(self, unlabeled: Dataset, labeled: Dataset, iteration: int = 1, **kwargs):
		_, G = self.graph_builder(unlabeled)
		self.G = G
		self.uncertainty_scores = self._calc_uncertainties(unlabeled, labeled, **kwargs, model=self.model, G=G)

		weights = None
		if self.weighted:
			weights = np.random.beta(1, [1. / iteration, 1. / iteration, iteration], size=(3))
			weights = weights / weights.sum()
			
		final_scores = self.sum_dicts(self.uncertainty_scores, coef=weights)
		print(final_scores)
		return sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:self.budget_per_iter]


	def _calc_uncertainties(self, unlabeled, labeled, **kwargs):
		scores = dict()

		for uncertainty_type, uncertainty in self.uncertainty_dicts.items():
			if self.__nx_attr(uncertainty_type):
				scores[uncertainty_type] = uncertainty(kwargs.get('G'))
				continue

			scores[uncertainty_type] = uncertainty(unlabeled, **kwargs)
		
		return scores
		
	def sum_dicts(self, scores_dicts, coef=None):
		if coef is None:
			coef = [1 / len(scores_dicts)] * len(scores_dicts)

		s = {}
		for i, sub_scores in enumerate(scores_dicts.values()):
			for idx, score in sub_scores.items():
				s[idx] = s.get(idx, 0) + coef[i] * score
		return s