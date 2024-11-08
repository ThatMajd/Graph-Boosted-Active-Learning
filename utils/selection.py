from utils.metrics import Uncertainty, UCAggregator
from typing import Iterable
import numpy as np

class Selector:
	def __init__(self, k: int, *uncertainties: Iterable[Uncertainty | UCAggregator], **kwargs):
		assert len(uncertainties) > 0, 'At least one uncertainty measure should be provided!'
		self.ucs = uncertainties[0] if len(uncertainties) == 1 else UCAggregator(*uncertainties)
		self.k = k
		self.__AL4GE = kwargs.get('AL4GE')

	def select(self, unlabeled, iteration: int = 1, **kwargs):
		if self.__AL4GE:
			coef = np.random.beta(1, [1. / iteration, 1. / iteration, iteration], size=(3))
			coef = coef / coef.sum()
			kwargs['coef'] = coef
		
		ucs_scores = self.ucs(unlabeled, **kwargs)
		return sorted(ucs_scores, key=lambda x: ucs_scores[x], reverse=True)[:self.k]

		

