from utils.metrics import Uncertainty, UCAggregator
from typing import Iterable
import numpy as np

class Selector:
	def __init__(self, k: int, uncertainties: UCAggregator, **kwargs):
		assert len(uncertainties) > 0, 'At least one uncertainty measure should be provided!'
		self.ucs = uncertainties
		self.k = k
		self.__AL4GE = kwargs.get('AL4GE')
		self.__coef = kwargs.get('coef')

	def select(self, unlabeled, iteration: int = 1, **kwargs):
		if self.__AL4GE:
			coef = np.random.beta(1, [1. / iteration, 1. / iteration, iteration], size=(3))
			coef = coef / coef.sum()
			kwargs['coef'] = coef
		elif self.__coef:
			assert len(self.ucs) == len(self.__coef), 'Number of coefficients should be the same as number of uncertainties!'
			if kwargs.get('coef') is None:
				coef_betas = np.array([iteration] * len(self.ucs), dtype=float)
				coef_betas[self.__coef] = 1. / iteration

				coef = np.random.beta(1, coef_betas, size=len(self.ucs))

				coef = coef / coef.sum()
				kwargs['coef'] = coef
		
		ucs_scores = self.ucs(unlabeled, **kwargs)
		return sorted(ucs_scores, key=lambda x: ucs_scores[x], reverse=True)[:self.k]

		

