from utils.Uncertainty import Uncertainty
from torch.utils.data import Dataset
import torch

class SelectionCriterion:
    def __init__(self, criterions_dep: dict, budget_per_iter, weighted=False):
        """
        criterions_dep = 
        {
            "entropy": {"model": model},
            "pagerank": {"graph": G}
        }
        """
        self.budget_per_iter = budget_per_iter
        self.weighted = weighted
        self.crit_dicts = {crit: (Uncertainty(crit), criterions_dep[crit]) for crit in criterions_dep.keys()}
    

    def select(self, unlabeled: Dataset, labeled: Dataset, iteration: int = 0):
        self.crit_scores = self._calc_crits(unlabeled, labeled)

        if self.weighted:
            # TODO
            # add weighting logic here
            pass
        else:
            weights = len(self.crit_dicts) * [1]

        final_scores = self.sum_dicts(*self.crit_dicts, coef=weights)
        return sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:self.budget_per_iter]


    def _calc_crits(self, unlabeled, labeled):
        crit_scores = dict()

        # TODO
        # Decided later 
        X = torch.tensor([0])

        for crit, (func_crit, func_dep) in self.crit_dicts.items():
            # for every criterion provide it with the data and its dependancies and then calculate it and store it
            crit_scores[crit] = func_crit(X, func_dep)
        
        return crit_scores
        
    def sum_dicts(self, *dicts, coef=None):
        if coef is None:
            coef = [1 / len(dicts)] * len(dicts)

        s = {}
        for k in dicts[0].keys():
            # s[k] = a[k] + b.get(k, 0)
            s[k] = sum([coef[i] * e.get(k, 0) for i, e in enumerate(dicts)])
        return s