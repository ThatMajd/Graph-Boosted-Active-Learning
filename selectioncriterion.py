from utils.Uncertainty import Uncertainty
from torch.utils.data import Dataset
from GraphBuilder import GraphBuilder
import networkx as nx
import torch

class SelectionCriterion:
    def __init__(self, criterions: list, budget_per_iter, weighted=False, **kwargs):
        """
        criterions_dep = 
        {
            "entropy": {"model": model},
            "pagerank": {"graph": G},
            "density": {}
        }
        """
        model = kwargs["model"]
        metric = kwargs["metric"]
        threshold = kwargs.get("threshold")

        self.criterions_dep = {
            "entropy": {"model": kwargs["model"]},
            "nx": {"graph": None},
            "density": {},
        }


        self.budget_per_iter = budget_per_iter
        self.weighted = weighted

        self.graph_builder = GraphBuilder(metric=metric, threshold=threshold)
        self.model = model

        self.crit_dicts = dict()

        for crit in criterions:
            print(crit)
            # if self.__nx_func(crit):
            #     self.crit_dicts[crit] = {crit: (Uncertainty(crit))
                                             
            self.crit_dicts[crit] = (Uncertainty(crit), self.criterions_dep[crit]) if not self.__nx_func(crit) else (Uncertainty(crit), self.criterions_dep["nx"])

    def __nx_func(self, func_str):
        return hasattr(nx, func_str) and callable(eval(f'nx.{func_str}'))


    def update_graph(self, G: nx.Graph):
        if "nx" in self.criterions_dep:
            self.criterions_dep["nx"]["graph"] = G


    def select(self, unlabeled: Dataset, labeled: Dataset, iteration: int = 0):

        # TODO
        # CHECK THIS SHIT
        _, G = self.graph_builder(unlabeled.data)

        self.update_graph(G)

        self.crit_scores = self._calc_crits(unlabeled, labeled)


        if self.weighted:
            # TODO
            # add weighting logic here
            pass
        else:
            weights = len(self.crit_dicts) * [1]

        return self.crit_scores
        final_scores = self.sum_dicts(*self.crit_scores, coef=weights)
        return sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:self.budget_per_iter]


    def _calc_crits(self, unlabeled, labeled):
        crit_scores = dict()

        # TODO
        # Decided later 
        X = torch.tensor([[1, 1, 1], [2, 2, 2], [0, 1, 0]], dtype=torch.float16)

        for crit, (func_crit, func_dep) in self.crit_dicts.items():
            # for every criterion provide it with the data and its dependancies and then calculate it and store it
            crit_scores[crit] = func_crit(X, **func_dep)
        
        return crit_scores
        
    def sum_dicts(self, *dicts, coef=None):
        if coef is None:
            coef = [1 / len(dicts)] * len(dicts)

        s = {}
        for k in dicts[0].keys():
            # s[k] = a[k] + b.get(k, 0)
            s[k] = sum([coef[i] * e.get(k, 0) for i, e in enumerate(dicts)])
        return s