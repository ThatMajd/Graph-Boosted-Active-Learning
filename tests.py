from selectioncriterion import SelectionCriterion
import numpy as np
x = np.random.normal(size=(5, 3))

s = SelectionCriterion('pagerank',
					   'eigenvector_centrality',
					   budget_per_iter=2,)
print(s.select(x, x))


