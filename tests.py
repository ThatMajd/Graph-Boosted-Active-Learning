from utils.metrics import Similarity, Uncertainty, UCAggregator
from utils.builders import GraphBuilder 
from utils.selection import Selector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pipeline import GAL
from ref.ActiveLearning_OG import AL
from sklearn.linear_model import LogisticRegression

import pickle

with open('data/dataset_q1.pkl', 'rb') as f:
	dataset = pickle.load(f)

dataset['available_pool_samples'] = dataset['available_pool_samples'][:60]
dataset['available_pool_labels'] = dataset['available_pool_labels'][:60]

al = GAL(dataset=dataset, 
		 classifier=LogisticRegression(), 
		 budget_per_iter=5, 
		 iterations=10,
		 AL4GE=True,
		 threshold=2)

res_gal = al.run(plot=True)

AL_class = AL(dataset=dataset,
			  selection_criterion='custom',
			  iterations=10,
			  budget_per_iter=5,
			  train_limit=10000)

res_al = AL_class.run_pipeline()

plt.plot(res_gal, label='res_gal')
plt.plot(res_al, label='res_al')
plt.legend()
plt.show()







# X = np.random.uniform(.1, 5, size=(5, 2))
# print(X)


# ucs = UCAggregator(
# 	Uncertainty('entropy'),
# 	Uncertainty('pagerank'),
# 	Uncertainty('density_kmean'),
# )

# # print('A:')
# sim = Similarity('euclidean')
# # A = sim(X)
# # print(A)

# builder = GraphBuilder(sim)
# G, E = builder.build(X)
# # print(G.nodes)
# # nx.draw(G, with_labels=True)
# # plt.show()


# print(ucs.calc(X, G=G, n_clusters=2, coef=[.2, .2, .6]))

# sel = Selector(3, ucs)
# print(sel.select(X, G=G, n_clusters=2, coef=[.2, .2, .6]))
