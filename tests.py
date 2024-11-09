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

train_size, pool_size, test_size = None, 2_000, 1_000

iterations = 100
budg_per = 20

# np.random.choice()

dataset['available_pool_samples'] = dataset['available_pool_samples'][:pool_size]
dataset['available_pool_labels'] = dataset['available_pool_labels'][:pool_size]

dataset["test_samples"] = dataset["test_samples"][:test_size]
dataset["test_labels"] = dataset["test_labels"][:test_size]


al = GAL(dataset=dataset, 
		 classifier=LogisticRegression(), 
		 budget_per_iter=budg_per, 
		 iterations=iterations,
		 gnn_epochs=25,
		 quantile=.01,
		 AL4GE=True,
		 use_gnn=False)

# nx.draw(al.create_train_graph(pytorch=False))
# plt.show()

res_gal = al.run(plot=False)

AL_class = AL(dataset=dataset,
			  selection_criterion='custom',
			  iterations=iterations,
			  budget_per_iter=budg_per,
			  train_limit=10000,)

res_al = AL_class.run_pipeline()

if al.use_gnn:
	plt.plot(res_gal['aggr'], label='res_gal')
	plt.plot(res_gal['GNN'], label='GNN', alpha=.5)
	plt.plot(res_gal['LR'], label='LR', alpha=.5)
else:
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
