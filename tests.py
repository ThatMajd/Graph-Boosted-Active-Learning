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

# with open('data/dataset_q1.pkl', 'rb') as f:
# 	dataset = pickle.load(f)

# train_size, pool_size, test_size = None, 2_000, 1_000

# iterations = 100
# budg_per = 20

# # np.random.choice()

# dataset['available_pool_samples'] = dataset['available_pool_samples'][:pool_size]
# dataset['available_pool_labels'] = dataset['available_pool_labels'][:pool_size]

# dataset["test_samples"] = dataset["test_samples"][:test_size]
# dataset["test_labels"] = dataset["test_labels"][:test_size]


# al = GAL(dataset=dataset, 
# 		 classifier=LogisticRegression(), 
# 		 budget_per_iter=budg_per, 
# 		 iterations=iterations,
# 		 gnn_epochs=25,
# 		 quantile=.01,
# 		 AL4GE=True,
# 		 use_gnn=False)

# # nx.draw(al.create_train_graph(pytorch=False))
# # plt.show()

# res_gal = al.run(plot=False)
# selection_criteria = ['random', 'custom']
# accuracy_scores_dict = {}
# for criterion in selection_criteria:
# 	AL_class = AL(dataset=dataset,
# 			   selection_criterion=criterion,
# 			   iterations=iterations,
# 			   budget_per_iter=budg_per,
# 			   train_limit=int(1e6))
# 	accuracy_scores_dict[criterion] = AL_class.run_pipeline()

# # AL_class = AL(dataset=dataset,
# # 			  selection_criterion='custom',
# # 			  iterations=iterations,
# # 			  budget_per_iter=budg_per,
# # 			  train_limit=10000,)

# # res_al = AL_class.run_pipeline()

# if al.use_gnn:
# 	plt.plot(res_gal['aggr'], label='GAL')
# 	plt.plot(res_gal['GNN'], label='GNN', alpha=.5)
# 	plt.plot(res_gal['LR'], label='LR', alpha=.5)
# else:
# 	plt.plot(res_gal, label='GAL')

# for criterion, accuracy_scores in accuracy_scores_dict.items():
# 	plt.plot(accuracy_scores, label=criterion)
# 	# plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)

# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# # plt.xticks(range(1, len(accuracy_scores) + 1))
# plt.legend()
# plt.show()

# # plt.plot(res_al, label='AL')
# # plt.legend()
# # plt.show()







X = np.random.uniform(.1, 5, size=(5, 2))
y = np.array([0, 0, 0, 1, 1])
print(X)


ucs = UCAggregator(
	Uncertainty('entropy'),
	Uncertainty('pagerank'),
	Uncertainty('density_kmean'),
)

# print('A:')
sim = Similarity('euclidean')
# A = sim(X)
# print(A)

builder = GraphBuilder(sim)
G = builder.build(X, y)


for node_idx, node_attr in G.nodes.items():
	print(node_idx, node_attr)
var = Uncertainty('area_variance', labels=[0, 1])
print(var(G))

nx.draw(G, with_labels=True)
plt.show()



# print(ucs.calc(X, G=G, n_clusters=2, coef=[.2, .2, .6]))

# sel = Selector(3, ucs)
# print(sel.select(X, G=G, n_clusters=2, coef=[.2, .2, .6]))
