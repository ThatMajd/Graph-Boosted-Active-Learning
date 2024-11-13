from sklearn.linear_model import LogisticRegression
from utils.dataset_wrapper import WrapperDataset
from ref.ActiveLearning_OG import AL
import matplotlib.pyplot as plt
from pipeline import GAL
import numpy as np
import wandb
import os
os.listdir('data')
def sample(dataset, train_size=None, pool_size=2_000, test_size=1_000):
	n_pool = len(dataset['available_pool_samples'])
	n_test = len(dataset['test_samples'])

	pool_sample_idx = np.random.choice(range(n_pool), pool_size)
	test_sample_idx = np.random.choice(range(n_test), test_size)

	dataset['available_pool_samples'] = dataset['available_pool_samples'][pool_sample_idx]
	dataset['available_pool_labels'] = dataset['available_pool_labels'][pool_sample_idx]

	dataset["test_samples"] = dataset["test_samples"][test_sample_idx]
	dataset["test_labels"] = dataset["test_labels"][test_sample_idx]
	
ds_name = 'iris.pkl'
data_object = WrapperDataset(ds_name)
dataset = data_object.dataset
# sample(dataset)

input_dim = data_object.dim
output_dim = data_object.num_labels
labels = output_dim
dataset['available_pool_labels'].shape
iterations = 21
budg_per = 5
gnn_epochs = 5
quantile = .01
hidden_dim = 64
use_gnn = True
entropies = ['entropy_e',
			 'density_kmean',
			 'pagerank',
			 'area_variance', 
			 ('entropy_e',
	 		  'density_kmean',
			  'pagerank',
			  'area_variance',)]
GAL_dict = {}

for e in entropies:
	print(e)
	al = GAL(dataset,
		 LogisticRegression(),
		 budg_per,
		 uncertainty_measures=[e] if isinstance(e, str) else e,
		 coef=[True, False, False, False] if isinstance(e, list) else None,
		 iterations=iterations,
		 gnn_epochs=gnn_epochs,
		 quantile=quantile,
		 labels=labels,
		 input_dim=input_dim,
		 hidden_dim=hidden_dim,
		 output_dim=output_dim,
		 use_gnn=use_gnn,
		 train_graph_include_test=False,)
	GAL_dict[e] = al.run(plot=False)


al = GAL(dataset,
	LogisticRegression(),
	budg_per,
	uncertainty_measures=[e] if isinstance(e, str) else e,
	coef=[True, False, False, False] if isinstance(e, list) else None,
	iterations=iterations,
	gnn_epochs=gnn_epochs,
	quantile=quantile,
	labels=labels,
	input_dim=input_dim,
	hidden_dim=hidden_dim,
	output_dim=output_dim,
	use_gnn=use_gnn,
	train_graph_include_test=False,)
GAL_dict['AL4GE'] = al.run(plot=False)
selection_criteria = ['random', 'custom']
accuracy_scores_dict = {}
for criterion in selection_criteria:
	AL_class = AL(dataset=dataset,
			   selection_criterion=criterion,
			   iterations=iterations,
			   budget_per_iter=budg_per,
			   train_limit=int(1e6))
	accuracy_scores_dict[criterion] = AL_class.run_pipeline()

# GAL_dict['entropy_e']
# plt.plot(res_gal['aggr'], label='GAL')
# if al.use_gnn:
# 	plt.plot(res_gal['GNN'], label='GNN', alpha=.5)
# 	plt.plot(res_gal['LR'], label='LR', alpha=.5)

for criterion, accuracy_scores in GAL_dict.items():
	print(f'iris.pkl,{criterion},{criterion == "AL4GE"},{np.mean(accuracy_scores["GNN"]):.3f},{np.mean(accuracy_scores["LR"]):.3f},{np.mean(accuracy_scores["aggr"]):.3f}')
	# print(criterion, f'{np.mean(accuracy_scores['aggr']):.3f}')
	plt.plot(accuracy_scores['aggr'], label=criterion)

for criterion, accuracy_scores in accuracy_scores_dict.items():
	print(f'iris.pkl,{criterion},{criterion == "AL4GE"},{np.mean(accuracy_scores):.3f}')
	plt.plot(accuracy_scores, label=criterion)
	# plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label=criterion)

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
# plt.xticks(range(1, len(accuracy_scores) + 1))
plt.legend()
plt.show()
