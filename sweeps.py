import os
from pprint import pprint
import copy

command = ['${env}', '${interpreter}', 'main.py', '${args}']

sweep_config = {
	'method': 'grid',
	'name': None,
}

metric = {
	'name': 'aggr',
	'goal': 'maximize',
}


parameters = {
	"dataset": {
		"values": []  # Replace with actual dataset options
	},
	"classifier": {
		"values": ["LogisticRegression"]
	},
	"iterations": {
		"values": []
	},
	"budget_per_iter": {
		"values": []
	},
	"quantile": {
		"values": [0.1, 0.25, 0.5]
	},
	"sim_metric": {
		"values": ["cosine", "euclidean"]  # Replace with actual similarity metrics
	},
	# "uncertainty_measures": {
	#     "values": [["measure1", "measure2"], ["measure3"]]  # Replace with actual measures
	# },
	"AL4GE": {
		"values": [False]
	},
	"uncertainty_measures": {
		"values": [None]
	},
	"coef": {
		"values": [None]
	},
	"n_clusters": {
		"values": [2, 3, 4]
	},
	"use_gnn": {
		"values": []
	},
	"gnn_epochs": {
		"values": [5, 15, 25]
	},
	"gnn_hidden": {
		"values": [32, 64]  # Hidden layer sizes for GNN
	},
	'wandb': {
		'value': 1
	},
}

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters
sweep_config['command'] = command


sweep_configs = {}
datasets = os.listdir('data') + ['clustered.pkl', 'unclustered.pkl']
sweep_params = {
	'iris': {
		'iterations': 21,
		'budget_per_iter': 5,
	},
	'wineQT': {
		'iterations': 100,
		'budget_per_iter': 8,
	},
	'lab_dataset_2000': {
		'iterations': 100,
		'budget_per_iter': 20,
	},
	'clustered': {
		'iterations': 100,
		'budget_per_iter': 30,
	},
	'unclustered': {
		'iterations': 100,
		'budget_per_iter': 30,
	},
}

use_gnn_options = [True]
measure_options = [['entropy_e', 'density_kmean', 'pagerank', 'area_variance'], [['entropy_e', 'density_kmean', 'pagerank', 'area_variance',]]]
measure_coef = [True, False, False, False]

from utils.dataset_wrapper import WrapperDataset
i = 0
for dataset_name in datasets:
	t = copy.deepcopy(sweep_config)
	t['parameters']['dataset'] = {'values': [dataset_name]}
	e = dataset_name.split('.')[0]

	for k, v in sweep_params[e].items():
		t['parameters'][k] = {'values': [v]}

	for use_gnn in use_gnn_options:
		tt = copy.deepcopy(t)
		tt['parameters']['use_gnn'] = {'values': [use_gnn]}

		for uc in measure_options:
			ttt = copy.deepcopy(tt)
			ttt['name'] = f'{e}_single'
			if len(uc) == 1:
				ttt['name'] = f'{e}_aggr'
				ttt['parameters']['coef'] = {'values': [measure_coef]}
			ttt['parameters']['uncertainty_measures'] = {'values': uc}

			sweep_configs[i] = ttt
			i += 1

			# print(dataset_name, use_gnn, uc)

		
		ttt = copy.deepcopy(tt)
		ttt['parameters']['AL4GE'] = {'values': [True]}
		ttt['name'] = f'{e}_AL4GE'
		sweep_configs[i] = ttt
		i += 1
		# print(dataset_name, use_gnn, 'AL4GE')

		


pprint(sweep_configs)
print(len(sweep_configs))





