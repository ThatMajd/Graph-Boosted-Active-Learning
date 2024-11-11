import wandb

wandb_username = 'majedbishara-technion-israel-institute-of-technology'
wandb_project = 'Bsc_Finale'

command = ['${env}', '${interpreter}', 'main.py', '${args}']

sweep_config={
    'method': 'grid',
    'metric': {
        'name': 'Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        "dataset": {
            "values": ["lab_dataset.pkl"]  # Replace with actual dataset options
        },
        "classifier": {
            "values": ["LogisticRegression"]
        },
        "iterations": {
            "values": [100]
        },
        "uncertainty_measures": {
            "values": ["pagerank"]
        },
        "budget_per_iter": {
            "values": [8]
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
			"values": [['entropy_e',
						'density_kmean',
						'pagerank',
						'area_variance',],]
		},
		"coef": {
			"values": [[True, True, False, False], ]
		},
        "n_clusters": {
            "values": [2, 4, 5]
        },
        "use_gnn": {
            "values": [True]
        },
        "gnn_epochs": {
            "values": [5, 15, 25]
        },
        "hidden_size": {
            "values": [16, 32, 64]  # Hidden layer sizes for GNN
        },
        'wandb': {
            'value': 1
        }
    },
    'command': command
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project)

print("=== Run this command ===")
print(f"wandb agent {wandb_username}/{wandb_project}/{sweep_id}")