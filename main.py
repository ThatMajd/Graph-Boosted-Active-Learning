import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pipeline import GAL
from ref.ActiveLearning_OG import AL
from sklearn.linear_model import LogisticRegression
from utils.dataset_wrapper import WrapperDataset
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--classifier", type=str, required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument("--budget_per_iter", type=int)
parser.add_argument("--quantile", type=float)
parser.add_argument("--sim_metric", type=str)
parser.add_argument("--uncertainty_measures", type=iter)
parser.add_argument("--AL4GE", type=bool)
parser.add_argument("--n_clusters", type=int)
parser.add_argument("--use_gnn", type=bool)
parser.add_argument("--gnn_epochs", type=int)
parser.add_argument("--gnn_hidden", type=int, default=16)
parser.add_argument("--plot", type=bool)
parser.add_argument("--wandb", type=bool, default=False)

args = parser.parse_args()

data_object = WrapperDataset(args.dataset)
dataset = data_object.dataset
input_dim = data_object.dim
output_dim = data_object.num_labels

gal = GAL(dataset=dataset,
    classifier=LogisticRegression(),
    budget_per_iter=args.budget_per_iter,
    iterations=args.iterations,
    uncertainty_measures=args.uncertainty_measures,
    quantile=args.quantile,
    sim_metric=args.sim_metric,
    use_gnn=args.use_gnn,
    gnn_epochs=args.gnn_epochs,
    input_dim=input_dim,
    gnn_hidden=args.gnn_hidden,
    output_dim=output_dim,
    plot=args.plot,
    AL4GE=args.AL4GE,
    n_clusters=args.n_clusters,)

res_gal = gal.run(plot=False)

selection_criteria = ['random', 'custom']
accuracy_scores_dict = {}
for criterion in selection_criteria:
	AL_class = AL(dataset=dataset,
			   selection_criterion=criterion,
			   iterations=args.iterations,
			   budget_per_iter=args.budget_per_iter,
			   train_limit=int(1e6),)
	accuracy_scores_dict[criterion] = AL_class.run_pipeline()
 
 
accuracy_scores_dict = accuracy_scores_dict | res_gal
 
wandb.init()
for step in range(args.iterations):
    LOG = {"step": step} | {k: accuracy_scores_dict[k][step] for k in accuracy_scores_dict.keys()} 
    wandb.log(LOG)
