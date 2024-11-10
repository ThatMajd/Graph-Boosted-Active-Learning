import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pipeline import GAL
from ref.ActiveLearning_OG import AL
from sklearn.linear_model import LogisticRegression
from utils.dataset_wrapper import WrapperDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--classifier", type=str)
parser.add_argument("--iterations", type=int)
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

args = parser.parse_args()

data_object = WrapperDataset(args.dataset)
dataset = data_object["data"]
input_dim = data_object["dim"]
output_dim = data_object["num_labels"]

GAL(dataset=data_object,
    classifier=LogisticRegression(),
    budget_per_iter=args.budget_per_iter,
    iterations=args.iterations
    uncertainty_measures=args.uncertainty_measures,
    quantile=args.quantile,
    sim_metric=args.sim_metric,
    use_gnn=args.use_gnn,
    gnn_epochs=args.gnn_epochs,
    input_dim=input_dim,
    hidden_dim=parser.gnn_hidden,
    output_dim=output_dim,
    plot=args.plot,
    AL4GE=args.AL4GE,
    n_clusters=args.n_clusters)