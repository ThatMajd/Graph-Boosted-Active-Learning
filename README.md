<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.5em"> Graph Boosted Active Learning GAL</h1>

<p align='center' style="text-align:center;font-size:1em;">
    <a>Majd Bishara</a>&nbsp;,&nbsp;
    <a>Saleem Kheer-Eldeen</a>&nbsp;,&nbsp;
    <a>Lana Haj</a>&nbsp;
    <br/> <b>Technion Institute of Technology</b><br/> 
    
</p>


## Overview

Graph Active Learning (GAL) is an approach that uses Graph Neural Networks (GNNs) to enhance active learning on tabular data. Active Learning (AL) identifies the most valuable data points to label, optimizing model training. Traditional AL methods often rely solely on model predictions, which limits their effectiveness. GAL improves on this by using a GNN to incorporate the underlying structure of the data, enabling a more informed selection of data points for labeling.

## Files
- [builders.py](utils/builders.py): The tabular data is transformed into a graph where each data point represents a node, and based on similarity matric the edges are created.
- [metrics.py](utils.metrics.py): This file provides classes and methods to calculate similarity and uncertainty matrics.
- [pipeline.py](pipeline.py): Implements the GAL pipeline. It combines uncertainty-based active learning with graph-based methods, leveraging both a traditional classifier and a Graph Neural Network (GNN) model.
- [gnn_models.py](gnn_models.py): In this file we implemented the GNN models.

## Datasets
We tested GAL on 5 datasets: 
- Iris
- Wine Quality
- Lab dataset
- Clustered (synthetic dataset)
- Unclustered (synthetic dataset)

## Installation and Run 
- clone the repository
- Install the necessary dependencies: `pip install -r requirements.txt`
- Run the Code: `main.py --params`

