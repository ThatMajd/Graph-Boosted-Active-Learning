from utils.metrics import Similarity, Uncertainty, UCAggregator
from utils.builders import GraphBuilder
from utils.wrappers import ModelWrapper
from utils.selection import Selector
import matplotlib.pyplot as plt
from tqdm import trange
import networkx as nx
import numpy as np
from utils.gnn_models import SimpleGNN
import torch


class GAL:
    def __init__(self,
                 dataset,
                 classifier,
                 budget_per_iter: int,
                 iterations: int = 10,
                 quantile: float = 0.5,
                 sim_metric: str = 'euclidean',
                 *uncertainty_measures,
                 **kwargs):
        
        self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
        self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
        self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset['available_pool_labels']

        self.similarity = Similarity(sim_metric)
        self.graph_builder = GraphBuilder(self.similarity)

        if kwargs.get('AL4GE'):
            self.ucs = UCAggregator(
                Uncertainty('entropy_e'),
                Uncertainty('density_kmean'),
                Uncertainty('pagerank'),
            )
            self.nx_flag = True
            self.selector = Selector(budget_per_iter, self.ucs, AL4GE=True)
        else:
            self.ucs = UCAggregator(Uncertainty(e) for e in uncertainty_measures)
            self.nx_flag = any(e.nx_flag for e in self.ucs.ucs)
            self.selector = Selector(budget_per_iter, self.ucs)

        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        self.classifier = ModelWrapper(classifier)
        self.n_clusters = kwargs.get('n_clusters', 2)
        self.quantile = quantile
        
        assert self.iterations * self.budget_per_iter <= len(dataset["available_pool_labels"]), "Not enough samples in pool"
        
        self.use_gnn = kwargs.get("use_gnn", "False")
        if self.use_gnn:
            input_dim = 3  # Assuming features are in columns of data_x
            hidden_dim = 16
            output_dim = 4  # Assuming classification
            
            self.epochs = kwargs.get("gnn_epochs", 5)
            self.gnn_model = SimpleGNN(input_dim, hidden_dim, output_dim)
            self.optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.007)
            self.criterion = torch.nn.CrossEntropyLoss()
            
            self.eval_graph = None

    def update_indices(self, selection_indices):
        self.train_samples = np.vstack((self.train_samples, self.available_pool_samples[selection_indices]))
        self.train_labels = np.hstack((self.train_labels, self.available_pool_labels[selection_indices]))
        # Shuffle the train set to avoid bias caused by the order of the samples
        idx = np.random.permutation(len(self.train_labels))
        self.train_samples = self.train_samples[idx]
        self.train_labels = self.train_labels[idx]
        self.available_pool_samples = np.delete(self.available_pool_samples, selection_indices, axis=0)
        self.available_pool_labels = np.delete(self.available_pool_labels, selection_indices)
    
    def _evaluate_model(self, trained_model):
        """
        Evaluate the model
        :param trained_model: the trained model
        :return: the accuracy of the model on the test set
        """
        preds = trained_model.predict(self.test_samples)
        return round(np.mean(preds == self.test_labels), 3)
    
    def _train_gnn(self):
        train_x, train_y = self.train_samples, self.train_labels
        pool_x, pool_y = self.available_pool_samples, self.available_pool_labels
        
        data_x = np.concat([train_x, pool_x])
        data_y = np.concat([train_y, pool_y])
        
        # Create a pytorch Graph
        G = self.graph_builder(data_x, data_y, self.quantile, pytorch=True)
        
        train_mask = np.array([True] * len(train_x) + [False] * len(pool_x))
        pool_mask = np.array([False] * len(train_x) + [True] * len(pool_x))
        
        G.train_mask = train_mask
        G.pool_mask = pool_mask
        
        optimizer = self.optimizer
        criterion = self.criterion
        
        self.gnn_model.train()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.gnn_model(G)
            loss = criterion(out[G.train_mask], G.y[G.train_mask])  # Use only train mask nodes
            loss.backward()
            optimizer.step()
            
            _, preds = out[G.train_mask].max(dim=1)  # Get the index of the max log-probability
            correct = preds.eq(G.y[G.train_mask]).sum().item()
            accuracy = correct / G.train_mask.sum().item()  # Compute accuracy as a ratio
            
            # Print training loss and accuracy
            # if epoch + 1 == self.epochs: 
            #     print(f'[GNN] - Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy:.4f}')
        
        return accuracy
            
    def _evaluate_gnn(self):
        if not self.eval_graph:
            self.eval_graph = G = self.graph_builder(
                self.test_samples,
                self.test_labels,
                self.quantile,
                pytorch=True
            )
        else:
            G = self.eval_graph
    
        self.gnn_model.eval()
        
        with torch.no_grad():
            out = self.gnn_model(G)
        
        # Calculate accuracy on test data
        confs, preds = out.max(dim=1)  # Get the predicted classes
        correct = preds.eq(G.y).sum().item()
        accuracy = correct / G.y.size(0)  # Total number of test nodes
        
        # print(f"[GNN] - Test Accuracy: {accuracy:.4f}")
        return out, accuracy
        
  

    def run(self, **kwargs):
        accuracy_scores = []
        iterations_progress = trange(self.iterations)

        n_clusters = kwargs.get('n_clusters', self.n_clusters)
        graph_flag = kwargs.get('plot', False)

        for iter in iterations_progress:
            self.classifier.fit(self.train_samples, self.train_labels)
            
            if self.use_gnn:
                gnn_train_acc = self._train_gnn()
            
            nx_G = self.graph_builder.build(
                self.available_pool_samples,
                self.available_pool_labels,
                self.quantile)
            if graph_flag:
                pos = dict(zip(range(len(self.available_pool_samples)), self.available_pool_samples[:, [0, 1]]))
                nx.draw(nx_G, pos=pos, with_labels=True)
                plt.show()
            selection_indices = self.selector.select(self.available_pool_samples, 
                                                     iter+1, 
                                                     n_clusters=n_clusters,
                                                     G=nx_G,
                                                     model=self.classifier)
            self.update_indices(selection_indices)
            # accuracy = self._evaluate_model(self.classifier)
            
            cls_out = self.classifier(self.test_samples)
            if self.use_gnn:
                gnn_out, gnn_test_acc= self._evaluate_gnn()
                
                assert cls_out.shape == gnn_out.shape
                
                # Aggregartion Function
                final_preds = cls_out
                
                labels = np.argmax(final_preds, axis=1)
                
                accuracy = (labels == self.test_labels).sum() / cls_out.shape[0]
                
                accuracy_scores.append(accuracy)

                iterations_progress.set_postfix({"Accuracy": accuracy, "GNN Train Acc": gnn_train_acc, "GNN Test Acc": gnn_test_acc})
            
            else:
                accuracy = self._evaluate_model(self.classifier)
                accuracy_scores.append(accuracy)

                iterations_progress.set_postfix({"Accuracy": accuracy})
        
        return accuracy_scores
