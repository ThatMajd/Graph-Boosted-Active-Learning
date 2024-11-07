import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import trange
import argparse

from selectioncriterion import SelectionCriterion


class ActiveLearningPipeline:
    def __init__(self,
                 dataset,
                 classifying_model,
                 selection_criterion,
                 weighted_selection: bool,
                 iterations: int,
                 budget_per_iter: int,
                 graph_building_function: str,
                 graph_threshold: float):
        """
        ActiveLearningPipeline class to manage the process of active learning using a graph-based selection criterion.
        This class facilitates iterative selection of data points based on a similarity-based graph, aiming to improve the 
        performance of a given classifier by selectively labeling new data points.

        Attributes:
        -----------
        dataset : object
            The dataset to be used in the active learning pipeline. Must support indexing and slicing for selection.
        classifying_model : object
            The model used to classify data points in the active learning loop. Should support fit and predict methods.
        selection_criterion : str
            The criterion to select data points for labeling. Common options include 'uncertainty' and 'density'.
        weighted_selection : bool
            Whether to apply weighted selection to the chosen samples for each iteration.
        iterations : int
            The number of iterations to perform in the active learning loop.
        budget_per_iter : int
            The number of data points to label in each iteration.
        graph_building_function : str
            Name of the function or method to create a similarity graph based on the dataset features.
        graph_threshold : float
            Threshold for similarity scores to establish edges in the graph; values above this create an edge.

        Methods:
        --------
        """
        
        # Parse data correctly
        self.train_samples, self.train_labels = dataset['train_samples'], dataset['train_labels']
        self.test_samples, self.test_labels = dataset['test_samples'], dataset['test_labels']
        self.available_pool_samples, self.available_pool_labels = dataset['available_pool_samples'], dataset['available_pool_labels']

        self.iterations = iterations
        self.budget_per_iter = budget_per_iter
        
        # SelectionCriterion should be a given 1 or more Uncertainty functions that will be used for selecting samples
        self.selection_criterion = SelectionCriterion(selection_criterion, 
                                                     budget_per_iter=self.budget_per_iter,
                                                     weighted=weighted_selection,
                                                     similarity_metric=graph_building_function,
                                                     threshold=graph_threshold)

        if classifying_model == "LogisticRegression":
            self.cls_model = LogisticRegression()
        else:
            print("Call Mahmod")

    def run_pipeline(self):
        """
        Run the active learning pipeline
        """
        accuracy_scores = []
        iterations_progress = trange(self.iterations)

        for iteration in iterations_progress:            
            trained_model = self._train_model()
            
            new_selected_samples = self.selection_criterion.select(unlabeled=self.available_pool_samples, 
                                                                   labeled=self.train_samples, 
                                                                   iteration=iteration)
            
            # new_selected_samples = self._random_sampling()
            
            self._update_indices(new_selected_samples)
            accuracy = self._evaluate_model(trained_model)
            accuracy_scores.append(accuracy)

            iterations_progress.set_postfix({"Accuracy": accuracy})
    
        return accuracy_scores
    
    def _random_sampling(self):
        return np.random.choice(range(self.available_pool_samples.shape[0]), self.budget_per_iter)

    def _train_model(self):
        """
        Train the model
        """
        return self.cls_model.fit(self.train_samples, self.train_labels)

    def _update_indices(self, new_selected_samples):
        """
        Update the indices such that the new selected samples are added to the train set and removed from the available pool
        """
        self.train_samples = np.vstack((self.train_samples, self.available_pool_samples[new_selected_samples]))
        self.train_labels = np.hstack((self.train_labels, self.available_pool_labels[new_selected_samples]))
        # Shuffle the train set to avoid bias caused by the order of the samples
        idx = np.random.permutation(len(self.train_labels))
        self.train_samples = self.train_samples[idx]
        self.train_labels = self.train_labels[idx]
        self.available_pool_samples = np.delete(self.available_pool_samples, new_selected_samples, axis=0)
        self.available_pool_labels = np.delete(self.available_pool_labels, new_selected_samples)

    def _evaluate_model(self, trained_model):
        """
        Evaluate the model
        :param trained_model: the trained model
        :return: the accuracy of the model on the test set
        """
        preds = trained_model.predict(self.test_samples)
        return round(np.mean(preds == self.test_labels), 3)