from ActiveLearning import ActiveLearningPipeline
import pickle
with open('dataset_q1.pkl', 'rb') as f:
	dataset = pickle.load(f)


al = ActiveLearningPipeline(dataset=dataset, 
                            classifying_model="LogisticRegression", 
                            selection_criterion="entropy", 
                            weighted_selection=False, 
                            iterations=200, 
                            budget_per_iter=5, 
                            graph_building_function="cosine", 
                            graph_threshold=0.8)

res = al.run_pipeline()

print(res)