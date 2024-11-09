import pickle

class WrapperDataset:
    def __init__(self, name):
        with open(f'./data/{name}', 'rb') as f:
            loaded = pickle.load(f)

        self.dataset = loaded["data"]
        self.num_labels = loaded["num_labels"]
        self.dim = loaded["dim"]