import numpy as np

class KNN:
    
    def __init__(self, n_neighbors, weight = "uniform"):
        
        self.n_neighbors = n_neighbors
        self.weight = weight  # uniform, distance
    
    
    def fit(self, x, y):  # x, y : np.array
        
        self.x = x
        self.y = y
        
        
    def predict(self, test_x):  # test_x : (1, n_features)
        
        distances = np.zeros(len(self.x))
        
        for i, x in enumerate(self.x):
            distances[i] = np.sqrt(sum((x - test_x) ** 2))
        
        sorted_distances = distances[np.argsort(distances)]
        sorted_labels = self.y[np.argsort(distances)]
        n_neighbors_distances = sorted_distances[:self.n_neighbors]
        n_neighbors_label = sorted_labels[:self.n_neighbors]
        
        if self.weight == "uniform":
            labels, votes = np.unique(n_neighbors_label, return_counts=True)
            print(labels, votes)
            pred_y = labels[np.argmax(votes)]
            
        elif self.weight == "distance":
            sum_n_neighbors_distances = sum(n_neighbors_distances)
            n_neighbors_distances = 1 - (n_neighbors_distances / sum_n_neighbors_distances)
            label_weight_arr = np.concatenate([n_neighbors_label.reshape(-1, 1), n_neighbors_distances.reshape(-1, 1)], axis=1)
            
            best_score = 0
            pred_y = None
            for unique in np.unique(n_neighbors_label):
                score = sum(label_weight_arr[label_weight_arr[:, 0] == unique][:, 1])
                if best_score <= score:
                    best_score = score
                    pred_y = unique
            
        return pred_y