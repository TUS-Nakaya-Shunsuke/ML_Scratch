import numpy as np


class KMeans():
    
    def __init__(self, n_clusters = 4, max_iter = 100, init = "kmeans++"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.init = init
        
    
    def fit(self, x):  # x = (n_samples, n_features)
        
        n_data = x.shape[0]
        data_idx = np.arange(n_data)
        
        if self.init == "random":
            init_data_idx = np.random.choice(data_idx, self.n_clusters, replace=False)
            centers = x[init_data_idx, :]
        
        elif self.init == "kmeans++":
            n_features = x.shape[1]
            centers = np.zeros((self.n_clusters, n_features))
            distance = np.zeros((n_data, self.n_clusters))
            choice_idx = []
            prob = np.repeat(1/n_data, n_data)
            
            for i in range(self.n_clusters):
                choice_idx.append(int(np.random.choice(data_idx, 1, p=prob)))
                centers[i, :] = x[choice_idx[i], :]
                distance[:, i] = np.sum((x - centers[i, :])**2, axis=1)
                distance[choice_idx, :] = 0.
                prob = np.sum(distance, axis=1) / np.sum(distance)
            
        else:
            raise ValueError("init is difined random or kmeans++")
        
        belong_clusters = np.zeros(n_data)

        for it in range(self.max_iter):
            for i in range(n_data):
                belong_clusters[i] = np.argmin(np.sum((x[i, :] - centers) ** 2, axis=1))

            pre_centers = centers.copy()

            for k in range(self.n_clusters):
                centers[k, :] = x[belong_clusters == k, :].mean(axis=0)

            if np.all(centers == pre_centers):
                break

        print(f"iter : {it}")
        self.centers = centers
        
        
    def predict(self, x):  # x = (n_samples, n_features)
        n_data = x.shape[0]
        
        belongings = np.zeros(n_data)
        
        for i in range(n_data):
                belongings[i] = np.argmin(np.sum((x[i, :] - self.centers) ** 2, axis=1))
        
        return belongings