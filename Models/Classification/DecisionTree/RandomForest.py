import numpy as np
import pandas as pd
from DecisionTree import DecisionTree

class RandomForest:

    def __init__(self, n_trees=10):
        self._n_trees = n_trees
        self._forest = [None] * self._n_trees
        self._used_features_set = [None] * self._n_trees

    def fit(self, X, y, random_state=0):  # X : pd.DataFrame, y : pd.Series
        self._classes = np.unique(y)
        sampled_X, sampled_y = self._bootstrap_sample(X, y, random_state=random_state)
        for i, (sampled_Xi, sampled_yi) in enumerate(zip(sampled_X, sampled_y)):
            tree = DecisionTree()
            tree.fit(sampled_Xi, sampled_yi)
            self._forest[i] = tree

    # データセットから `._n_trees` 個のサブセットを返す関数
    def _bootstrap_sample(self, X, y, random_state=0):

        n_features = len(X.columns)
        n_features_forest = int(np.floor(np.sqrt(n_features)))
        bootstrapped_X = []
        bootstrapped_y = []

        np.random.seed(random_state)
        for i in range(self._n_trees):
            idx = np.random.choice(len(y), size=len(y))
            col_idx = np.random.choice(n_features, size=n_features_forest, replace=False)
            bootstrapped_X.append(X.iloc[idx, col_idx])
            bootstrapped_y.append(y.iloc[idx])
            self._used_features_set[i] = col_idx
        return bootstrapped_X, bootstrapped_y

    def predict(self, X):
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self._forest[0] is None:
            raise ValueError('Model not fitted yet')

        # 決定木群による投票
        votes = pd.DataFrame()
        for i, (tree, used_features) in enumerate(zip(self._forest, self._used_features_set)):
            votes[i] = tree.predict(X.iloc[:, used_features])
        votes_array = votes.values

        # 投票結果を集計
        counts_array = np.zeros((len(X), len(self._classes)), dtype=int)
        for c in self._classes:
            counts_array[:, c] = np.sum(np.where(votes_array==c, 1, 0), axis=1)

        # 予測クラス毎に割合を計算し、probaとして返す
        proba = counts_array / self._n_trees
        return proba

    
# import numpy as np
# from DecisionTree import DecisionTreeClassifier

# class RandomForest:
    
#     def __init__(self, n_trees=10, max_depth=3):
#         self._n_trees = n_trees
#         self.max_depth = max_depth
#         self._forest = [None] * self._n_trees
#         self._used_features_set = [None] * self._n_trees
        
    
#     def fit(self, x, y, random_state=0):
#         self._classes = np.unique(y)
#         sampled_x, sampled_y = self._bootstrap_sample(x, y, random_state=random_state)
#         for i, (x_i, y_i) in enumerate(zip(sampled_x, sampled_y)):
#             each_tree = DecisionTreeClassifier(max_depth=self.max_depth)
#             each_tree.fit(x_i, y_i)
#             self._forest[i] = each_tree
        
        
#     def _bootstrap_sample(self, x, y, random_state=0):
#         n_features = x.shape[1]
#         n_features_forest = int(np.floor(np.sqrt(n_features)))
#         sampled_x = []
#         sampled_y = []
#         np.random.seed(random_state)
        
#         for i in range(self._n_trees):  # for each tree
#             idx = np.random.choice(len(y), size=len(y))
#             feature_idx = np.random.choice(n_features, size=n_features_forest, replace=False)
#             sampled_x.append(x[idx][:, feature_idx])
#             sampled_y.append(y[idx])
#             self._used_features_set[i] = feature_idx
        
#         return sampled_x, sampled_y
    
    
#     def predict(self, x):
#         proba = self.predict_proba(x)
#         return self._classes[np.argmax(proba, axis=1)]
    
    
#     def predict_proba(self, x):
#         if self._forest[0] is None:
#             raise ValueError("Model has not fitted yed.")
            
#         votes = []
#         for i, (tree, used_features) in enumerate(zip(self._forest, self._used_features_set)):
#             votes.append(tree.predict(x[:, used_features]))
#         votes_array = np.array(votes)
        
#         votes_result_array = np.zeros((len(x), len(self._classes)), dtype=int)
#         for c in self._classes:
#             votes_result_array[:, c] = np.sum(np.where(votes_array==c, 1, 0), axis=0)  # sum result for trees
            
#         proba = votes_result_array / self._n_trees
        
#         return proba