from collections import Counter
import numpy as np
import pandas as pd

class Node:

    def __init__(self):
        self.left = None      # 左側に分岐した子ノード
        self.right = None     # 右側に分岐した子ノード
        self.feature = None   # 分岐条件の特徴量
        self.value = None     # 分岐条件のしきい値
        self.label = None     # 自身の予測ラベル
        self.gain = None      # 自身の情報ゲイン

    def build(self, X, y):  # X : pd.Dataframe, y : pd.Series

        features = X.columns
        n_features = len(features)

        if len(y.unique()) == 1:
            self.label = y.unique()[0]
            return

        class_counts = Counter(y)  # ラベルの各クラスの数をdict型で返す
        self.label = max(class_counts, key=class_counts.get)

        # 最適な分岐条件を保持するための変数
        best_gain = 0.0
        best_feature = None
        best_value = None

        gini = self.gini(y)  # 情報ゲインの計算に必要

        for f in features:
            values = X[f].unique()

            for val in values:
                y_l = y[X[f] >= val]
                y_r = y[X[f] <  val]

                gain = self.information_gain(y_l, y_r, gini)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_value = val

        # 情報ゲインが増えていなければ終了
        if best_gain == 0:
            return
        else:
            self.gain = best_gain
            self.feature = best_feature
            self.value = best_value

            # 左側のノードを再帰的に構築
            X_l = X[X[self.feature] >= self.value]
            y_l = y[X[self.feature] >= self.value]
            self.left = Node()
            self.left.build(X_l, y_l)

            # 右側のノードを再帰的に構築
            X_r = X[X[self.feature] < self.value]
            y_r = y[X[self.feature] < self.value]
            self.right = Node()
            self.right.build(X_r, y_r)

    # ジニ不純度の計算
    def gini(self, y):
        classes = y.unique()
        gini = 1.0
        for c in classes:
            gini -= (len(y[y==c]) / len(y)) ** 2
        return gini

    # 情報ゲインの計算
    def information_gain(self, y_l, y_r, current_uncertainty):
        pl = float(len(y_l) / (len(y_l)+len(y_r)))
        pr = float(len(y_r) / (len(y_l)+len(y_r)))
        return current_uncertainty - (pl*self.gini(y_l) + pr*self.gini(y_r))

    # 一つのデータに対する予測
    def predict(self, datum):
        # 葉ノードでなければ分岐条件に従って分岐する
        if self.feature != None:
            if datum[self.feature] >= self.value:
                return self.left.predict(datum)
            else:
                return self.right.predict(datum)
        # 葉ノードなら自身の予測ラベルを返す
        else:
            return self.label
        
class DecisionTree:

    def __init__(self):
        self.root = None

    def fit(self, X, y):
        self.root = Node()
        self.root.build(X, y)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.root.predict(X.iloc[i]))
        return pd.Series(y_pred)



# import numpy as np

# def calculate_gini(data, label, feature_idx, threshold):
    
#     gini = 0
#     data_size = len(label)

#     # Division target left or right
#     # Division_target = [label[data[:, feature_idx] >= threshold], label[data[:, feature_idx] < threshold]]
#     label_left = label[data[:, feature_idx] >= threshold]
#     label_right = label[data[:, feature_idx] < threshold]

#     score_left = 0
#     score_right = 0
#     classes = np.unique(label)

#     for c in classes:
#         if len(label_left) != 0:
#             score_left += float((np.sum(label_left == c) / len(label_left))**2)
#         if len(label_right) != 0:
#             score_right += float((np.sum(label_right == c) / len(label_right))**2)

#     gini = ((1 - score_left)*len(label_left) + (1 - score_right)*len(label_right)) / data_size

#     return gini


# def search_best_split(data, label):
    
#     features = data.shape[1]
#     best_threshold = None
#     best_feature_idx = None
#     gini = None
#     gini_min = 1

#     for feature_idx in range(features):
#         values = data[:, feature_idx]
#         for value in values:
#             gini = calculate_gini(data, label, feature_idx, value)
#             if gini_min > gini:
#                 gini_min = gini
#                 best_threshold = value
#                 best_feature_idx = feature_idx

#     return gini_min, best_threshold, best_feature_idx


# class DecisionTreeNode():
    
#     def __init__(self, data, label, max_depth):
#         self.data = data
#         self.label = label
#         self.max_depth = max_depth
#         self.left = None
#         self.right = None
#         self.depth = None
#         self.threshold = None
#         self.feature_idx = None
#         self.gini_min = None
#         self.output = np.argmax(np.bincount(self.label))
        
    
#     def fit(self, depth):
        
#         if len(np.unique(self.label)) == 1:
#             self.label = np.unique(self.label)
#             return
        
#         self.depth = depth
#         self.gini_min, self.threshold, self.feature_idx = search_best_split(self.data, self.label)
        
#         if self.depth == self.max_depth or self.gini_min == 0 or len(self.label) == 0:
#             return
        
#         left_idx = self.data[:, self.feature_idx] >= self.threshold
#         right_idx = self.data[:, self.feature_idx] < self.threshold
        
#         self.left = DecisionTreeNode(self.data[left_idx], self.label[left_idx], self.max_depth)
#         self.right = DecisionTreeNode(self.data[right_idx], self.label[right_idx], self.max_depth)
#         self.left.fit(self.depth + 1)
#         self.right.fit(self.depth + 1)
        
        
#     def predict(self, data):
        
#         if len(np.unique(self.label)):
#             return self.label
        
#         if self.gini_min == 0. or self.depth == self.max_depth or len(self.label) == 0:
#             return self.output
        
#         else:
#             if data[self.feature_idx] > self.threshold:
#                 return self.left.predict(data)
#             else:
#                 return self.right.predict(data)
            
            
# class DecisionTreeClassifier():
    
#     def __init__(self, max_depth=3):
#         self.max_depth = max_depth
#         self.tree = None
   
#     def fit(self, data, label):
#         initial_depth = 0
#         self.tree = DecisionTreeNode(data, label, self.max_depth)
#         self.tree.fit(initial_depth)
   
#     def predict(self, data):
#         pred = []
#         for d in data:
#             pred.append(self.tree.predict(d))
#         return np.array(pred)