import numpy as np

def calculate_gini(data, label, feature_idx, threshold):
    gini = 0
    data_size = len(label)

    # Division target left or right
    # division_target = [label[data[:, feature_idx] >= threshold], label[data[:, feature_idx] < threshold]]
    label_left = label[data[:, feature_idx] >= threshold]
    label_right = label[data[:, feature_idx] < threshold]

    score_left = 0
    score_right = 0
    classes = np.unique(label)

    for c in classes:
        score_left += float((np.sum(label_left == c) / len(label_left))**2)
        score_right += float((np.sum(label_right == c) / len(label_right))**2)

    gini = ((1 - score_left)*len(label_left) + (1 - score_right)*len(label_right)) / data_size

    return gini


def search_best_split(data, label):
    features = data.shape[1]
    best_threshold = None
    best_feature_idx = None
    gini = None
    gini_min = 1

    for feature_idx in range(features):
        values = data[:, feature_idx]
        for value in values:
            gini = calculate_gini(data, label, feature_idx, value)
            if gini_min > gini:
                gini_min = gini
                best_threshold = value
                best_feature_idx = feature_idx

    return gini_min, best_threshold, best_feature_idx


class DecisionTreeNode():
    def __init__(self, data, label, max_depth):
        self.data = data
        self.label = label
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.depth = None
        self.threshold = None
        self.feature_idx = None
        self.gini_min = None
        self.output = np.argmax(np.bincount(label))
        
    
    def fit(self, depth):
        self.depth = depth
        self.gini_min, self.threshold, self.feature_idx = search_best_split(self.data, self.label)
        
        if self.depth == self.max_depth or self.gini_min == 0:
            return
        
        left_idx = self.data[:, self.feature_idx] >= self.threshold
        right_idx = self.data[:, self.feature_idx] < self.threshold
        
        self.left = DecisionTreeNode(self.data[left_idx], self.label[left_idx], self.max_depth)
        self.right = DecisionTreeNode(self.data[right_idx], self.label[right_idx], self.max_depth)
        self.left.fit(self.depth + 1)
        self.right.fit(self.depth + 1)
        
        
    def predict(self, data):
        if self.gini_min == 0. or self.depth == self.max_depth:
            return self.output
        else:
            if data[self.feature_idx] > self.threshold:
                return self.left.predict(data)
            else:
                return self.right.predict(data)
            
            
class DecisionTreeClassifier():
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
   
    def fit(self, data, label):
        initial_depth = 0
        self.tree = DecisionTreeNode(data, label, self.max_depth)
        self.tree.fit(initial_depth)
   
    def predict(self, data):
        pred = []
        for d in data:
            pred.append(self.tree.predict(d))
        return np.array(pred)