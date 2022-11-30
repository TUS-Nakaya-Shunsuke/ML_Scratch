import numpy as np

class LogisticRegression():
    
    def __init__(self, lr, n_iter=100000, intercept=True):
        self.lr = lr
        self.n_iter = n_iter
        self.intercept = intercept
        
    def add_intercept(self, x):
        add = np.ones((x.shape[0], 1))
        return np.concatenate((x, add), axis=1)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cross_entropy(self, hat, y):
        return (-y * np.log(hat) - (1 - y) * np.log(1 - hat)).mean()
    
    def fit(self, x, y):
        y = y.reshape(-1, 1)
        if self.intercept:
            x = self.add_intercept(x)
        self.theta = np.zeros((x.shape[1], 1))  # weights initialization
        
        z = np.dot(x, self.theta)
        hat = self.sigmoid(z)
        
        for i in range(self.n_iter):
            grad = np.dot(x.T, (hat - y)) / y.shape[0]
            self.theta -= self.lr * grad
            z = np.dot(x, self.theta)
            hat = self.sigmoid(z)
            loss = self.cross_entropy(hat, y)
            
    def predict_proba(self, x):
        if self.intercept:
            x = self.add_intercept(x)
        return self.sigmoid(np.dot(x, self.theta))
            
    def predict(self, x):
        return self.predict_proba(x).round()