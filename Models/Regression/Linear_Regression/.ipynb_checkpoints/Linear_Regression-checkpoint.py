import numpy as np

class SimpleLinearRegression():
    def __init__(self):
        self.coef = None
        self.intercept = None
        
    def fit(self, x, y):
        coef = np.cov(x, y)[0][1] / np.var(x)
        intercept = np.mean(y) - coef * np.mean(x)
        self.coef = coef
        self.intercept = intercept
        
    def predict(self, x):
        assert self.coef != None and self.intercept != None, "ValueError : Not fitted yet! Please fit model."
        pred = self.coef * x + self.intercept
        return pred
    

class MultipleLinearRegression():
    def __init__(self):
        self.coef = None
        self.intercept = None
        
    def fit(self, x, y): # x : (num, features)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        x_t = x.T
        X = np.vstack([np.ones(x_t.shape[1]), x_t]).T
        theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.coef = theta[1:]
        self.intercept = theta[0]
        
    def predict(self, x): # x : (num, features)
        assert self.coef.all() != None and self.intercept.all() != None, "ValueError : Not fitted yet! Please fit model."
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        pred = x.dot(self.coef) + self.intercept
        return pred