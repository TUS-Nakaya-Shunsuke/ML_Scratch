import numpy as np

class Linear_Regression():
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