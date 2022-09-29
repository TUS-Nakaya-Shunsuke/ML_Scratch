import numpy as np
import random

class HardMarginSVM():
    def __init__(self, lr=1e-3, epoch=1000, random_state=None):
        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state
        self.is_trained = False
        
        
    def fit(self, x, y):
        self.num_samples = x.shape[0]
        self.num_features = x.shape[1]
        self.weight = np.zeros(self.num_features)
        self.b = 0
        r_generator = np.random.RandomState(self.random_state) 
        self.alpha = r_generator.normal(loc=0., scale=1e-2, size=self.num_samples) # From Normal Distribution
        
        for _ in range(self.epoch):
            self._cycle(x, y)
        
        indexes_sv = [idx for idx in range(self.num_samples) if self.alpha[idx] != 0] # Reduce amount of calculation
        
        for idx in indexes_sv:
            self.weight += self.alpha[idx] * y[idx] * x[idx]
            
        for idx in indexes_sv:
            self.b += y[idx] - (self.weight @ x[idx])
        self.b = self.b / len(indexes_sv)

        self.is_trained = True
        
        
    def predict(self, x):
        if not self.is_trained:
            raise Exception("This model hasn't trained yet.")
      
        hyperplane = x @ self.weight + self.b
        result = np.where(hyperplane > 0, 1, -1)
        return result
    
    
    def _cycle(self, x, y):
        y = y.reshape(-1, 1)
        H = (y @ y.T) * (x @ x.T)
        grad = np.ones(self.num_samples) - H @ self.alpha
        self.alpha += self.lr * grad
        self.alpha = np.where(self.alpha < 0, 0, self.alpha)