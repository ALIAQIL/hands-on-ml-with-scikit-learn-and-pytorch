import numpy as np

class TSNE:
    def __init__(self,
                 n_components=2,
                 lr=200.0,
                 n_iter=1000,
                 momentum=0.8,
                 random_state=None):
        self.n_components = n_components
        self.lr = lr
        self.n_iter = n_iter

    def compute_distribution(self,x):
        YX = np.asarray(x)

        diff = YX[:, None, :] - YX[None, :, :]
        dist2 = np.sum(diff**2, axis=2)

        num = 1.0 / (1.0 + dist2)
        np.fill_diagonal(num, 0.0)

        Q = num / np.sum(num)
        return Q
    
    def kl_divergence(self,P,Q,eps=1e-12):
        return np.sum(P * np.log(P/(Q+eps)))
    
    def initialize(self):
        np.random.seed(0)    
        n_samples = len(self.X)           
        Yy = 0.0001 * np.random.randn(n_samples, self.n_components)
        return Yy


    def gradient(self,y):
        Y = np.asarray(y)
        X = np.asarray(self.X)
        P = self.compute_distribution(X)
        Q = self.compute_distribution(Y)

        diff = Y[:, None, :] - Y[None, :, :]

        dist2 = np.sum(diff**2, axis=2)
        num = 1.0 / (1.0 + dist2)
        np.fill_diagonal(num, 0.0)

        nump = P - Q

        return 4 * np.sum((nump * num)[:, :, None]*diff,axis=1)
    
    def fit(self,X):
        self.X=X
        y = self.initialize()
        
        for _ in range(self.n_iter):
            grad = self.gradient(y)
            y-=self.lr * grad
        
        return y

