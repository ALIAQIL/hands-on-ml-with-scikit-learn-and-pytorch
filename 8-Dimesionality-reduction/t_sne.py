import numpy as np

class TSNE:
    def __init__(self,
                 n_components=2,
                 lr=0.1,
                 n_iter=1000,
                 momentum=0.8,
                 exaggeration_term=12,
                 random_state=42):
        self.n_components = n_components
        self.lr = lr
        self.n_iter = n_iter
        self.momentum = momentum
        self.random_state = random_state
        self.exaggeration_term = exaggeration_term

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
        np.random.seed(self.random_state)    
        n_samples = len(self.X)           
        Yy = 0.0001 * np.random.randn(n_samples, self.n_components)
        return Yy


    def gradient(self,y):
        Y = np.asarray(y)
        X = np.asarray(self.X)
        P = self.compute_distribution(X)
        Q = self.compute_distribution(Y)
        P = (P+P.T)/(2*len(self.X))  # similiarity
        diff = Y[:, None, :] - Y[None, :, :]

        dist2 = np.sum(diff**2, axis=2)
        num = 1.0 / (1.0 + dist2)
        np.fill_diagonal(num, 0.0)

        nump = P - Q

        return 4 * np.sum((nump * num)[:, :, None]*diff,axis=1)
    
    def fit(self, X):
        self.X = X
        y_1 = self.initialize()
        y_2 = y_1.copy()

        # Adam Hyperparameters
        ro1 = 0.9
        ro2 = 0.999
        eps = 1e-8
        s = np.zeros_like(y_1)
        r = np.zeros_like(y_1)

        # Start range from 1 to avoid division by zero in bias correction
        for i in range(1, self.n_iter + 1):
            grad = self.gradient(y_1)

            s = ro1 * s + (1 - ro1) * grad
            r = ro2 * r + (1 - ro2) * (grad**2)

            s_hat = s / (1 - ro1**i)
            r_hat = r / (1 - ro2**i)

            upd_grad = s_hat / (np.sqrt(r_hat) + eps)

            y_1copy = y_1.copy()

            y_1 -= (self.lr * self.exaggeration_term * upd_grad) + self.momentum * (y_1 - y_2)

            y_2 = y_1copy

        return y_1

