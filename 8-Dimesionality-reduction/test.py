from t_sne import TSNE
import numpy as np
import matplotlib.pyplot as plt


X = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])

colors = ['red', 'blue', 'green', 'purple']


tsne = TSNE(n_components=1)
y = tsne.fit(X)
print("Output coordinates:\n", y)


plt.figure(figsize=(8, 6))


plt.subplot(2, 1, 1)      

plt.scatter(X[:, 0], X[:, 1], c=colors, s=100, label='Input Points')
plt.title("Input Data (2D)")
plt.grid(True, linestyle='--', alpha=0.6)


plt.subplot(2, 1, 2)      
y_axis_zeros = np.zeros_like(y)

plt.scatter(y, y_axis_zeros, c=colors, s=100, alpha=0.8, label='Output Points')
plt.title("t-SNE Result (1D)")
plt.yticks([])            
plt.grid(True, axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()