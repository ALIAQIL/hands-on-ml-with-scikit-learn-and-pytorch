import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_swiss_roll
from t_sne import TSNE 

# 1. Generate 1000 Samples
X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# 2. Run your TSNE implementation
# Increasing n_iter may be necessary for 1000 points to stabilize
tsne = TSNE(n_components=2,)
y_tsne = tsne.fit(X_swiss)

# 3. Define Helper Function for Axes
def set_xyz_axes(ax, axes):
    ax.set_xlim3d(axes[0], axes[1])
    ax.set_ylim3d(axes[2], axes[3])
    ax.set_zlim3d(axes[4], axes[5])
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_zlabel("$x_3$", fontsize=12)

# 4. Setup Plot Configuration
darker_hot = ListedColormap(plt.cm.hot(np.linspace(0, 0.8, 256)))
axes_limits = [-11.5, 14, -2, 23, -12, 15]

# 5. Create Comparison Plot
fig = plt.figure(figsize=(12, 5))

# --- Left Plot: 3D Swiss Roll ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=t, cmap=darker_hot)
ax1.view_init(10, -70)
set_xyz_axes(ax1, axes_limits)
ax1.set_title("Original Swiss Roll (3D)")

# --- Right Plot: 2D t-SNE ---
ax2 = fig.add_subplot(122)
# Using 't' for colors ensures we see how the manifold was unrolled
ax2.scatter(y_tsne[:, 0], y_tsne[:, 1], c=t, cmap=darker_hot, s=15)
ax2.set_title("t-SNE Projection (2D)")
ax2.set_xlabel("$y_1$")
ax2.set_ylabel("$y_2$")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()