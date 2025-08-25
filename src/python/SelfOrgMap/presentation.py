from minisom import MiniSom
from sklearn.datasets import make_blobs, make_swiss_roll, make_moons
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt

# Create 3D blobs
# X, y = make_blobs(n_samples=300, n_features=3, centers=2, random_state=1, cluster_std=3)

# Generate 2D circles
# X_2d, y = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=1)

"""
Datengenerierung
"""

# Generate 2D Moons
X_2d, y = make_moons(n_samples=300, noise=0.05, random_state=3)
scaler = MinMaxScaler(feature_range=(0.2, 0.8))
X_2d = scaler.fit_transform(X_2d)
z = ((-X_2d[:, 0] + X_2d[:, 1]) / 2).reshape(-1, 1)
X = np.column_stack([X_2d, z])

"""
Initialisierung der Self-Organizing Map
"""


som = MiniSom(
    x=12,
    y=12,
    input_len=3,
    sigma=5,
    learning_rate=0.3,
    decay_function="asymptotic_decay",
    sigma_decay_function="asymptotic_decay",
    neighborhood_function="gaussian",
    topology="hexagonal",
)
som.normalize_random_weights_init(X)


"""
Training
"""

som.train(X, 100, use_epochs=True, verbose=True)

"""
Darstellung
"""

fig = plt.figure(figsize=(8, 6))
# Draw neighbor connections between SOM neurons
ax = fig.add_subplot(111, projection="3d")

# # Get neuron positions in grid
# grid_x, grid_y = som._weights.shape[:2]
# # Plot hexagonal neighborhood connections
# for i, j in product(range(grid_x), range(grid_y)):
#     neuron_pos = som.get_weights()[i, j]
#     # Hexagonal grid neighbors (even-q vertical layout)
#     if j % 2 == 0:
#         neighbors = [
#             (i - 1, j),  # left
#             (i + 1, j),  # right
#             (i, j - 1),  # top
#             (i, j + 1),  # bottom
#             (i - 1, j - 1),  # top-left
#             (i - 1, j + 1),  # bottom-left
#         ]
#     else:
#         neighbors = [
#             (i - 1, j),  # left
#             (i + 1, j),  # right
#             (i, j - 1),  # top
#             (i, j + 1),  # bottom
#             (i + 1, j - 1),  # top-right
#             (i + 1, j + 1),  # bottom-right
#         ]
#     for ni, nj in neighbors:
#         if 0 <= ni < grid_x and 0 <= nj < grid_y:
#             neighbor_pos = som.get_weights()[ni, nj]
#             ax.plot(
#                 [neuron_pos[0], neighbor_pos[0]],
#                 [neuron_pos[1], neighbor_pos[1]],
#                 [neuron_pos[2], neighbor_pos[2]],
#                 color="gray",
#                 alpha=0.5,
#                 linewidth=1,
#             )
# # Get neuron weights and plot them
# weights = som.get_weights().reshape(-1, 3)
# ax.scatter(
#     weights[:, 0],
#     weights[:, 1],
#     weights[:, 2],
#     c="red",
#     marker="x",
#     s=80,
#     label="SOM Neurons",
# )
# ax.legend()
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=50, alpha=0.4)
ax.set_xlabel("Merkmal 1")
ax.set_ylabel("Merkmal 2")
ax.set_zlabel("Merkmal 3")
plt.title("Untrainierte SOM")

som.plot_som_planes(fnames=["Merkmal 1", "Merkmal 2", "Merkmal 3"], save=False)
som.plot_u_matrix(save=False)
som.plot_som_hits(data=X, save=False)
plt.show()
