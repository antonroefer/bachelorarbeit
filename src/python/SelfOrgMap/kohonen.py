import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm


def plot_som(color, cmap, ax):
    """
    Function for plotting a hexagonal SOM map.

    Args:
        color (np.ndarray): A 1D array of color values for each hexagon.
        cmap (matplotlib.colors.Colormap): The colormap to use.
        ax (matplotlib.axes.Axes): The axes to plot on.
    """
    k = color.size
    sk = int(np.sqrt(k))
    color = color.flatten()

    # Hexagon pattern vertices
    x_hex = np.array([0, 0.5, 0.5, 0, -0.5, -0.5])
    y_hex = np.array(
        [
            np.sqrt((np.tan(np.deg2rad(30)) * 0.5) ** 2 + 0.5**2),
            np.tan(np.deg2rad(30)) * 0.5,
            -np.tan(np.deg2rad(30)) * 0.5,
            -np.sqrt((np.tan(np.deg2rad(30)) * 0.5) ** 2 + 0.5**2),
            -np.tan(np.deg2rad(30)) * 0.5,
            np.tan(np.deg2rad(30)) * 0.5,
        ]
    )

    # Calculate midpoints for a staggered grid
    xm = np.zeros(k)
    ym = np.zeros(k)
    for i in range(sk):  # rows
        for j in range(sk):  # cols
            idx = i * sk + j
            xm[idx] = j + 0.5 * (i % 2)
            ym[idx] = i * (y_hex[0] + np.tan(np.deg2rad(30)) * 0.5)

    # Plot each hexagon
    for i in range(k):
        ax.add_patch(
            Polygon(
                np.column_stack([x_hex + xm[i], y_hex + ym[i]]),
                facecolor=cmap(color[i]),
                edgecolor="gray",
            )
        )

    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_diff_som(map_weights, cmap):
    """
    Function for plotting the U-Matrix (Unified Distance Matrix) of a SOM.

    Args:
        map_weights (np.ndarray): The SOM weights of size (sk, sk, M).
        cmap (matplotlib.colors.Colormap): The colormap to use.
    """
    sk, _, M = map_weights.shape
    k = sk * sk

    # Hexagon pattern vertices
    x_hex = np.array([0, 0.5, 0.5, 0, -0.5, -0.5])
    y_hex = np.array(
        [
            np.sqrt((np.tan(np.deg2rad(30)) * 0.5) ** 2 + 0.5**2),
            np.tan(np.deg2rad(30)) * 0.5,
            -np.tan(np.deg2rad(30)) * 0.5,
            -np.sqrt((np.tan(np.deg2rad(30)) * 0.5) ** 2 + 0.5**2),
            -np.tan(np.deg2rad(30)) * 0.5,
            np.tan(np.deg2rad(30)) * 0.5,
        ]
    )

    # Calculate midpoints for a staggered grid
    xm = np.zeros(k)
    ym = np.zeros(k)
    for i in range(sk):
        for j in range(sk):
            idx = i * sk + j
            xm[idx] = j + 0.5 * (i % 2)
            ym[idx] = i * (y_hex[0] + np.tan(np.deg2rad(30)) * 0.5)

    # Calculate distances between neighboring nodes
    u_matrix = np.zeros((sk, sk))
    for i in range(sk):
        for j in range(sk):
            neighbors = []
            # Right neighbor
            if j + 1 < sk:
                neighbors.append(map_weights[i, j + 1, :])
            # Top-right neighbor
            if i + 1 < sk:
                if (i % 2) == 0:  # even rows
                    if j + 1 < sk:
                        neighbors.append(map_weights[i + 1, j + 1, :])
                else:  # odd rows
                    neighbors.append(map_weights[i + 1, j, :])
            # Top-left neighbor
            if i + 1 < sk:
                if (i % 2) == 0:  # even rows
                    neighbors.append(map_weights[i + 1, j, :])
                else:  # odd rows
                    if j - 1 >= 0:
                        neighbors.append(map_weights[i + 1, j - 1, :])

            if neighbors:
                distances = [
                    np.linalg.norm(map_weights[i, j, :] - n) for n in neighbors
                ]
                u_matrix[i, j] = np.mean(distances)

    fig, ax = plt.subplots()
    plot_som(u_matrix, cmap, ax)

    # Add text labels
    for i in range(k):
        ax.text(
            xm[i],
            ym[i],
            str(i + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=u_matrix.min(), vmax=u_matrix.max())
        ),
        ax=ax,
        label="Difference between cells",
    )
    ax.set_title("U-Matrix (Neighbor Distances)")


if __name__ == "__main__":
    import sys

    # --- Load and Prepare Data ---
    # Load features from feature_vectors.npz
    try:
        data = np.load("feature_vectors.npz")
        features = data["feature_stack"]
        names = data["feature_names"]
    except FileNotFoundError:
        print("feature_vectors.npz not found. Creating a dummy file for demonstration.")
        sys.exit()

    # Reshape features from (x, y, z) to (x*y, z) if necessary
    if features.ndim == 3:
        features = features.reshape(-1, features.shape[2])

    # --- SOM Initialization ---
    M = features.shape[1]  # number of features
    n = features.shape[0]  # number of data points
    k = 144  # number of nodes
    sk = int(np.sqrt(k))

    # Create coordinate grid for the map
    indices = np.indices((sk, sk))
    ind_a = indices[0]  # row indices
    ind_b = indices[1]  # col indices

    # Initialize the weight map with random values
    map_weights = np.random.rand(sk, sk, M)

    # --- SOM Training ---
    # Initialize parameters
    sigma0 = 0.5 * sk
    nu0 = 0.2
    tau0 = 1.0
    numit = 10  # number of iterations

    print("Training SOM...")
    for i in range(numit):
        # Decay parameters
        sigma = sigma0 * np.exp(-i / tau0)
        nu = nu0 * np.exp(-i / tau0)

        # Shuffle features for stochastic training
        np.random.shuffle(features)

        for j in tqdm(range(n), desc=f"Epoche {i + 1}/{numit}", leave=False):
            # Find Best Matching Unit (BMU)
            sample = features[j, :]
            dist = np.linalg.norm(sample - map_weights, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
            bpz_a, bpz_b = bmu_idx

            # Calculate distances from BMU to all other nodes in the grid
            entf = np.sqrt((bpz_a - ind_a) ** 2 + (bpz_b - ind_b) ** 2)

            # Calculate neighborhood influence (beta)
            beta = nu * np.exp(-(entf**2) / (2 * sigma**2))

            # Update weights
            # Reshape beta to (sk, sk, 1) for broadcasting
            map_weights += beta[:, :, np.newaxis] * (sample - map_weights)
        print(f"Iteration {i + 1}/{numit} complete.")

    # --- Post-Training Analysis ---
    # Calculate hit map (number of points per node)
    hits = np.zeros((sk, sk))
    bmu_indices = []

    # BMU Calculation: For each sample, find the Best Matching Unit (BMU)
    for j in tqdm(range(n), desc="BMU Calculation", leave=False):
        sample = features[j, :]
        dist = np.linalg.norm(sample - map_weights, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
        hits[bmu_idx] += 1
        bmu_indices.append(np.ravel_multi_index(bmu_idx, (sk, sk)))

    # --- Plotting Results ---
    # Plot Hit Map
    fig_hits, ax_hits = plt.subplots()
    plot_som(hits, plt.get_cmap("hot_r"), ax_hits)
    fig_hits.colorbar(
        plt.cm.ScalarMappable(
            cmap=plt.get_cmap("hot_r"),
            norm=plt.Normalize(vmin=hits.min(), vmax=hits.max()),
        ),
        ax=ax_hits,
        label="Number of Hits",
    )
    ax_hits.set_title("SOM Hit Histogram")

    # Plot U-Matrix
    plot_diff_som(map_weights, plt.get_cmap("hot_r"))

    # Plot Feature Planes
    num_features = map_weights.shape[2]
    fig_features, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i in range(num_features):
        feature_plane = map_weights[:, :, i]
        plot_som(feature_plane, plt.get_cmap("jet"), axes[i])
        fig_features.colorbar(
            plt.cm.ScalarMappable(
                cmap=plt.get_cmap("jet"),
                norm=plt.Normalize(vmin=feature_plane.min(), vmax=feature_plane.max()),
            ),
            ax=axes[i],
        )
        axes[i].set_title(names[i])

    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
