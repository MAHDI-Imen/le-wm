"""Create dimensionality reduction plots and analysis visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_embeddings(embeddings_file: str) -> tuple:
    """Load embeddings and metadata from file."""
    data = np.load(embeddings_file)
    embeddings = data["embeddings"]
    timesteps = data["timesteps"].astype(int)
    episodes = data["episodes"].astype(int)
    return embeddings, timesteps, episodes


def plot_single_method(
    reduced_data: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    output_path: str,
):
    """Create a single scatter plot for a dimensionality reduction method."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=labels,
        cmap="viridis",
        s=6,
        alpha=0.6,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Timestep (within episode)", fontsize=11)

    ax.set_title(f"LeWM latent space — {method_name.upper()}", fontsize=13)
    ax.set_xlabel(f"{method_name.upper()} dim 1", fontsize=11)
    ax.set_ylabel(f"{method_name.upper()} dim 2", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {output_path}")
    plt.close()


def plot_comprehensive_umap(
    reduced: np.ndarray,
    timesteps: np.ndarray,
    episodes: np.ndarray,
    output_path: str,
):
    """Create comprehensive 4-panel UMAP visualization."""
    num_episodes = episodes.max() + 1
    num_timesteps = timesteps.max() + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Colored by timestep
    sc1 = axes[0, 0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=timesteps,
        cmap="viridis",
        s=5,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0, 0].set_title(
        "Embeddings colored by Timestep (within episode)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("UMAP dim 1")
    axes[0, 0].set_ylabel("UMAP dim 2")
    cbar1 = fig.colorbar(sc1, ax=axes[0, 0])
    cbar1.set_label("Timestep")

    # 2. Colored by episode
    sc2 = axes[0, 1].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=episodes,
        cmap="tab10",
        s=5,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0, 1].set_title(
        "Embeddings colored by Episode", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("UMAP dim 1")
    axes[0, 1].set_ylabel("UMAP dim 2")
    cbar2 = fig.colorbar(sc2, ax=axes[0, 1])
    cbar2.set_label("Episode ID")

    # 3. Plot trajectories - connect timesteps within each episode
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_episodes, 10)))
    for ep_id in range(min(num_episodes, 10)):  # Limit to 10 episodes for clarity
        mask = episodes == ep_id
        ep_reduced = reduced[mask]
        ax.plot(
            ep_reduced[:, 0],
            ep_reduced[:, 1],
            "o-",
            color=colors[ep_id],
            label=f"Ep {ep_id}",
            alpha=0.7,
            markersize=6,
            linewidth=2,
        )
    ax.set_title(
        "Episode Trajectories in Embedding Space", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)

    # 4. Scatter colored by timestep with separate legend
    ax = axes[1, 1]
    for t in range(int(num_timesteps)):
        mask = timesteps == t
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"T={t}",
            s=5,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
    ax.set_title("Timestep Clusters in Embedding Space", fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comprehensive UMAP analysis → {output_path}")
    plt.close()


def plot_methods_comparison(
    embeddings: np.ndarray,
    timesteps: np.ndarray,
    output_path: str,
):
    """Create comparison visualization of PCA, UMAP, and t-SNE."""
    print("\nReducing dimensionality with multiple methods...")

    # PCA
    print("  Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca_reduced = pca.fit_transform(embeddings)
    print(
        f"  PCA explained variance: {pca.explained_variance_ratio_}\n"
        f"  Total: {pca.explained_variance_ratio_.sum():.2%}"
    )

    # UMAP
    print("  Running UMAP...")
    reducer = umap.UMAP(n_components=2, verbose=False)
    umap_reduced = reducer.fit_transform(embeddings)

    # t-SNE
    print("  Running t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, verbose=1)
    tsne_reduced = tsne.fit_transform(embeddings)

    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    methods = [("UMAP", umap_reduced), ("PCA", pca_reduced), ("t-SNE", tsne_reduced)]

    for idx, (method_name, reduced_data) in enumerate(methods):
        ax = axes[idx]
        sc = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=timesteps,
            cmap="viridis",
            s=5,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )
        ax.set_title(
            f"{method_name} - Colored by Timestep", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel(f"{method_name} dim 1")
        ax.set_ylabel(f"{method_name} dim 2")
        fig.colorbar(sc, ax=ax, label="Timestep")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved methods comparison → {output_path}\n")
    plt.close()

    return umap_reduced, pca_reduced, tsne_reduced


def plot_comprehensive_tsne(
    reduced: np.ndarray,
    timesteps: np.ndarray,
    episodes: np.ndarray,
    output_path: str,
):
    """Create comprehensive 4-panel t-SNE visualization."""
    num_episodes = episodes.max() + 1
    num_timesteps = timesteps.max() + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Colored by timestep
    sc1 = axes[0, 0].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=timesteps,
        cmap="viridis",
        s=5,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0, 0].set_title(
        "Embeddings colored by Timestep (within episode)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0, 0].set_xlabel("t-SNE dim 1")
    axes[0, 0].set_ylabel("t-SNE dim 2")
    cbar1 = fig.colorbar(sc1, ax=axes[0, 0])
    cbar1.set_label("Timestep")

    # 2. Colored by episode
    sc2 = axes[0, 1].scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=episodes,
        cmap="tab10",
        s=5,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    axes[0, 1].set_title(
        "Embeddings colored by Episode", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("t-SNE dim 1")
    axes[0, 1].set_ylabel("t-SNE dim 2")
    cbar2 = fig.colorbar(sc2, ax=axes[0, 1])
    cbar2.set_label("Episode ID")

    # 3. Plot trajectories - connect timesteps within each episode
    ax = axes[1, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_episodes, 10)))
    for ep_id in range(min(num_episodes, 10)):  # Limit to 10 episodes for clarity
        mask = episodes == ep_id
        ep_tsne = reduced[mask]
        ax.plot(
            ep_tsne[:, 0],
            ep_tsne[:, 1],
            "o-",
            color=colors[ep_id],
            label=f"Ep {ep_id}",
            alpha=0.7,
            markersize=6,
            linewidth=2,
        )
    ax.set_title(
        "Episode Trajectories in Embedding Space", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)

    # 4. Scatter colored by timestep with separate legend
    ax = axes[1, 1]
    for t in range(num_timesteps):
        mask = timesteps == t
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f"T={t}",
            s=5,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
        )
    ax.set_title("Timestep Clusters in Embedding Space", fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comprehensive t-SNE analysis → {output_path}")
    plt.close()


def create_all_visualizations(
    embeddings_file: str = "embeddings/rlbench/embeddings.npz",
    output_dir: str = "analysis",
):
    """Create all dimensionality reduction and analysis visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings, timesteps, episodes = load_embeddings(embeddings_file)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique timesteps: {np.unique(timesteps)}")
    print(f"  Unique episodes: {np.unique(episodes)}\n")

    # Run all reduction methods
    umap_reduced, pca_reduced, tsne_reduced = plot_methods_comparison(
        embeddings, timesteps, f"{output_dir}/methods_comparison.png"
    )

    # Create comprehensive visualizations
    plot_comprehensive_umap(
        umap_reduced, timesteps, episodes, f"{output_dir}/umap_comprehensive.png"
    )
    plot_comprehensive_tsne(
        tsne_reduced, timesteps, episodes, f"{output_dir}/tsne_comprehensive.png"
    )

    # Create single method plots
    plot_single_method(umap_reduced, timesteps, "UMAP", f"{output_dir}/umap.png")
    plot_single_method(pca_reduced, timesteps, "PCA", f"{output_dir}/pca.png")
    plot_single_method(tsne_reduced, timesteps, "t-SNE", f"{output_dir}/tsne.png")

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_all_visualizations()
