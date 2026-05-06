"""Analyze subgoal transitions by extracting keyframe embeddings and their trajectories."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

sns.set_style("whitegrid")
os.environ["STABLEWM_HOME"] = "/home/mahdi/workspace/le-wm/data"


def load_embeddings(embedding_path: str = "embeddings/rlbench/embeddings.npz"):
    """Load embeddings, timesteps, and episode information."""
    data = np.load(embedding_path)
    embeddings = data["embeddings"]  # (N, D)
    timesteps = data["timesteps"]  # (N,) - values like 0, 1, 2, etc.
    episodes = data["episodes"]  # (N,) - episode indices
    return embeddings, timesteps, episodes


def create_keypoint_vector(timesteps: np.ndarray) -> np.ndarray:
    """
    Create keypoint vector marking frames where timestep changes.

    For frames where timestep changes: keypoint = new_timestep
    For other frames: keypoint = 0
    """
    keypoint = np.zeros_like(timesteps)
    # Find where timestep changes
    timestep_changes = np.diff(timesteps, prepend=timesteps[0])
    change_mask = timestep_changes != 0
    keypoint[change_mask] = timesteps[change_mask]
    return keypoint


def extract_subgoal_frames(
    embeddings: np.ndarray,
    timesteps: np.ndarray,
    episodes: np.ndarray,
    keypoint: np.ndarray,
) -> dict:
    """Extract frames where timestep transitions occur (keypoint > 0)."""
    subgoal_mask = keypoint > 0
    subgoal_embeddings = embeddings[subgoal_mask]
    subgoal_timesteps = timesteps[subgoal_mask]
    subgoal_episodes = episodes[subgoal_mask]
    subgoal_indices = np.where(subgoal_mask)[0]  # Original indices in full array

    return {
        "embeddings": subgoal_embeddings,
        "timesteps": subgoal_timesteps,
        "episodes": subgoal_episodes,
        "indices": subgoal_indices,
    }


def fit_pca(subgoal_embeddings: np.ndarray, n_components: int = 2) -> tuple:
    """Fit PCA on subgoal embeddings and return model + transformed data."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(subgoal_embeddings)

    pca = PCA(n_components=n_components)
    subgoal_pca = pca.fit_transform(embeddings_scaled)

    return pca, scaler, subgoal_pca


def analyze_trajectories(
    embeddings: np.ndarray,
    timesteps: np.ndarray,
    episodes: np.ndarray,
    keypoint: np.ndarray,
    pca: PCA,
    scaler: StandardScaler,
    output_dir: str = "analysis/subgoals",
):
    """
    Analyze if frames between subgoals form meaningful trajectories in PCA space.

    For each subgoal transition, extract the intermediate frames and project them
    into the fitted PCA space to visualize the trajectory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Project all embeddings into PCA space
    embeddings_scaled = scaler.transform(embeddings)
    all_pca = pca.transform(embeddings_scaled)

    # Find subgoal indices
    subgoal_indices = np.where(keypoint > 0)[0]

    print(f"Found {len(subgoal_indices)} subgoal transition points")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    # Collect trajectories between consecutive subgoals
    trajectories = []
    for i in range(len(subgoal_indices) - 1):
        start_idx = subgoal_indices[i]
        end_idx = subgoal_indices[i + 1]

        # Get intermediate frames (excluding endpoints to avoid overlap)
        traj_indices = np.arange(start_idx, end_idx + 1)
        traj_timesteps = timesteps[traj_indices]
        traj_episodes = episodes[traj_indices]

        # Only include if within same episode
        if len(np.unique(traj_episodes)) == 1:
            trajectories.append(
                {
                    "indices": traj_indices,
                    "pca_coords": all_pca[traj_indices],
                    "timesteps": traj_timesteps,
                    "start_timestep": timesteps[start_idx],
                    "end_timestep": timesteps[end_idx],
                    "episode": traj_episodes[0],
                }
            )

    return all_pca, subgoal_indices, trajectories


def plot_subgoal_space(
    all_pca: np.ndarray,
    subgoal_indices: np.ndarray,
    timesteps: np.ndarray,
    output_dir: str = "analysis/subgoals",
):
    """Plot subgoal frames in PCA space colored by timestep."""
    plt.figure(figsize=(12, 8))

    # Plot all frames as background
    plt.scatter(
        all_pca[:, 0],
        all_pca[:, 1],
        c=timesteps,
        alpha=0.2,
        s=10,
        cmap="tab20",
        label="All frames",
    )

    # Plot subgoal frames prominently
    subgoal_pca = all_pca[subgoal_indices]
    subgoal_timesteps = timesteps[subgoal_indices]
    plt.scatter(
        subgoal_pca[:, 0],
        subgoal_pca[:, 1],
        c=subgoal_timesteps,
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        cmap="tab20",
        label="Subgoal transitions",
        zorder=10,
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Embeddings in PCA Space - Subgoal Transitions")
    plt.colorbar(label="Timestep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/subgoal_pca_space.png", dpi=150)
    print(f"Saved: {output_dir}/subgoal_pca_space.png")
    plt.close()


def plot_trajectories(
    trajectories: list,
    output_dir: str = "analysis/subgoals",
    max_plots: int = 20,
):
    """Plot sample trajectories in PCA space."""
    n_plots = min(len(trajectories), max_plots)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i, traj in enumerate(trajectories[:n_plots]):
        ax = axes[i]
        pca_coords = traj["pca_coords"]

        # Plot trajectory line
        ax.plot(
            pca_coords[:, 0],
            pca_coords[:, 1],
            "o-",
            alpha=0.6,
            markersize=4,
            linewidth=2,
        )

        # Mark start and end
        ax.plot(pca_coords[0, 0], pca_coords[0, 1], "go", markersize=10, label="Start")
        ax.plot(pca_coords[-1, 0], pca_coords[-1, 1], "r*", markersize=15, label="End")

        # Add timestep labels
        start_ts = traj["start_timestep"]
        end_ts = traj["end_timestep"]
        ax.set_title(
            f"Episode {int(traj['episode'])}: Timestep {int(start_ts)}→{int(end_ts)}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sample_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/sample_trajectories.png")
    plt.close()


def plot_trajectories_by_episode(
    all_pca: np.ndarray,
    trajectories: list,
    subgoal_indices: np.ndarray,
    subgoal_timesteps: np.ndarray,
    timesteps: np.ndarray,
    episodes: np.ndarray,
    output_dir: str = "analysis/subgoals",
):
    """
    Plot all trajectories by episode in one plot.

    Each trajectory is colored by timestep indices, with subgoal locations marked.
    """
    # Get unique episodes that have trajectories
    episode_list = sorted(set(traj["episode"] for traj in trajectories))

    # Create one plot per episode (or multiple if many episodes)
    episodes_per_plot = 4
    n_plots = (len(episode_list) + episodes_per_plot - 1) // episodes_per_plot

    for plot_idx in range(n_plots):
        start_ep_idx = plot_idx * episodes_per_plot
        end_ep_idx = min(start_ep_idx + episodes_per_plot, len(episode_list))
        episodes_to_plot = episode_list[start_ep_idx:end_ep_idx]

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for subplot_idx, episode_id in enumerate(episodes_to_plot):
            ax = axes[subplot_idx]

            # Get subgoals for this episode
            episode_mask = episodes == episode_id
            episode_subgoal_indices = subgoal_indices[episode_mask[subgoal_indices]]
            episode_subgoal_pca = all_pca[episode_subgoal_indices]
            episode_subgoal_timesteps = subgoal_timesteps[episode_mask[subgoal_indices]]

            # Get trajectories for this episode
            episode_trajs = [t for t in trajectories if t["episode"] == episode_id]

            # Plot each trajectory with color gradient based on timestep
            cmap = plt.cm.get_cmap(
                "tab20"
                if len(set(t["start_timestep"] for t in episode_trajs)) <= 20
                else "hsv"
            )

            for traj in episode_trajs:
                pca_coords = traj["pca_coords"]
                timesteps_in_traj = traj["timesteps"]

                # Normalize timesteps for color mapping
                min_ts = timesteps_in_traj.min()
                max_ts = timesteps_in_traj.max()
                if max_ts > min_ts:
                    ts_normalized = (timesteps_in_traj - min_ts) / (max_ts - min_ts)
                else:
                    ts_normalized = np.ones_like(timesteps_in_traj) * 0.5

                # Plot trajectory segments colored by timestep
                for i in range(len(pca_coords) - 1):
                    color = cmap(ts_normalized[i])
                    ax.plot(
                        pca_coords[i : i + 2, 0],
                        pca_coords[i : i + 2, 1],
                        color=color,
                        linewidth=2,
                        alpha=0.7,
                        zorder=2,
                    )

            # Plot subgoal locations prominently
            ax.scatter(
                episode_subgoal_pca[:, 0],
                episode_subgoal_pca[:, 1],
                c=episode_subgoal_timesteps,
                s=400,
                marker="*",
                edgecolors="black",
                linewidths=2.5,
                cmap="tab20",
                zorder=10,
                label="Subgoal transitions",
            )

            # Add subgoal timestep labels
            for i, (x, y) in enumerate(episode_subgoal_pca):
                ax.annotate(
                    f"{int(episode_subgoal_timesteps[i])}",
                    (x, y),
                    fontsize=9,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="white",
                    zorder=11,
                )

            ax.set_xlabel("PC1", fontsize=11)
            ax.set_ylabel("PC2", fontsize=11)
            ax.set_title(
                f"Episode {int(episode_id)} - All Trajectories",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, zorder=0)
            ax.legend(loc="best")

        # Hide unused subplots
        for j in range(len(episodes_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        save_path = f"{output_dir}/trajectories_by_episode_part{plot_idx + 1}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        plt.close()

    # Create a summary plot with all episodes overlaid (if not too many)
    if len(episode_list) <= 10:
        fig, ax = plt.subplots(figsize=(14, 10))

        cmap = plt.cm.get_cmap("tab20")

        for traj_idx, traj in enumerate(trajectories):
            pca_coords = traj["pca_coords"]
            timesteps_in_traj = traj["timesteps"]
            episode_id = traj["episode"]

            # Normalize timesteps for color mapping
            min_ts = timesteps_in_traj.min()
            max_ts = timesteps_in_traj.max()
            if max_ts > min_ts:
                ts_normalized = (timesteps_in_traj - min_ts) / (max_ts - min_ts)
            else:
                ts_normalized = np.ones_like(timesteps_in_traj) * 0.5

            # Plot trajectory segments with alpha based on episode
            for i in range(len(pca_coords) - 1):
                color = cmap(ts_normalized[i])
                ax.plot(
                    pca_coords[i : i + 2, 0],
                    pca_coords[i : i + 2, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.5,
                    zorder=2,
                )

        # Plot all subgoals
        ax.scatter(
            all_pca[subgoal_indices, 0],
            all_pca[subgoal_indices, 1],
            c=subgoal_timesteps,
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            cmap="tab20",
            zorder=10,
            label="Subgoal transitions",
        )

        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_title(
            "All Trajectories - All Episodes Overlaid", fontsize=13, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(loc="best", fontsize=11)

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/all_trajectories_overlay.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {output_dir}/all_trajectories_overlay.png")
        plt.close()


def analyze_trajectory_smoothness(trajectories: list) -> dict:
    """
    Analyze trajectory properties to assess meaningfulness.

    Metrics:
    - Path length: total distance traveled
    - Direct distance: distance between start and end
    - Tortuosity: path length / direct distance (closer to 1 = more direct)
    """
    metrics = []

    for traj in trajectories:
        pca_coords = traj["pca_coords"]

        # Calculate distances between consecutive points
        deltas = np.diff(pca_coords, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)
        path_length = np.sum(segment_lengths)

        # Direct distance from start to end
        direct_distance = np.linalg.norm(pca_coords[-1] - pca_coords[0])

        # Avoid division by zero
        if direct_distance > 1e-6:
            tortuosity = path_length / direct_distance
        else:
            tortuosity = 0

        # Average step size
        avg_step = np.mean(segment_lengths) if len(segment_lengths) > 0 else 0

        metrics.append(
            {
                "path_length": path_length,
                "direct_distance": direct_distance,
                "tortuosity": tortuosity,
                "avg_step": avg_step,
                "n_frames": len(pca_coords),
                "start_timestep": traj["start_timestep"],
                "end_timestep": traj["end_timestep"],
            }
        )

    return metrics


def plot_trajectory_statistics(metrics: list, output_dir: str = "analysis/subgoals"):
    """Plot statistics about trajectory smoothness and meaningfulness."""
    if not metrics:
        print("No trajectory metrics to plot")
        return

    metrics_array = np.array(
        [
            [m["path_length"] for m in metrics],
            [m["direct_distance"] for m in metrics],
            [m["tortuosity"] for m in metrics],
            [m["avg_step"] for m in metrics],
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Path length
    axes[0, 0].hist(metrics_array[0], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Path Length")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Path Lengths")
    axes[0, 0].grid(True, alpha=0.3)

    # Direct distance
    axes[0, 1].hist(
        metrics_array[1], bins=30, alpha=0.7, edgecolor="black", color="orange"
    )
    axes[0, 1].set_xlabel("Direct Distance")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Direct Distances")
    axes[0, 1].grid(True, alpha=0.3)

    # Tortuosity (path_length / direct_distance)
    axes[1, 0].hist(
        metrics_array[2], bins=30, alpha=0.7, edgecolor="black", color="green"
    )
    axes[1, 0].set_xlabel("Tortuosity (Path/Direct)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Tortuosity (1 = straight line)")
    axes[1, 0].grid(True, alpha=0.3)

    # Average step size
    axes[1, 1].hist(
        metrics_array[3], bins=30, alpha=0.7, edgecolor="black", color="red"
    )
    axes[1, 1].set_xlabel("Average Step Size")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Average Step Sizes")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_statistics.png", dpi=150)
    print(f"Saved: {output_dir}/trajectory_statistics.png")
    plt.close()

    # Print summary statistics
    print("\n=== Trajectory Metrics Summary ===")
    print(f"Mean path length: {np.mean(metrics_array[0]):.4f}")
    print(f"Mean direct distance: {np.mean(metrics_array[1]):.4f}")
    print(f"Mean tortuosity: {np.mean(metrics_array[2]):.4f}")
    print(f"Mean avg step size: {np.mean(metrics_array[3]):.4f}")


def train_subgoal_classifier(
    embeddings: np.ndarray,
    keypoint: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "analysis/subgoals",
) -> dict:
    """
    Train a classifier to predict subgoal transitions from embeddings.

    Classes: 0 = no transition, 1, 2, 3, ... = transition to subgoal N
    """
    print("\n=== Training Subgoal Classifier ===")

    # Prepare data: labels are the keypoint values
    X = embeddings
    y = keypoint.astype(int)

    # Downsample class 0 (no transition) to match transition frames
    class_0_mask = y == 0
    class_nonzero_mask = y > 0

    n_transitions = np.sum(class_nonzero_mask)
    n_no_transitions = np.sum(class_0_mask)

    print(f"Original class distribution:")
    print(f"  Class 0 (no transition): {n_no_transitions}")
    print(f"  Classes 1+ (transitions): {n_transitions}")

    # Downsample class 0 to match transition count
    class_0_indices = np.where(class_0_mask)[0]
    np.random.seed(random_state)
    downsampled_class_0_indices = np.random.choice(
        class_0_indices, size=n_transitions, replace=False
    )

    # Combine downsampled class 0 with all transitions
    balanced_indices = np.concatenate([downsampled_class_0_indices, np.where(class_nonzero_mask)[0]])
    np.random.shuffle(balanced_indices)

    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    print(f"\nAfter downsampling:")
    print(f"  Class 0 (no transition): {np.sum(y_balanced == 0)}")
    print(f"  Classes 1+ (transitions): {np.sum(y_balanced > 0)}")
    print(f"  Total balanced samples: {len(y_balanced)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state, stratify=y_balanced
    )

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Classes: {sorted(np.unique(y_balanced))}")
    print(f"Balanced training set class distribution:")
    for cls in sorted(np.unique(y_balanced)):
        count = np.sum(y_balanced == cls)
        pct = 100 * count / len(y_balanced)
        print(f"  Class {cls}: {count} ({pct:.1f}%)")

    # Train random forest classifier with balanced class weights
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    print("\nTraining classifier...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(
        "\nNote: With imbalanced data, accuracy is misleading. See classification report below."
    )
    print("\nPer-class performance on test set:")
    print(classification_report(y_test, y_pred_test))

    # Feature importance
    feature_importance = clf.feature_importances_
    top_features = np.argsort(feature_importance)[-10:]
    print(f"\nTop 10 important features (dimensions): {top_features}")
    print(f"Top 10 importance scores: {feature_importance[top_features]}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Subgoal Classifier - Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    print(f"Saved: {output_dir}/confusion_matrix.png")
    plt.close()

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_n = 20
    top_indices = np.argsort(feature_importance)[-top_n:]
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.xlabel("Importance")
    plt.ylabel("Dimension")
    plt.title(f"Top {top_n} Important Embedding Dimensions")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=150)
    print(f"Saved: {output_dir}/feature_importance.png")
    plt.close()

    return {
        "model": clf,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "confusion_matrix": cm,
        "feature_importance": feature_importance,
    }


def visualize_subgoal_frames(
    keypoint: np.ndarray,
    episodes: np.ndarray,
    timesteps: np.ndarray,
    dataset_name: str = "rlbench/rlbench",
    output_dir: str = "analysis/subgoals",
    num_episodes_to_show: int = 5,
):
    """
    Visualize actual frames corresponding to subgoal transitions.

    Shows frames where transitions occur to verify consistency.
    """
    import stable_worldmodel as swm

    print("\n=== Visualizing Subgoal Frames ===")

    # Load dataset to get frames
    dataset = swm.data.HDF5Dataset(dataset_name, num_steps=4, frameskip=5)

    # Find subgoal frames per episode
    subgoal_mask = keypoint > 0
    subgoal_indices = np.where(subgoal_mask)[0]
    subgoal_episodes = episodes[subgoal_indices]
    subgoal_timesteps = timesteps[subgoal_indices]

    # Group by episode
    unique_episodes = sorted(np.unique(episodes))[:num_episodes_to_show]

    for ep_idx in unique_episodes:
        ep_idx = int(ep_idx)
        try:
            episode_data = dataset.load_episode(ep_idx)
            frames = episode_data["pixels"]  # (T, C, H, W)

            # Find subgoals for this episode
            ep_mask = subgoal_episodes == ep_idx
            ep_subgoal_indices_global = subgoal_indices[ep_mask]
            ep_subgoal_timesteps = subgoal_timesteps[ep_mask]

            # Map global indices to local indices within this episode
            # This requires knowing where this episode starts in the global array
            ep_start_idx = np.where(episodes == ep_idx)[0][0]
            ep_end_idx = np.where(episodes == ep_idx)[0][-1] + 1
            ep_length = ep_end_idx - ep_start_idx

            ep_subgoal_local_indices = ep_subgoal_indices_global - ep_start_idx

            # Get frames for each subgoal (and neighbors for context)
            context_frames = 2
            fig_rows = len(ep_subgoal_local_indices)
            fig_cols = 2 * context_frames + 1

            fig, axes = plt.subplots(
                fig_rows, fig_cols, figsize=(fig_cols * 2, fig_rows * 2)
            )
            if fig_rows == 1:
                axes = axes.reshape(1, -1)

            for row_idx, (frame_idx, ts_val) in enumerate(
                zip(ep_subgoal_local_indices, ep_subgoal_timesteps)
            ):
                frame_idx = int(frame_idx)
                # Get context frames
                start_idx = max(0, frame_idx - context_frames)
                end_idx = min(len(frames), frame_idx + context_frames + 1)

                context_indices = np.arange(start_idx, end_idx)

                for col_idx, ctx_idx in enumerate(context_indices):
                    if col_idx < fig_cols:
                        ax = axes[row_idx, col_idx]

                        frame = frames[ctx_idx]
                        # Convert (C, H, W) to (H, W, C) for display
                        if frame.shape[0] == 3:
                            frame = np.transpose(frame, (1, 2, 0))

                        # Normalize to 0-1 if needed
                        if frame.max() > 1.0:
                            frame = frame / 255.0

                        ax.imshow(frame)

                        if ctx_idx == frame_idx:
                            ax.set_title(
                                f"Subgoal {int(ts_val)}", fontweight="bold", color="red"
                            )
                            ax.set_facecolor("yellow")
                        else:
                            ax.set_title(f"Frame {ctx_idx}")

                        ax.axis("off")

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/subgoal_frames_episode_{ep_idx}.png",
                dpi=100,
                bbox_inches="tight",
            )
            print(f"Saved: {output_dir}/subgoal_frames_episode_{ep_idx}.png")
            plt.close()

        except Exception as e:
            print(f"Error loading episode {ep_idx}: {e}")


def main(
    embedding_path: str = "embeddings/rlbench/embeddings.npz",
    output_dir: str = "analysis/subgoals",
    n_components: int = 2,
):
    """Main analysis pipeline."""
    print("Loading embeddings...")
    embeddings, timesteps, episodes = load_embeddings(embedding_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Timesteps: {timesteps.shape}, unique: {len(np.unique(timesteps))}")
    print(f"Episodes: {episodes.shape}, unique: {len(np.unique(episodes))}")

    print("\nCreating keypoint vector...")
    keypoint = create_keypoint_vector(timesteps)
    n_subgoals = np.sum(keypoint > 0)
    print(f"Found {n_subgoals} subgoal transitions")

    print("\nExtracting subgoal frames...")
    subgoal_data = extract_subgoal_frames(embeddings, timesteps, episodes, keypoint)
    print(f"Subgoal embeddings: {subgoal_data['embeddings'].shape}")

    print("\nFitting PCA on subgoal frames...")
    pca, scaler, subgoal_pca = fit_pca(
        subgoal_data["embeddings"], n_components=n_components
    )

    print("\nAnalyzing trajectories between subgoals...")
    all_pca, subgoal_indices, trajectories = analyze_trajectories(
        embeddings, timesteps, episodes, keypoint, pca, scaler, output_dir
    )
    print(f"Found {len(trajectories)} trajectories between subgoals")

    print("\nComputing trajectory metrics...")
    metrics = analyze_trajectory_smoothness(trajectories)

    print("\nGenerating visualizations...")
    plot_subgoal_space(all_pca, subgoal_indices, timesteps, output_dir)
    plot_trajectories(trajectories, output_dir)
    plot_trajectory_statistics(metrics, output_dir)

    print("\nTraining subgoal classifier...")
    classifier_results = train_subgoal_classifier(
        embeddings, keypoint, output_dir=output_dir
    )

    print("\nVisualizing subgoal frames...")
    visualize_subgoal_frames(
        keypoint, episodes, timesteps, output_dir=output_dir, num_episodes_to_show=5
    )

    # Save analysis data
    analysis_data = {
        "keypoint": keypoint,
        "subgoal_embeddings": subgoal_data["embeddings"],
        "subgoal_timesteps": subgoal_data["timesteps"],
        "subgoal_indices": subgoal_data["indices"],
        "subgoal_pca": subgoal_pca,
        "all_pca": all_pca,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "pca_components": pca.components_,
    }

    analysis_path = Path(output_dir) / "analysis_data.npz"
    np.savez(analysis_path, **analysis_data)
    print(f"\nSaved analysis data: {analysis_path}")

    # Save metrics as numpy array for further analysis
    metrics_path = Path(output_dir) / "trajectory_metrics.npy"
    np.save(metrics_path, metrics)
    print(f"Saved trajectory metrics: {metrics_path}")


if __name__ == "__main__":
    main()
