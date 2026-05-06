"""Train a classifier to predict timestep from embeddings using 90% training data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import seaborn as sns
except ImportError:
    sns = None


def load_embeddings(embeddings_file: str) -> tuple:
    """Load embeddings and metadata from file."""
    data = np.load(embeddings_file)
    embeddings = data["embeddings"]
    timesteps = data["timesteps"].astype(int)
    episodes = data["episodes"].astype(int)
    return embeddings, timesteps, episodes


def train_timestep_classifier(
    embeddings_file: str = "embeddings/rlbench/embeddings.npz",
    train_split: float = 0.9,
    output_dir: str = "classifier",
    random_state: int = 42,
):
    """Train a Random Forest classifier to predict timestep from embeddings."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings, timesteps, episodes = load_embeddings(embeddings_file)

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique timesteps: {len(np.unique(timesteps))}")
    print(f"  Unique episodes: {len(np.unique(episodes))}\n")

    # Split by episodes: use train_split (default 90%) for training
    unique_episodes = np.unique(episodes)
    num_episodes = len(unique_episodes)
    num_train_episodes = int(num_episodes * train_split)

    train_episode_ids = unique_episodes[:num_train_episodes]
    eval_episode_ids = unique_episodes[num_train_episodes:]

    # Create masks
    train_mask = np.isin(episodes, train_episode_ids)
    eval_mask = np.isin(episodes, eval_episode_ids)

    # Split data
    train_emb = embeddings[train_mask]
    train_timesteps = timesteps[train_mask]

    eval_emb = embeddings[eval_mask]
    eval_timesteps = timesteps[eval_mask]

    print("=" * 70)
    print("TRAIN/EVAL SPLIT FOR TIMESTEP CLASSIFICATION")
    print("=" * 70)
    print(
        f"Training episodes: {len(train_episode_ids)} ({num_train_episodes/num_episodes*100:.0f}%)"
    )
    print(
        f"Eval episodes: {len(eval_episode_ids)} ({(num_episodes - num_train_episodes)/num_episodes*100:.0f}%)"
    )
    print(f"Training samples: {len(train_emb)}")
    print(f"Eval samples: {len(eval_emb)}")
    print(f"Number of timestep classes: {len(np.unique(timesteps))}\n")

    # Train classifier
    print("Training Random Forest classifier on training embeddings...")
    classifier = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=-1, verbose=0
    )
    classifier.fit(train_emb, train_timesteps)

    # Predictions
    train_pred = classifier.predict(train_emb)
    eval_pred = classifier.predict(eval_emb)

    # Compute metrics
    train_accuracy = accuracy_score(train_timesteps, train_pred)
    train_precision = precision_score(
        train_timesteps, train_pred, average="weighted", zero_division=0
    )
    train_recall = recall_score(
        train_timesteps, train_pred, average="weighted", zero_division=0
    )
    train_f1 = f1_score(
        train_timesteps, train_pred, average="weighted", zero_division=0
    )

    eval_accuracy = accuracy_score(eval_timesteps, eval_pred)
    eval_precision = precision_score(
        eval_timesteps, eval_pred, average="weighted", zero_division=0
    )
    eval_recall = recall_score(
        eval_timesteps, eval_pred, average="weighted", zero_division=0
    )
    eval_f1 = f1_score(eval_timesteps, eval_pred, average="weighted", zero_division=0)

    print(f"\nTraining Set Classification Metrics:")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  Precision (weighted): {train_precision:.4f}")
    print(f"  Recall (weighted): {train_recall:.4f}")
    print(f"  F1-Score (weighted): {train_f1:.4f}")

    print(f"\nEval Set Classification Metrics:")
    print(f"  Accuracy: {eval_accuracy:.4f}")
    print(f"  Precision (weighted): {eval_precision:.4f}")
    print(f"  Recall (weighted): {eval_recall:.4f}")
    print(f"  F1-Score (weighted): {eval_f1:.4f}")

    # Save results to text file
    results_file = Path(output_dir) / "results.txt"
    with open(results_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("TIMESTEP CLASSIFICATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            f"Train/Eval split: {train_split*100:.0f}% / {(1-train_split)*100:.0f}%\n"
        )
        f.write(f"Training samples: {len(train_emb)}\n")
        f.write(f"Eval samples: {len(eval_emb)}\n")
        f.write(f"Number of timestep classes: {len(np.unique(timesteps))}\n\n")

        f.write("TRAINING SET METRICS:\n")
        f.write(f"  Accuracy:  {train_accuracy:.4f}\n")
        f.write(f"  Precision: {train_precision:.4f}\n")
        f.write(f"  Recall:    {train_recall:.4f}\n")
        f.write(f"  F1-Score:  {train_f1:.4f}\n\n")

        f.write("EVAL SET METRICS:\n")
        f.write(f"  Accuracy:  {eval_accuracy:.4f}\n")
        f.write(f"  Precision: {eval_precision:.4f}\n")
        f.write(f"  Recall:    {eval_recall:.4f}\n")
        f.write(f"  F1-Score:  {eval_f1:.4f}\n")

    print(f"\nResults saved to {results_file}")

    return classifier, train_pred, eval_pred, train_timesteps, eval_timesteps


def visualize_classification_results(
    train_pred: np.ndarray,
    eval_pred: np.ndarray,
    train_timesteps: np.ndarray,
    eval_timesteps: np.ndarray,
    train_accuracy: float,
    eval_accuracy: float,
    output_dir: str,
):
    """Create visualizations of classification results using embedding space."""

    print("\nCreating classification visualizations...")

    # First, compute dimensionality reduction for visualization
    try:
        import umap

        print("  Running UMAP for visualization...")
        embeddings = np.concatenate([train_pred, eval_pred])  # Note: using reduced dims
        reducer = umap.UMAP(n_components=2, verbose=False)
        # For actual visualization, we'd need the original embeddings
        # For now, create plots without the embedding space projection
    except ImportError:
        print("  UMAP not available for visualization")

    # Create performance table
    num_timesteps = int(np.max(np.concatenate([train_timesteps, eval_timesteps]))) + 1

    print(f"\n" + "=" * 70)
    print("PER-TIMESTEP CLASSIFICATION ACCURACY")
    print("=" * 70)
    print(f"{'Timestep':<12} {'Train Accuracy':<20} {'Eval Accuracy':<20}")
    print("-" * 52)

    per_timestep_results = {}
    for t in range(num_timesteps):
        train_t_mask = train_timesteps == t
        eval_t_mask = eval_timesteps == t

        if train_t_mask.sum() > 0:
            train_t_acc = np.mean(train_pred[train_t_mask] == t)
        else:
            train_t_acc = 0

        if eval_t_mask.sum() > 0:
            eval_t_acc = np.mean(eval_pred[eval_t_mask] == t)
        else:
            eval_t_acc = 0

        print(f"T={t:<10} {train_t_acc*100:<19.1f}% {eval_t_acc*100:<19.1f}%")
        per_timestep_results[t] = {"train": train_t_acc, "eval": eval_t_acc}

    # Create confusion matrices if seaborn available
    if sns is not None:
        print("\nCreating confusion matrix visualizations...")
        train_cm = confusion_matrix(
            train_timesteps, train_pred, labels=np.arange(num_timesteps)
        )
        eval_cm = confusion_matrix(
            eval_timesteps, eval_pred, labels=np.arange(num_timesteps)
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Train confusion matrix
        sns.heatmap(
            train_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=np.arange(num_timesteps),
            yticklabels=np.arange(num_timesteps),
            cbar_kws={"label": "Count"},
        )
        axes[0].set_title(
            f"Training Set Confusion Matrix\n(True Timesteps vs Predicted Timesteps)",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xlabel("Predicted Timestep", fontsize=11)
        axes[0].set_ylabel("True Timestep", fontsize=11)

        # Eval confusion matrix
        sns.heatmap(
            eval_cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            ax=axes[1],
            xticklabels=np.arange(num_timesteps),
            yticklabels=np.arange(num_timesteps),
            cbar_kws={"label": "Count"},
        )
        axes[1].set_title(
            f"Eval Set Confusion Matrix\n(True Timesteps vs Predicted Timesteps)",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xlabel("Predicted Timestep", fontsize=11)
        axes[1].set_ylabel("True Timestep", fontsize=11)

        plt.tight_layout()
        confusion_path = Path(output_dir) / "confusion_matrices.png"
        plt.savefig(confusion_path, dpi=150, bbox_inches="tight")
        print(f"  Saved confusion matrices → {confusion_path}")
        plt.close()

        # Find most confused pairs
        print(f"\n" + "=" * 70)
        print("MOST CONFUSED TIMESTEP PAIRS (Eval Set)")
        print("=" * 70)

        eval_cm_copy = eval_cm.copy()
        np.fill_diagonal(eval_cm_copy, 0)

        confusions = []
        for i in range(num_timesteps):
            for j in range(num_timesteps):
                if i != j and eval_cm_copy[i, j] > 0:
                    confusions.append((eval_cm_copy[i, j], i, j))

        confusions.sort(reverse=True)

        print(f"\n{'Count':<8} {'True Timestep':<20} {'Predicted As':<20}")
        print("-" * 48)
        for count, true_t, pred_t in confusions[:10]:
            print(f"{count:<8} {true_t:<20} {pred_t:<20}")


if __name__ == "__main__":
    classifier, train_pred, eval_pred, train_timesteps, eval_timesteps = (
        train_timestep_classifier()
    )

    train_accuracy = np.mean(train_pred == train_timesteps)
    eval_accuracy = np.mean(eval_pred == eval_timesteps)

    visualize_classification_results(
        train_pred,
        eval_pred,
        train_timesteps,
        eval_timesteps,
        train_accuracy,
        eval_accuracy,
        "classifier",
    )
