"""
Dataset visualization for LeWM.

Loads episodes from a dataset and visualizes them as a grid where:
- Rows represent timesteps/steps within episodes
- Columns represent different episodes

Usage:
    python dataset_visualization.py --dataset ogbench/cube_single_expert --num_episodes 2 --num_steps 8
    python dataset_visualization.py --dataset ogbench/cube_single_expert --num_episodes 4 --frameskip 2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, Subset

from utils import get_img_preprocessor


def load_dataset(dataset_name, num_steps, frameskip, img_size):
    """Load and prepare the HDF5 dataset."""
    print(f"Loading dataset: {dataset_name}")
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=["pixels"],
    )
    transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=img_size
    )
    dataset.transform = spt.data.transforms.Compose(transform)
    return dataset


def get_raw_frames(dataset, window_idx, num_steps, frameskip):
    """Extract raw frames for a window without transforms for visualization.

    This reads the raw HDF5 data to get the actual pixel values.
    """
    # Get the item with transforms applied
    item = dataset[window_idx]
    pixels = item["pixels"]  # (T, C, H, W) - already transformed
    return pixels


def denormalize_frame(frame):
    """Denormalize from standard ImageNet normalization."""
    # ImageNet normalization constants
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Denormalize: x_orig = (x_norm * std) + mean
    denorm = (frame * std) + mean
    return np.clip(denorm, 0, 1)


def visualize_episodes(dataset, episode_indices, num_steps, output_path, frameskip):
    """Create a grid visualization of episodes where rows are timesteps and columns are episodes."""
    num_episodes = len(episode_indices)

    # Collect frames for all episodes
    all_frames = []
    for idx in episode_indices:
        frames = get_raw_frames(dataset, idx, num_steps, frameskip)
        # frames shape: (T, C, H, W)
        all_frames.append(frames)

    # Convert to numpy if needed
    if isinstance(all_frames[0], torch.Tensor):
        all_frames = [f.cpu().numpy() for f in all_frames]

    # Get frame dimensions
    num_frames = all_frames[0].shape[0]
    c, h, w = all_frames[0].shape[1], all_frames[0].shape[2], all_frames[0].shape[3]

    # Create grid of subplots (rows=timesteps, cols=episodes)
    fig, axes = plt.subplots(num_frames, num_episodes, figsize=(4*num_episodes, 3*num_frames))

    # Handle case where there's only 1 row or 1 column
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    if num_episodes == 1:
        axes = axes.reshape(-1, 1)

    for step_idx in range(num_frames):
        for ep_idx in range(num_episodes):
            ax = axes[step_idx, ep_idx]
            frame = all_frames[ep_idx][step_idx]

            # Convert from (C, H, W) to (H, W, C) for display
            if frame.shape[0] == 3:  # RGB
                frame = np.transpose(frame, (1, 2, 0))
                # Denormalize from standard ImageNet normalization
                frame = denormalize_frame(np.transpose(frame, (2, 0, 1)))
                frame = np.transpose(frame, (1, 2, 0))
            elif frame.shape[0] == 1:  # Grayscale
                frame = frame.squeeze()
                frame = np.clip(frame, 0, 1)
            else:
                frame = np.transpose(frame, (1, 2, 0))
                frame = np.clip(frame, 0, 1)

            ax.imshow(frame, cmap='gray' if frame.ndim == 2 else None)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add step index on left side
            if ep_idx == 0:
                ax.set_ylabel(f'Step {step_idx}', fontsize=10, fontweight='bold')

            # Add episode index on top
            if step_idx == 0:
                ax.set_title(f'Episode {ep_idx}', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization → {output_path}")
    plt.close()


def print_dataset_info(dataset, num_steps, frameskip):
    """Print basic dataset statistics."""
    print(f"\nDataset Info:")
    print(f"  Total windows: {len(dataset)}")
    print(f"  Steps per window: {num_steps}")
    print(f"  Frameskip: {frameskip}")

    # Sample a batch to see shapes
    sample = dataset[0]
    pixels = sample["pixels"]
    print(f"  Frame shape: {pixels.shape}")
    print(f"  Frame dtype: {pixels.dtype}")
    print(f"  Frame value range: [{pixels.min():.3f}, {pixels.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataset episodes as a grid (rows=steps, cols=episodes)"
    )
    parser.add_argument(
        "--dataset",
        default="ogbench/cube_single_expert",
        help="Dataset name",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=8,
        help="Frames per window",
    )
    parser.add_argument(
        "--frameskip",
        type=int,
        default=5,
        help="Dataset frameskip",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size",
    )
    parser.add_argument(
        "--output",
        default="dataset_rollout.png",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode selection",
    )
    args = parser.parse_args()

    # Set random seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    dataset = load_dataset(
        args.dataset,
        args.num_steps,
        args.frameskip,
        args.img_size,
    )

    print_dataset_info(dataset, args.num_steps, args.frameskip)

    # Select random episodes
    n = min(args.num_episodes, len(dataset))
    episode_indices = np.sort(rng.choice(len(dataset), size=n, replace=False))
    print(f"\nVisualizing {n} episodes: {episode_indices}")

    # Create visualization
    visualize_episodes(
        dataset,
        episode_indices.tolist(),
        args.num_steps,
        args.output,
        args.frameskip,
    )
    print("Done!")


if __name__ == "__main__":
    main()
