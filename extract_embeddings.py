"""Extract embeddings from the model using a dataloader."""

import json
import os
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from tqdm import tqdm

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_img_preprocessor

# os.environ["STABLEWM_HOME"] = "le-wm/data"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_model(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Build model from config.json using local classes, then load weights.pt."""
    ckpt_dir = Path(checkpoint_dir)

    checkpoint_name = "weights.pt"
    # checkpoint_name = "lewm-rlbench_epoch_8_object.ckpt"

    if "object" in checkpoint_name:
        # .ckpt file
        model = torch.load(
            ckpt_dir / checkpoint_name, map_location=device, weights_only=False
        )
    elif checkpoint_name.endswith(".pt"):
        with open(ckpt_dir / "config.json") as f:
            cfg = json.load(f)

        enc_cfg = cfg["encoder"]
        pred_cfg = cfg["predictor"]
        act_cfg = cfg["action_encoder"]
        proj_cfg = cfg["projector"]
        pp_cfg = cfg["pred_proj"]

        encoder = spt.backbone.utils.vit_hf(
            enc_cfg["size"],
            patch_size=enc_cfg["patch_size"],
            image_size=enc_cfg["image_size"],
            pretrained=enc_cfg["pretrained"],
            use_mask_token=enc_cfg["use_mask_token"],
        )

        predictor = ARPredictor(
            num_frames=pred_cfg["num_frames"],
            input_dim=pred_cfg["input_dim"],
            hidden_dim=pred_cfg["hidden_dim"],
            output_dim=pred_cfg["output_dim"],
            depth=pred_cfg["depth"],
            heads=pred_cfg["heads"],
            mlp_dim=pred_cfg["mlp_dim"],
            dim_head=pred_cfg["dim_head"],
            dropout=pred_cfg["dropout"],
            emb_dropout=pred_cfg["emb_dropout"],
        )

        action_encoder = Embedder(
            input_dim=act_cfg["input_dim"],
            emb_dim=act_cfg["emb_dim"],
        )

        projector = MLP(
            input_dim=proj_cfg["input_dim"],
            hidden_dim=proj_cfg["hidden_dim"],
            output_dim=proj_cfg["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

        pred_proj = MLP(
            input_dim=pp_cfg["input_dim"],
            hidden_dim=pp_cfg["hidden_dim"],
            output_dim=pp_cfg["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

        model = JEPA(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
        )
        state_dict = torch.load(
            ckpt_dir / checkpoint_name, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def extract_embeddings(
    dataset_name: str = "ogbench/cube_single_expert",
    num_steps: int = 4,
    frameskip: int = 5,
    img_size: int = 224,
    num_episodes: int = 100,
    timesteps_per_episode: int = 10,
    checkpoint_dir: str = "checkpoints/OGB",
    device: str = "cuda",
    output_dir: str = "embeddings",
):
    """Extract embeddings from episodes using the model with batched processing."""

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset (like train.py)
    print(f"Loading dataset: {dataset_name}")
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        num_steps=num_steps,
        frameskip=frameskip,
    )
    transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=img_size
    )
    dataset.transform = spt.data.transforms.Compose(transform)

    # Load model
    model = load_model(checkpoint_dir, device)

    print(f"\nExtracting embeddings from {num_episodes} episodes...")
    print(f"  Timesteps per episode: {timesteps_per_episode}")
    print(f"  Processing on device: {device}")

    embeddings_list = []
    timesteps_list = []
    episodes_list = []

    for episode_idx in tqdm(range(num_episodes), desc="Processing episodes"):
        # Load full episode
        episode_data = dataset.load_episode(episode_idx)
        frames = episode_data["pixels"]  # (T_total, C, H, W)

        # Take first T timesteps
        frames = frames[:timesteps_per_episode]  # (T, C, H, W)

        # Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
        frames_batch = frames.unsqueeze(0)

        # Process in batches to avoid memory issues
        batch_size = 256
        if frames.shape[0] < batch_size:
            num_batches = 1
        else:
            num_batches = frames.shape[0] // batch_size + 1

        episode_embeddings = []
        # for batch_idx in range(num_batches):
        #     start_idx = batch_idx * batch_size
        #     end_idx = min((batch_idx + 1) * batch_size)

        #     # frames_batch is (T, 1, C, H, W) - reshape to (B, T, C, H, W)
        #     batch_frames = (
        #         frames_batch[start_idx:end_idx].permute(1, 0, 2, 3, 4).to(device)
        #     )  # (1, B, C, H, W)

        # Extract embeddings - model expects (B, T, C, H, W)
        with torch.no_grad():
            output = model.encode({"pixels": frames_batch.to(device)})

        emb = output["emb"]  # (B, T, D)
        emb = rearrange(emb, "b t ... -> (b t) ...")  # (B, D)
        episode_embeddings.append(emb.cpu().numpy())

        # Concatenate all batches for this episode
        episode_embeddings = np.concatenate(episode_embeddings, axis=0)  # (T, D)
        embeddings_list.append(episode_embeddings)
        # timesteps_list.append(np.arange(timesteps_per_episode))
        timesteps_list.append(episode_data["subgoal_state"].squeeze())
        episodes_list.append(np.full(frames.shape[0], episode_idx))

    # Concatenate all episodes - flatten to (N, D)
    embeddings = np.concatenate(embeddings_list)  # (N, D)
    timesteps = np.concatenate(timesteps_list, axis=0)
    episodes = np.concatenate(episodes_list, axis=0)

    dataset_dir = Path(output_dir) / Path(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Save to output directory
    emb_path = Path(output_dir) / Path(dataset_name) / "embeddings.npz"
    np.savez(
        emb_path,
        embeddings=embeddings,
        timesteps=timesteps,
        episodes=episodes,
    )

    print(f"\nExtraction complete!")
    print(f"  Total sequences processed: {num_episodes}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")
    print(f"  Episodes shape: {episodes.shape}")
    print(f"  Saved to: {emb_path}")

    return embeddings, timesteps, episodes


if __name__ == "__main__":
    # Extract embeddings from episodes
    extract_embeddings(num_episodes=1000, timesteps_per_episode=1000, frameskip=2)
