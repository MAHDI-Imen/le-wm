"""
Embedding analysis for LeWM.

Loads a trained JEPA checkpoint, encodes observations from a dataset,
reduces the latent space to 2D, and plots each embedding colored by
its timestep (step_idx within the episode).

Usage:
    python embedding_analysis.py --checkpoint checkpoints/OGB
    python embedding_analysis.py --checkpoint checkpoints/OGB --dataset ogbench/cube_single_expert --method tsne
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, Subset

from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from utils import get_img_preprocessor


def load_model(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Build model from config.json using local classes, then load weights.pt."""
    ckpt_dir = Path(checkpoint_dir)
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
        ckpt_dir / "weights.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def collect_embeddings(model, loader, step_idx_per_window, num_steps, device):
    """Encode all frames in each window.

    Returns:
        embs:      (N * num_steps, D) float32 numpy array
        step_idxs: (N * num_steps,)   int numpy array
    """
    all_embs = []
    window_count = 0

    for batch in loader:
        pixels = batch["pixels"].to(device).float()  # (B, T, C, H, W)
        b, t = pixels.shape[:2]

        with torch.no_grad():
            output = model.encode({"pixels": pixels})
        emb = output["emb"].cpu().numpy()  # (B, T, D)
        all_embs.append(emb.reshape(b * t, -1))

        window_count += b

    embs = np.concatenate(all_embs, axis=0)  # (N*T, D)

    # Build per-frame step indices:
    # for window i starting at step s, frame j has step s + j
    starts = step_idx_per_window  # (N,)
    offsets = np.arange(num_steps)  # (T,)
    # outer sum: (N, T) -> flatten to (N*T,)
    step_idxs = (starts[:, None] + offsets[None, :]).reshape(-1)

    return embs, step_idxs


def reduce_dimensions(embs, method, random_state):
    print(
        f"Running {method.upper()} on {len(embs)} embeddings (dim={embs.shape[1]})..."
    )
    if method == "umap":
        try:
            import umap  # noqa: PLC0415

            reducer = umap.UMAP(n_components=2, random_state=random_state, verbose=True)
        except ImportError:
            print(
                "umap-learn not installed — falling back to t-SNE. `pip install umap-learn`"
            )
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE  # noqa: PLC0415

        perplexity = min(30, len(embs) - 1)
        reducer = TSNE(
            n_components=2, random_state=random_state, perplexity=perplexity, verbose=1
        )

    reduced = reducer.fit_transform(embs)
    return reduced, method


def plot(reduced, step_idxs, method, output_path, dataset_name):
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=step_idxs,
        cmap="viridis",
        s=6,
        alpha=0.6,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Step index (timestep within episode)", fontsize=11)

    ax.set_title(f"LeWM latent space — {dataset_name} ({method.upper()})", fontsize=13)
    ax.set_xlabel(f"{method.upper()} dim 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} dim 2", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LeWM latent space colored by timestep"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory containing config.json and weights.pt",
    )
    parser.add_argument(
        "--dataset",
        default="ogbench/cube_single_expert",
        help="Dataset name (default: pusht_expert_train)",
    )
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of trajectory windows to encode",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Frames per window (history_size + num_preds, default: 4)",
    )
    parser.add_argument(
        "--frameskip", type=int, default=5, help="Dataset frameskip (default: 5)"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output", default="embedding_analysis.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── model ──────────────────────────────────────────────────────────────
    print(f"Loading model from {args.checkpoint} on {args.device}...")
    model = load_model(args.checkpoint, args.device)

    # ── dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = swm.data.HDF5Dataset(
        args.dataset,
        num_steps=args.num_steps,
        frameskip=args.frameskip,
        keys_to_load=["pixels"],
    )
    transform = get_img_preprocessor(
        source="pixels", target="pixels", img_size=args.img_size
    )
    dataset.transform = spt.data.transforms.Compose(transform)

    # subsample dataset indices
    n = min(args.n_samples, len(dataset))
    indices = np.sort(rng.choice(len(dataset), size=n, replace=False))

    # fetch starting step_idx for each sampled window
    step_idx_all = dataset.get_col_data("step_idx")
    step_idx_per_window = step_idx_all[indices].astype(int)

    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(args.device == "cuda"),
    )

    # ── encode ─────────────────────────────────────────────────────────────
    print(
        f"Encoding {n} windows × {args.num_steps} frames = {n * args.num_steps} embeddings..."
    )
    embs, step_idxs = collect_embeddings(
        model, loader, step_idx_per_window, args.num_steps, args.device
    )
    print(f"Embedding matrix: {embs.shape}")

    # ── dimensionality reduction ────────────────────────────────────────────
    reduced, method_used = reduce_dimensions(embs, args.method, args.seed)

    # ── plot ───────────────────────────────────────────────────────────────
    plot(reduced, step_idxs, method_used, args.output, args.dataset)


if __name__ == "__main__":
    main()
