"""Load a single batch from the dataset and run a JEPA forward pass."""

import json
from pathlib import Path

import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from utils import get_img_preprocessor, get_column_normalizer
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def load_model(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Build model from config.json using local classes, then load weights.pt.

    The config uses stable_worldmodel._target_ references that are not
    importable from the installed package, so we instantiate local equivalents.
    """
    ckpt_dir = Path(checkpoint_dir)
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    enc_cfg  = cfg["encoder"]
    pred_cfg = cfg["predictor"]
    act_cfg  = cfg["action_encoder"]
    proj_cfg = cfg["projector"]
    pp_cfg   = cfg["pred_proj"]

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

    state_dict = torch.load(ckpt_dir / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


CHECKPOINT = "checkpoints/OGB"
DATASET    = "ogbench/cube_single_expert"
DATA_DIR   = "data"           # local cache dir containing the .h5 files
NUM_STEPS  = 4   # history_size (3) + num_preds (1)
FRAMESKIP  = 5
IMG_SIZE   = 224
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── dataset ────────────────────────────────────────────────────────────────
dataset = swm.data.HDF5Dataset(
    DATASET,
    num_steps=NUM_STEPS,
    frameskip=FRAMESKIP,
    keys_to_load=["pixels", "action", "observation"],
    keys_to_cache=["action", "observation"],
    cache_dir=DATA_DIR,
)

transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=IMG_SIZE)]
for col in ["action", "observation"]:
    transforms.append(get_column_normalizer(dataset, col, col))
dataset.transform = spt.data.transforms.Compose(*transforms)

loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(loader))

print("── batch keys & shapes ──────────────────────────────")
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"  {k:12s}: {tuple(v.shape)}  dtype={v.dtype}")

# ── model ──────────────────────────────────────────────────────────────────
model = load_model(CHECKPOINT, DEVICE)

batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
batch["action"] = torch.nan_to_num(batch["action"], 0.0)

# ── forward pass ───────────────────────────────────────────────────────────
with torch.no_grad():
    output = model.encode(batch)

    ctx_len  = 3  # history_size
    n_preds  = 1  # num_preds
    emb      = output["emb"]       # (B, T, D)
    act_emb  = output["act_emb"]   # (B, T, D)

    ctx_emb  = emb[:, :ctx_len]
    ctx_act  = act_emb[:, :ctx_len]
    pred_emb = model.predict(ctx_emb, ctx_act)  # (B, n_preds, D)
    tgt_emb  = emb[:, n_preds:]                 # (B, T - n_preds, D)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()

print("\n── output shapes ────────────────────────────────────")
print(f"  emb:       {tuple(emb.shape)}")
print(f"  act_emb:   {tuple(act_emb.shape)}")
print(f"  ctx_emb:   {tuple(ctx_emb.shape)}")
print(f"  pred_emb:  {tuple(pred_emb.shape)}")
print(f"  tgt_emb:   {tuple(tgt_emb.shape)}")
print(f"\n  pred_loss: {pred_loss.item():.6f}")
