from pathlib import Path
import argparse, csv, random, warnings
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from dataset import (
    NASAVisualizationDataset,
    source_video_split,
    exploratory_time_block_split,
)
from models.vae import ConvVAE, vae_loss

BASE = Path(__file__).resolve().parent
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8
DEFAULT_FPS = 5.0
DEFAULT_FLOW_SCALE = 1.0
DEFAULT_FLOW_WEIGHT = 0.3
DEFAULT_BETA = 0.01
DEFAULT_KL_WARMUP_EPOCHS = 10


def warmup_beta(target_beta, warmup_epochs, epoch):
    """Linearly ramp KL weight from zero to target over the first N epochs."""
    if warmup_epochs <= 0:
        return float(target_beta)
    return float(target_beta) * min(max(float(epoch) / warmup_epochs, 0.0), 1.0)


def device_auto():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )


def split_indices(ds, mode, fraction, seed, gap):
    if mode == "auto":
        mode = "source" if len(ds.sources) >= 2 else "exploratory"
    return (
        source_video_split(ds, fraction, seed)
        if mode == "source"
        else exploratory_time_block_split(ds, val_fraction=fraction, gap=gap),
        mode,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(BASE / "data/raw"))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--resume", nargs="?", const=str(BASE / "checkpoints/best_vae.pt"), default=None, metavar="CHECKPOINT", help="Fine-tune from a saved best checkpoint (defaults to checkpoints/best_vae.pt)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--kl-warmup-epochs", type=int, default=DEFAULT_KL_WARMUP_EPOCHS, help="Linearly ramp training beta to --beta; 0 disables warmup")
    p.add_argument("--flow-weight", type=float, default=DEFAULT_FLOW_WEIGHT)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--sequence-fps", type=float, default=30)
    p.add_argument("--flow-scale", type=float, default=DEFAULT_FLOW_SCALE)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--split", choices=["auto", "source", "exploratory"], default="auto")
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--overlay-mask")
    p.add_argument("--align", action="store_true")
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    ds = NASAVisualizationDataset(
        a.data,
        size=a.size,
        fps=a.fps,
        sequence_fps=a.sequence_fps,
        flow_scale=a.flow_scale,
        cache_dir=BASE / "data/cache",
        overlay_mask=a.overlay_mask,
        align=a.align,
    )
    (tr, va), split = split_indices(ds, a.split, a.val_fraction, a.seed, a.gap)
    train = DataLoader(Subset(ds, tr), batch_size=a.batch_size, shuffle=True)
    val = DataLoader(Subset(ds, va), batch_size=a.batch_size)
    dev = device_auto() if a.device == "auto" else torch.device(a.device)
    model = ConvVAE(a.latent_dim, image_size=a.size).to(dev)
    resume_ck = None
    if a.resume:
        resume_ck = torch.load(a.resume, map_location=dev, weights_only=True)
        model_cfg = resume_ck.get("model_config", {})
        old_size = model_cfg.get("image_size", model_cfg.get("size"))
        if model_cfg.get("latent_dim") != a.latent_dim or old_size != a.size:
            raise ValueError("Resume checkpoint model dimensions do not match --latent-dim/--size.")
        if resume_ck.get("sources") != ds.sources or resume_ck.get("dataset_length") != len(ds):
            raise ValueError("Resume checkpoint sources or sample count differ from the current dataset.")
        saved_signatures = resume_ck.get("source_signatures")
        current_signatures = [ds._cache_key(Path(source)).stem for source in ds.sources]
        if saved_signatures is None:
            warnings.warn("Resume checkpoint has no source signatures; source paths and sample count were checked only.", RuntimeWarning)
        elif saved_signatures != current_signatures:
            raise ValueError("Resume checkpoint source content/preprocessing signatures differ.")
        expected_preprocessing = {"size": a.size, "fps": a.fps, "sequence_fps": a.sequence_fps,
                                  "flow_scale": a.flow_scale, "overlay_mask": a.overlay_mask, "align": a.align}
        if resume_ck.get("preprocessing") != expected_preprocessing:
            raise ValueError("Resume checkpoint preprocessing differs from the current arguments.")
        old_config = resume_ck.get("config", {})
        for key, current in (("beta", a.beta), ("flow_weight", a.flow_weight)):
            if key in old_config and float(old_config[key]) != current:
                raise ValueError(f"Resume checkpoint {key} differs from the current arguments.")
        if "kl_warmup_epochs" not in old_config:
            warnings.warn("Checkpoint has no kl_warmup_epochs metadata; fine-tuning uses the requested warmup.", RuntimeWarning)
        elif int(old_config["kl_warmup_epochs"]) != a.kl_warmup_epochs:
            raise ValueError("Resume checkpoint kl_warmup_epochs differs from the current arguments.")
        if resume_ck.get("train_indices") != tr or resume_ck.get("validation_indices") != va:
            raise ValueError("Resume checkpoint split indices differ; preserve split mode, fraction, gap, and seed.")
        model.load_state_dict(resume_ck["model"])
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ckdir = BASE / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)
    out = BASE / "results"
    out.mkdir(parents=True, exist_ok=True)
    history = []
    history_path = out / ("metrics_finetune.csv" if resume_ck is not None else "metrics.csv")
    curve_path = out / ("loss_curves_finetune.png" if resume_ck is not None else "loss_curves.png")
    if a.epochs < 1:
        p.error("--epochs must be at least 1")

    def epoch(loader, training, beta):
        model.train(training)
        sums = np.zeros(4)
        count = 0
        batches = (
            tqdm(loader, desc=f"Epoch {current_epoch}/{a.epochs} train", leave=False)
            if training
            else loader
        )
        for b in batches:
            inp = b["input"].to(dev)
            image = b["image"].to(dev)
            flow = b["flow"].to(dev)
            n = len(inp)
            if training:
                opt.zero_grad()
            with torch.set_grad_enabled(training):
                o = model(inp)
                loss, parts = vae_loss(o, image, flow, beta, a.flow_weight)
                if training:
                    loss.backward()
                    opt.step()
            sums += (
                np.array(
                    [parts[k].item() for k in ("loss", "rgb_mse", "flow_mse", "kl")]
                )
                * n
            )
            count += n
            if training:
                avg = sums / count
                batches.set_postfix(
                    loss=f"{avg[0]:.4f}",
                    rgb=f"{avg[1]:.4f}",
                    flow=f"{avg[2]:.4f}",
                    kl=f"{avg[3]:.4f}",
                )
        return (sums / count).tolist()

    # Compare fine-tuned weights against the loaded checkpoint on the fixed-beta
    # validation objective. The checkpoint remains untouched unless this improves.
    best = epoch(val, False, a.beta)[0] if resume_ck is not None else float("inf")

    for ep in range(1, a.epochs + 1):
        current_epoch = ep
        training_beta = warmup_beta(a.beta, a.kl_warmup_epochs, ep)
        trm = epoch(train, True, training_beta)
        vam = epoch(val, False, a.beta)
        row = {
            "epoch": ep,
            "train_beta": training_beta,
            **{
                f"train_{k}": v
                for k, v in zip(("loss", "rgb_mse", "flow_mse", "kl"), trm)
            },
            **{
                f"val_{k}": v
                for k, v in zip(("loss", "rgb_mse", "flow_mse", "kl"), vam)
            },
        }
        history.append(row)
        print(row)
        if vam[0] < best:
            best = vam[0]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": {"latent_dim": a.latent_dim, "image_size": a.size},
                    "preprocessing": {
                        "size": a.size,
                        "fps": a.fps,
                        "sequence_fps": a.sequence_fps,
                        "flow_scale": a.flow_scale,
                        "overlay_mask": a.overlay_mask,
                        "align": a.align,
                    },
                    "split": {
                        "mode": split,
                        "requested_mode": a.split,
                        "val_fraction": a.val_fraction,
                        "gap": a.gap,
                        "seed": a.seed,
                    },
                    "train_indices": tr,
                    "validation_indices": va,
                    "dataset_length": len(ds),
                    "sources": ds.sources,
                    "source_signatures": [
                        ds._cache_key(Path(source)).stem for source in ds.sources
                    ],
                    "best_epoch": ep,
                    "best_val_loss": best,
                    **({
                        "fine_tune_from": str(a.resume),
                        "parent_best_epoch": resume_ck.get("best_epoch"),
                    } if resume_ck is not None else {}),
                    "config": vars(a),
                },
                ckdir / "best_vae.pt",
            )
    with history_path.open("w", newline="") as f:
        fieldnames = ["epoch", "train_beta", "train_loss", "train_rgb_mse", "train_flow_mse", "train_kl",
                      "val_loss", "val_rgb_mse", "val_flow_mse", "val_kl"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)
    if history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            [r["epoch"] for r in history], [r["train_loss"] for r in history], label="train"
        )
        ax.plot(
            [r["epoch"] for r in history],
            [r["val_loss"] for r in history],
            label="validation",
        )
        ax.set(xlabel="epoch", ylabel="weighted loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(curve_path)
        plt.close(fig)


if __name__ == "__main__":
    main()
