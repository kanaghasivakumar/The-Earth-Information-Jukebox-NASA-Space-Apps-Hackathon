"""Memory-bounded evaluation of a saved NASA visualization VAE."""

from pathlib import Path
import argparse, csv, json, warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from dataset import NASAVisualizationDataset
from models.vae import ConvVAE
from train import device_auto, BASE
from visualize import latent_plots, posterior_surface


def checkpoint_image_size(checkpoint):
    config = checkpoint.get("model_config", {})
    size = config.get("image_size", config.get("size"))
    if size is None:
        raise ValueError(
            "Checkpoint lacks model_config.image_size (or legacy model_config.size)."
        )
    return int(size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE / "data/raw"))
    parser.add_argument("--checkpoint", default=str(BASE / "checkpoints/best_vae.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--posterior-frames", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = device_auto() if args.device == "auto" else torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    preprocessing = checkpoint["preprocessing"]
    dataset = NASAVisualizationDataset(
        args.data, **preprocessing, cache_dir=BASE / "data/cache"
    )
    if dataset.sources != checkpoint.get("sources"):
        raise ValueError("Dataset source paths/order differ from checkpoint.")
    if len(dataset) != checkpoint.get("dataset_length"):
        raise ValueError(
            "Dataset sample count differs from checkpoint; check preprocessing and files."
        )
    saved_signatures = checkpoint.get("source_signatures")
    signatures = [dataset._cache_key(Path(source)).stem for source in dataset.sources]
    if saved_signatures is None:
        warnings.warn(
            "Legacy checkpoint has no source_signatures; source paths and sample count were checked only.",
            RuntimeWarning,
        )
    elif saved_signatures != signatures:
        raise ValueError(
            "Source content/preprocessing signature differs from checkpoint."
        )

    validation_indices = checkpoint.get("validation_indices", [])
    train_indices = checkpoint.get("train_indices", [])
    if not validation_indices or max(validation_indices + train_indices) >= len(
        dataset
    ):
        raise ValueError(
            "Checkpoint validation/train indices are missing or out of range."
        )
    model = ConvVAE(
        checkpoint["model_config"]["latent_dim"],
        image_size=checkpoint_image_size(checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    validation_set = set(validation_indices)
    keep_example_count = min(8, len(validation_set))
    example_indices = set(sorted(validation_set)[:keep_example_count])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    latent_mu, latent_logvar, timestamps, source_ids = [], [], [], []
    flow_magnitudes, rgb_mse, flow_mse, kl_values = [], [], [], []
    examples = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input"].to(device)
            image = batch["image"].to(device)
            flow = batch["flow"].to(device)
            output = model(inputs, deterministic=True)
            batch_size = len(inputs)
            latent_mu.append(output["mu"].cpu().numpy())
            latent_logvar.append(output["logvar"].cpu().numpy())
            timestamps.extend(batch["timestamp"].numpy())
            source_ids.extend(batch["source_id"].numpy())
            flow_magnitudes.extend(
                torch.linalg.vector_norm(flow, dim=1).mean((1, 2)).cpu().numpy()
            )
            rgb_err = (output["rgb"] - image).pow(2).mean((1, 2, 3))
            flow_err = (output["flow"] - flow).pow(2).mean((1, 2, 3))
            kl = -0.5 * (
                1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp()
            ).sum(1)
            rgb_mse.extend(rgb_err.cpu().numpy())
            flow_mse.extend(flow_err.cpu().numpy())
            kl_values.extend(kl.cpu().numpy())

            for local_i in range(batch_size):
                global_i = offset + local_i
                if global_i in example_indices:
                    examples.append(
                        {
                            "index": global_i,
                            "image": image[local_i].cpu().numpy(),
                            "flow": flow[local_i].cpu().numpy(),
                            "rgb_reconstruction": output["rgb"][local_i].cpu().numpy(),
                            "flow_reconstruction": output["flow"][local_i]
                            .cpu()
                            .numpy(),
                        }
                    )
            offset += batch_size

    mu = np.concatenate(latent_mu)
    logvar = np.concatenate(latent_logvar)
    timestamps = np.asarray(timestamps)
    source_ids = np.asarray(source_ids)
    flow_magnitudes = np.asarray(flow_magnitudes)
    rgb_mse, flow_mse, kl_values = map(np.asarray, (rgb_mse, flow_mse, kl_values))
    validation = np.asarray(validation_indices, dtype=int)

    evaluation_dir = BASE / "results/evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir = BASE / "results/exploratory_all_frames"
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    active_dimensions = latent_plots(mu, timestamps, source_ids, exploratory_dir)
    np.savez_compressed(
        exploratory_dir / "latent.npz",
        mu=mu,
        logvar=logvar,
        timestamps=timestamps,
        source_ids=source_ids,
        source_paths=np.asarray(dataset.sources, dtype=str),
    )
    with (exploratory_dir / "latent.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp", "source_id", "source_path"]
            + [f"mu_{i}" for i in range(mu.shape[1])]
            + [f"logvar_{i}" for i in range(mu.shape[1])]
        )
        writer.writerows(
            [
                [
                    timestamps[i],
                    int(source_ids[i]),
                    dataset.sources[source_ids[i]],
                    *mu[i],
                    *logvar[i],
                ]
                for i in range(len(timestamps))
            ]
        )

    distances, velocities, velocity_flow, accelerations = [], [], [], []
    for source_id in np.unique(source_ids):
        indices = np.flatnonzero(source_ids == source_id)
        indices = indices[np.argsort(timestamps[indices])]
        if len(indices) < 2:
            continue
        dt = np.diff(timestamps[indices])
        steps = np.diff(mu[indices], axis=0)
        distance = np.linalg.norm(steps, axis=1)
        velocity = distance / np.maximum(dt, 1e-8)
        distances.extend(distance)
        velocities.extend(velocity)
        velocity_flow.extend(zip(velocity, flow_magnitudes[indices[1:]]))
        if len(indices) >= 3:
            accelerations.extend(
                np.linalg.norm(np.diff(mu[indices], n=2, axis=0), axis=1)
            )
    correlation = None
    if len(velocity_flow) > 1:
        velocity_values, flow_values = np.asarray(velocity_flow).T
        if np.std(velocity_values) and np.std(flow_values):
            correlation = float(np.corrcoef(velocity_values, flow_values)[0, 1])
    metrics = {
        "validation_rgb_mse": float(rgb_mse[validation].mean()),
        "validation_flow_mse": float(flow_mse[validation].mean()),
        "validation_mean_kl": float(kl_values[validation].mean()),
        "all_frame_latent_mean": mu.mean(0).tolist(),
        "all_frame_latent_std": mu.std(0).tolist(),
        "mean_temporal_distance": float(np.mean(distances)) if distances else None,
        "mean_temporal_velocity": float(np.mean(velocities)) if velocities else None,
        "latent_velocity_flow_magnitude_correlation": correlation,
        "mean_second_difference_acceleration": float(np.mean(accelerations))
        if accelerations
        else None,
        "split": checkpoint.get("split", {"mode": "legacy"}),
        "best_epoch": checkpoint.get("best_epoch"),
        "validation_samples": len(validation),
    }
    (evaluation_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if examples:
        fig, axes = plt.subplots(
            len(examples), 4, figsize=(11, 2.5 * len(examples)), squeeze=False
        )
        for row, sample in enumerate(examples):
            axes[row, 0].imshow(np.transpose(sample["image"], (1, 2, 0)))
            axes[row, 0].set_title("RGB input")
            axes[row, 1].imshow(np.transpose(sample["rgb_reconstruction"], (1, 2, 0)))
            axes[row, 1].set_title("RGB reconstruction")
            for column, field, title in (
                (2, sample["flow"], "Flow"),
                (3, sample["flow_reconstruction"], "Flow reconstruction"),
            ):
                u, v = field
                magnitude = np.sqrt(u * u + v * v)
                hsv = np.stack(
                    (
                        (np.arctan2(v, u) + np.pi) / (2 * np.pi),
                        np.ones_like(magnitude),
                        np.clip(magnitude, 0, 1),
                    ),
                    axis=-1,
                )
                axes[row, column].imshow(hsv_to_rgb(hsv))
                axes[row, column].set_title(f"{title} HSV")
            for axis in axes[row]:
                axis.axis("off")
        fig.tight_layout()
        fig.savefig(evaluation_dir / "reconstructions.png", dpi=130)
        plt.close(fig)

    chosen = np.linspace(0, len(mu) - 1, min(args.posterior_frames, len(mu)), dtype=int)
    for index in chosen:
        posterior_surface(
            mu[index],
            logvar[index],
            exploratory_dir / f"posterior_{index:04d}.png",
            f"Frame {index} posterior",
            latent_dim=mu.shape[1],
            dimensions=active_dimensions,
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
