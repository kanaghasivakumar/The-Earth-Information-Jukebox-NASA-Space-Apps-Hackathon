# NASA visualization convolutional VAE

Standalone research pipeline. Keep NASA animations in `data/raw/`; caches and outputs stay in this directory. The loader supports MP4, MOV, WebM, GIF, and ordered still-image sequences. It resizes RGB to 64×64, samples at the requested rate, and computes Farneback flow between consecutive sampled frames. Each item is a `(5,64,64)` tensor (RGB followed by normalized flow), with source ID and timestamp. The first frame has no predecessor and is omitted.

The 5097 NASA animation is credited to NASA/Goddard Space Flight Center Scientific Visualization Studio; the supplied hurricane visualization is [Hurricane Ian (2022)](https://svs.gsfc.nasa.gov/5097/).

## Experimental profile for the next run

The defaults now define a fresh experimental profile: latent dimension 2, 50 epochs, batch size 8, target sampling rate 5 FPS, flow scale 1, flow loss weight 0.3, and a 10-epoch linear KL warmup to beta 0.01. Training beta is logged each epoch. Validation and best-checkpoint selection always use the fixed target beta 0.01 so epoch comparisons remain consistent.

These settings are motivated by the prior 50-epoch result: validation RGB MSE was about 0.01292, flow MSE about 0.000434 (roughly a 30:1 gap), and latent standard deviations were near zero, with zero temporal distance and blurred RGB / nearly black flow reconstructions. Lowering the flow scale may brighten flow targets, while weight 0.3 is an empirical balancing start based on the old loss ratio. Clipping changes that ratio, so the new profile is an experiment and remains unverified until a future run. KL warmup is intended to reduce early posterior collapse; it is not a guarantee.

## Splits and preprocessing

Default `--split auto` uses whole-source holdout with multiple animations, or a clearly exploratory contiguous time-block split with a two-sample gap for a single source. One-video validation is not an independent-video generalization estimate. Checkpoints save exact split indices and source signatures. Evaluation uses those indices and reports reconstruction/KL metrics only for validation samples; all-frame latent plots and arrays are clearly labeled exploratory.

Static masking accepts a supplied mask image through `--overlay-mask`; pixels outside it are zeroed in RGB and flow. Translation-only ECC registration is enabled with `--align`. Both options are recorded in checkpoints.

## Install and start a fresh run

From `eic-vae/`:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python train.py --data data/raw --split auto --epochs 50 --batch-size 8 --lr 0.001 --latent-dim 2 --beta 0.01 --kl-warmup-epochs 10 --flow-weight 0.3 --fps 5 --sequence-fps 30 --flow-scale 1
python evaluate.py --data data/raw --checkpoint checkpoints/best_vae.pt --batch-size 8
```

For a one-source dataset, `auto` selects the exploratory time-block split. Device `auto` selects CUDA, Apple MPS, or CPU. The evaluation batch size can also be lowered if needed.

## Resume an existing checkpoint

`--resume` fine-tunes the selected best checkpoint with a fresh Adam optimizer. It requires the same data, preprocessing, model dimensions, and exact split. If any check differs, training stops and requires a fresh experiment. `--epochs` is the number of additional fine-tuning epochs, and the new run writes separate `metrics_finetune.csv` and `loss_curves_finetune.png` files. The current checkpoint used flow scale 10 and flow weight 1; continue it with those settings, for example:

```sh
python train.py --data data/raw --resume checkpoints/best_vae.pt --epochs 50 --batch-size 8 --lr 0.001 --latent-dim 2 --beta 0.01 --kl-warmup-epochs 10 --flow-weight 1 --fps 5 --sequence-fps 30 --flow-scale 10 --split auto
```

Resume starts from the selected best model weights and records new epochs after the existing metrics log; Adam state is reinitialized because optimizer state is not stored. For legacy checkpoints without source signatures, the loader warns and checks source paths and sample count. Evaluation accepts both `model_config.image_size` and legacy `model_config.size`.

Training writes `checkpoints/best_vae.pt`, `results/metrics.csv`, and `results/loss_curves.png`. Evaluation writes validation metrics and examples to `results/evaluation/`, and all-frame latent artifacts under `results/exploratory_all_frames/`. Optical flow, reconstruction errors, and latent trajectories are exploratory diagnostics; this VAE is not a physical hurricane forecasting model.

Run checks with `pytest -q`.
