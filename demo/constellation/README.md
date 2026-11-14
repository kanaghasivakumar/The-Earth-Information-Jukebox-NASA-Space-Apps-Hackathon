# Constellation (navigation view)

`python scripts/constellation.py` → every frame of 3912, 5046, 5097 (400 of 533 frames), 5313 embedded with
DINOv2-small; t-SNE 2D (UMAP is not installed in the env) + PCA 3D. Each point: dataset, frame, t, section label,
region, `xy`, `xyz_pca`. Datasets form separate islands (different visual styles); within an island the path is the
movie's trajectory. 5313 (night lights "wZooms") splits into two clusters: that split is the inset zoom panels
appearing at ~22 s, i.e. presentation, not data.
