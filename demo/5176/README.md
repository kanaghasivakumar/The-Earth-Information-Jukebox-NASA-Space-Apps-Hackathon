# 5176 SST Anomaly, near real time: the "today" track

`python -m jukebox.datasets.sst_anomaly_5176`. The movie has 901 daily frames, 26 May 2024 to 12 Nov 2026, regenerated daily. Frame dates come from the SVS frames-folder listing. The colorbar runs −4..+4 °C. The date box is masked; land and ice fail the colour match. 50% of pixels are valid.

The lead is the global mean anomaly (+0.35..+0.87 °C). There are six latitude-band voices. Controls: geophony = global anomaly; brightness = share of the ocean above +1 °C.
Audio round-trip: R² 0.978 (r 0.990, MAE 0.013 °C); shuffled control R² −1.13. There is no independent ground truth (MUR files need an Earthdata login).
Rerun to refresh: delete `data/raw/5176/` first, because the movie and the date list are cached.
