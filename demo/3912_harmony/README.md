# 3912 with model-written harmony

This demo takes the M1 data melody from `demo/3912` (global mean SST) and keeps it note for note.
The Anticipatory Music Transformer adds strings harmony and a bass line around it, and FluidSynth
renders the result with the FluidR3_GM SoundFont (MIT).

```
PYTHONPATH=. HF_HOME=/projects/e32706/omb8654/hf_cache \
  srun -A p32576 -p short -t 00:30:00 --mem=16G -c 4 \
  /projects/e32706/omb8654/conda/envs/jukebox/bin/python scripts/harmonize_demo.py
```

The run used `stanford-crfm/music-medium-800k` with seed 0 and top_p 0.95. The lead was presented to the model as flute (GM 73).

| File | What it is |
|---|---|
| `mix.mp3` | lead (flute) + harmony (string ensemble, GM 48) + bass (acoustic bass, GM 32) |
| `lead.mp3`, `harmony.mp3`, `bass.mp3` | the three stems, at the same gain as the mix |
| `score.mid` | MIDI of all voices. Model tracks are named `harmony:<program>` and `bass:<program>` |
| `timeline.json` | every note. Lead entries are identical (as parsed JSON) to `demo/3912/timeline.json`. Model notes have `value: null` and `source.program/model/seed` |
| `report.json` | note counts, clash report, timing, lead pitch recovery |
| `pianoroll.png` | orange = data lead (fixed), blue = harmony, green = bass |

Results for this render:
- 36 lead notes unchanged. The model added 101 harmony notes and 73 bass notes.
- 0 of 243 overlapping lead/model note pairs form a m2, tritone or M7, and no harmony note rises above the lead. Both hold by construction, because the sampler masks them.
- The data melody is still recoverable. pyin on the full mix matches the lead pitch within ½ semitone in 99.2% of 0.5 s frames, and in 100% on the lead stem alone. With a vibraphone lead (GM 11), the strings swamp it and the full-mix figure drops to 4–8%. That is why the lead is rendered as flute.
- Infill took 71 s for 66.5 s of music on 4 CPU cores (medium model, about 64 s per 60 s of audio). FluidSynth rendering took about 4 s.

The band pad voices from M1 are left out by default. Pass `--keep-bands` to add them quietly.
