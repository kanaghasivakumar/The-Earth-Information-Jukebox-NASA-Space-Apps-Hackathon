"""Copy the mood pilot results into demo/<id>_mood/ (small MP3s + README) and write demo/learned/MOOD.md.

    source scripts/mood/env.sh && python scripts/mood/export_demo.py 5313 3912 5209

For each track it uses the best available render: tag "base" (GPU) if present, else "small" (CPU).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
DEMO = ROOT / "demo"
TITLES = {"5209": "Zonal temperature anomalies 1880–2023", "5313": "Night lights 2012–2023",
          "3912": "Sea surface currents and temperature"}
TAGS = ("base", "base-cpu", "small")


def mp3(src: Path, dst: Path, kbps: int):
    subprocess.run([shutil.which("ffmpeg") or str(Path(sys.executable).parent / "ffmpeg"), "-v", "error", "-y",
                    "-i", str(src), "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", str(dst)], check=True)


def f(x, nd=3):
    return "–" if x is None else (f"{x:+.{nd}f}" if x < 0 else f"{x:.{nd}f}")


def load(tid: str, tag: str):
    d = OUT / f"{tid}_mood"
    rp, gp = d / f"mood_report_{tag}.json", d / f"generate_{tag}.json"
    if not (rp.exists() and gp.exists()):
        return None
    return {"rep": json.loads(rp.read_text()), "gen": json.loads(gp.read_text()),
            "w": json.loads((d / "weights.json").read_text()), "dir": d, "tag": tag}


HEADLINE = {"5313": "city", "3912": "tropic", "5209": "warm"}   # the anchor the story predicts should move


def headline_corr(w: dict, tid: str) -> tuple:
    """The story's anchor: vision-only r with its data curve, detrended r, blended r."""
    n = HEADLINE.get(tid) or next(iter(w["linked_correlations"]))
    d = w["linked_correlations"][n]
    return (f"{n}↔{d['curve']}", d.get("r_image_only"), d.get("r_image_only_detrended"), d.get("r_blended"))


def readme(tid: str, R: dict, others: list[dict]) -> str:
    rep, gen, w = R["rep"], R["gen"], R["w"]
    fid = rep["fidelity"]
    kc = rep["key_check"]
    lines = [f"# {tid} mood pilot: {TITLES[tid]}", "",
             "A learned **mood texture** under the unchanged integrated arrangement: video frames → SigLIP "
             "style weights over a fixed prompt vocabulary (blended with the track's data curves) → Magenta "
             f"RealTime 2 (`{gen['size']}`, {gen['device'].upper()}) generating continuously over the timeline. "
             "The mood never sets a pitch. The data melody and all other stems are exactly the "
             "`integrate` stems.", "",
             "| File | What it is |", "|---|---|",
             "| `mix_mood.mp3` | integrated arrangement + mood stem, −16 LUFS (compare with `demo/%s/mix.mp3`) |" % tid,
             "| `mood.mp3` | the mood stem alone, at its mix level (after ducking) |",
             "| `weights.png` | frames, SigLIP-only weights, blended weights, data curves |",
             "| `mood.png` | weights sent to the model, mood and mix spectrograms (cyan: lead pitch) |",
             "| `summary.json` | the numbers below plus prompts and settings |", "",
             "## Prompts (anchor: music prompt ← visual text, data link)", ""]
    for a in w["anchors"]:
        dl = f" — data: {'+' if a['data_link'][1] > 0 else '−'}{a['data_link'][0]}" if a["data_link"] else ""
        lines.append(f"- **{a['name']}**: “{a['music']}” ← “{a['visual']}”{dl}")
    lines += ["", "## Fidelity (note-level pyin round trip, same method as `arrange.py`)", "",
              "| Audio | R² | pitch within ½ st |", "|---|---|---|"]
    for k, lab in (("mix_without_mood", "full mix, no mood (integrate)"), ("mix_with_mood", "full mix + mood"),
                   ("control_mix_without_mood", "shuffled control mix, no mood"),
                   ("control_mix_with_mood", "shuffled control mix + mood")):
        lines.append(f"| {lab} | {f(fid[k].get('r2'))} | {f(fid[k].get('pitch_within_half_semitone'), 2)} |")
    lines += ["", f"R² change from the mood: {f(-rep['r2_drop'] if rep['r2_drop'] is not None else None, 4)}. "
              f"Mood sits {rep['mood_db_vs_melody']:.0f} dB under the melody's active RMS before the sidechain " +
              (f"(ducked {rep['ducking']['mean_gr_db']:.1f} dB on average)." if rep.get("ducking") else "(no sidechain)."), "",
              "## Does the mood follow the picture?", "",
              "Correlation of each data-linked anchor's weight with its data curve. *Vision only* is SigLIP "
              "alone; *blended* is what was sent to the model (it includes the data curve, so it correlates by "
              "construction). *Detrended* removes a linear trend from both series first.", "",
              "| Anchor ↔ curve | vision only | vision, detrended | blended |", "|---|---|---|---|"]
    for n, d in w["linked_correlations"].items():
        lines.append(f"| {n} ↔ {'' if d['sign'] > 0 else '1 − '}{d['curve']} | {f(d['r_image_only'], 2)} | "
                     f"{f(d['r_image_only_detrended'], 2)} | {f(d['r_blended'], 2)} |")
        for s, x in (d.get("segments") or {}).items():
            lines.append(f"| … inside “{s}” (n={x['n']}) | {f(x['r_image_only'], 2)} | | {f(x['r_blended'], 2)} |")
    sf_ = rep.get("style_follow") or {}
    if sf_.get("mean_r") is not None:
        lines += ["", "## Does the generated audio follow the weights?", "",
                  f"MusicCoCa audio embeddings of the raw mood ({sf_['clips']} clips of 10 s, every 2 s) against the "
                  f"prompts' text embeddings: mean r over anchors between “clip ↔ prompt similarity” and the weight "
                  f"the model was given = {f(sf_['mean_r'], 2)} (same audio against time-reversed weights: "
                  f"{f(sf_['mean_r_time_reversed_weights'], 2)}). The closest prompt is the most-weighted prompt in "
                  f"{sf_['top_prompt_agreement']:.0%} of clips (chance {sf_['chance_agreement']:.0%}).", "",
                  "| Anchor | r | share of clips closest |", "|---|---|---|"]
        for n_, v in sf_["r_per_anchor"].items():
            lines.append(f"| {n_} | {f(v, 2)} | {sf_['closest_prompt_share'][n_]:.0%} |")
    lines += ["", "## Key", "",
              f"Notes conditioning: `{gen['notes_mode']}` (out-of-scale pitches and everything from one semitone "
              "under the lead's lowest note upward set OFF; harmony chord tones below that FREE). "
              f"Chroma share on the track's scale: {kc['scale_share']:.2f} (chance {kc['scale_share_chance']:.2f})."]
    if "harmony_chroma_cos_shift0" in kc:
        lines.append(f"Chroma cosine with the harmony stem per 2 s: {kc['harmony_chroma_cos_shift0']:.2f} in place vs "
                     f"{kc['harmony_chroma_cos_other_shifts_mean']:.2f} averaged over the 11 transpositions; best "
                     f"shift {kc['best_shift_semitones']} st; shift 0 best in {kc['share_segments_shift0_best']:.0%} "
                     f"of segments → {'**clash**' if kc['clash'] else 'no systematic clash'}.")
    lines += ["", "## Runtime", "",
              f"{gen['size']} on {gen['device'].upper()}: {gen['gen_s']:.0f} s for {gen['audio_s']:.0f} s of audio "
              f"({gen['gen_s_per_audio_min']:.0f} s per audio minute, {gen['realtime_factor']:.2f}× real time), "
              f"plus {gen['load_s']:.0f} s load/compile."]
    for o in others:
        fo = o["rep"]["fidelity"]
        lines.append(f"Also rendered with `{o['gen']['size']}` ({o['gen']['device'].upper()}): full-mix R² with mood "
                     f"{f(fo['mix_with_mood']['r2'])}, {o['gen']['gen_s_per_audio_min']:.0f} s per audio minute.")
    lines += ["", "Reproduce:", "", "```bash",
              f"srun -A p32576 -p short -t 00:30:00 --mem=24G -c 8 scripts/mood/weights.sh {tid}",
              f"srun -A p32576 -p short -t 02:00:00 --mem=48G -c 8 scripts/mood/render.sh mrt2_small {tid}   # CPU",
              "sbatch scripts/mood/gpu_base.sbatch                                                     # GPU, base",
              "```", "", "Nothing here has been listened to; every judgement is from the metrics above.", ""]
    return "\n".join(lines)


def main(ids):
    rows = []
    for tid in ids:
        avail = [r for r in (load(tid, t) for t in TAGS) if r]
        if not avail:
            print(f"{tid}: no render yet")
            continue
        R, others = avail[0], avail[1:]
        dst = DEMO / f"{tid}_mood"
        dst.mkdir(parents=True, exist_ok=True)
        tag = R["tag"]
        mp3(R["dir"] / f"mix_mood_{tag}.wav", dst / "mix_mood.mp3", 128)
        mp3(R["dir"] / f"mood_stem_{tag}.wav", dst / "mood.mp3", 96)
        shutil.copy(R["dir"] / "weights.png", dst / "weights.png")
        shutil.copy(R["dir"] / f"mood_{tag}.png", dst / "mood.png")
        w = R["w"]
        summary = {"id": tid, "tag": tag, "generate": {k: v for k, v in R["gen"].items() if k != "log"},
                   "mix": R["rep"], "weights": {k: w[k] for k in ("model", "fps", "tau", "debias", "beta", "smooth_s",
                                                                  "anchors", "mean_weight", "mean_weight_image_only",
                                                                  "linked_correlations")},
                   "other_renders": [{"tag": o["tag"], "size": o["gen"]["size"], "device": o["gen"]["device"],
                                      "gen_s_per_audio_min": o["gen"]["gen_s_per_audio_min"],
                                      "fidelity": o["rep"]["fidelity"], "key_check": o["rep"]["key_check"]}
                                     for o in others]}
        (dst / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        (dst / "README.md").write_text(readme(tid, R, others))
        for r in avail:
            fid = r["rep"]["fidelity"]
            name, rv, rd, rb = headline_corr(r["w"], tid)
            rows.append({"id": tid, "size": r["gen"]["size"], "device": r["gen"]["device"],
                         "r2_without": fid["mix_without_mood"]["r2"], "r2_with": fid["mix_with_mood"]["r2"],
                         "ctrl": fid["control_mix_with_mood"]["r2"], "ctrl0": fid["control_mix_without_mood"]["r2"],
                         "corr": name, "r_vision": rv, "r_detr": rd, "r_blend": rb,
                         "spm": r["gen"]["gen_s_per_audio_min"],
                         "follow": (r["rep"].get("style_follow") or {}).get("mean_r"),
                         "follow_rev": (r["rep"].get("style_follow") or {}).get("mean_r_time_reversed_weights"), "clash": r["rep"]["key_check"].get("clash"),
                         "scale": r["rep"]["key_check"]["scale_share"]})
    (DEMO / "learned").mkdir(exist_ok=True)
    (DEMO / "learned" / "mood_table.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    tab = ["| Track | Model (device) | Full-mix R² without mood | with mood | Control R² (mix + mood; no mood) | "
           "Weight–data r: anchor, vision only (detrended) / blended | Audio follows weights: mean r (reversed) | Gen. s per audio min | Key clash? |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tab.append(f"| {r['id']} | {r['size']} ({r['device'].upper()}) | {f(r['r2_without'])} | {f(r['r2_with'])} | "
                   f"{f(r['ctrl'], 2)}; {f(r['ctrl0'], 2)} | {r['corr']}: {f(r['r_vision'], 2)} ({f(r['r_detr'], 2)}) / "
                   f"{f(r['r_blend'], 2)} | {f(r['follow'], 2)} ({f(r['follow_rev'], 2)}) | {r['spm']:.0f} | {'yes' if r['clash'] else 'no'} (scale share {r['scale']:.2f}) |")
    (DEMO / "learned" / "mood_table.md").write_text("\n".join(tab) + "\n")
    print("\n".join(tab))


if __name__ == "__main__":
    main(sys.argv[1:] or ["5313", "3912", "5209"])
