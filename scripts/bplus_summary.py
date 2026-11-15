"""One markdown table for the MuseControlLite B+ tracks, from demo/<id>_mclbplus/bplus_report.json.

    python scripts/bplus_summary.py            # prints the table used in demo/learned/README.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("5209", "5313", "3912", "4950", "5176", "5011", "5058", "5669", "5046")


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def dbs(x):
    return "–" if x is None else f"{10 * math.log10(max(x, 1e-12)):.1f}"


def main():
    print("| track | MCL take | lead-stem R² | full-mix R²: det / old aug / B+ | within ½ st (B+) | control R² (B+ e2e) "
          "| HF perc. energy, mix (dB): det / old / B+ | MCL bed HF perc. (dB) raw → clean | voice-bed share: det / old / B+ "
          "| mix 40–250 Hz (dB): det / old / B+ | B+ mix |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for tid in ORDER:
        p = ROOT / "demo" / f"{tid}_mclbplus" / "bplus_report.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        m, det, old = r["mix"], r["reference"]["deterministic"], r["reference"].get("old_augment") or {}
        take = "pilot (reused)" if r["reused_pilot_take"] else "new CPU render"
        mm, dm = m.get("mix") or {}, det.get("mix") or {}
        print(f"| {tid} | {take} | {f((det.get('lead_stem') or {}).get('r2'))} "
              f"| {f(dm.get('r2'))} / {f((old.get('mix') or {}).get('r2'))} / {f(mm.get('r2'))} "
              f"| {f(mm.get('pitch_within_half_semitone'), 2)} | {f((m.get('control_e2e') or {}).get('r2'))} "
              f"| {dbs(det['percussion']['hf_perc_share'])} / {dbs((old.get('percussion') or {}).get('hf_perc_share'))} / "
              f"{dbs(m['percussion']['hf_perc_share'])} "
              f"| {dbs(r['bed']['raw']['hf_perc_share'])} → {dbs(r['bed']['clean']['hf_perc_share'])} "
              f"| {f(det['energy_share']['voice_beds'])} / {f((old.get('energy_share') or {}).get('voice_beds'))} / "
              f"{f(m['energy_share']['voice_beds'])} "
              f"| {f(det['percussion']['low_db'], 1)} / {f((old.get('percussion') or {}).get('low_db'), 1)} / "
              f"{f(m['percussion']['low_db'], 1)} | {r['choice']['how']} |")


if __name__ == "__main__":
    main()
