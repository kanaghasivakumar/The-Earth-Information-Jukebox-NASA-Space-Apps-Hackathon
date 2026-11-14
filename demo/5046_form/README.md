# Form layer on 5046 (Daily Arctic Sea Ice, two-year movie) — the "returns" example

`python scripts/run_form.py --id 5046` → sections A B C **A B** (returns at 18.0 s and 21.5 s).
Using the burned-in dates: A = Apr 2021 → Jan 2022, B = Jan → Jul 2022 (winter/spring ice), C = Jul → Oct 2022
(summer melt), A returns Oct 2022 (freeze-up), B returns Jan 2023. The second year lines up with seasons; the first
year is lumped into one section. Novelty correlation DINOv2 vs handcrafted: 0.74. Camera shift ≤ 0.17 px.
The date labels in the corner also change, which may leak into the embedding (unverified).
