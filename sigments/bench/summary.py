"""Assemble results/summary.md: best setting per method and scene, and the transfer between scenes."""
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from paths import OUT
import csv
import glob

NAMES = {"proto_st": "prototype, default (stabilised + smoothed)", "proto_wh": "prototype, whitened + smoothed", "adaptels": "adaptels",
         "mrs_shape": "MRS, shape 0.5 / compactness 0.7", "mrs": "MRS, shape 0.1 / compactness 0.5"}
rows = []
for f in glob.glob(OUT + "full_krynki*_*.csv") + glob.glob(OUT + "final_krynki*_*.csv"):
    with open(f, newline="") as fh:
        rows += [r for r in csv.DictReader(fh) if r["algorithm"] in NAMES]
for r in rows:
    for key in r:
        if key not in ("scene", "algorithm"):
            r[key] = float(r[key])
by = {}
for r in rows:
    by.setdefault((r["scene"], r["algorithm"]), {})[r["param"]] = r

lines = ["# Results on the two Krynki windows", "",
         "Mean IoU of a reference object with the segment that overlaps it most. Crowns: the reference crowns.",
         "Patches: touching reference crowns joined into one object. Every method at its own best setting per scene.", ""]
for scene in ("krynki1", "krynki2"):
    lines += [f"## {scene}", "", "| method | best setting | crown IoU | crowns with IoU >= 0.5 | patch IoU | patches with IoU >= 0.5 | segments |", "|---|---|---|---|---|---|---|"]
    for alg in NAMES:
        runs = by.get((scene, alg))
        if not runs:
            continue
        bc = max(runs.values(), key=lambda r: r["crown_iou_mean"])
        bp = max(runs.values(), key=lambda r: r["patch_iou_mean"])
        setting = f"{bc['param']:g}" if bc["param"] == bp["param"] else f"{bc['param']:g} (crowns), {bp['param']:g} (patches)"
        lines.append(f"| {NAMES[alg]} | {setting} | {bc['crown_iou_mean']:.3f} | {bc['crown_iou50']:.2f} | {bp['patch_iou_mean']:.3f} | {bp['patch_iou50']:.2f} | {int(bp['segments'])} |")
    lines.append("")
lines += ["## Transfer: the setting chosen on one scene, applied to the other", "",
          "Patch IoU. \"Loss\" is the distance to the best the method can do on the target scene.", "",
          "| method | setting from Krynki 1 | on Krynki 2 | best on Krynki 2 | loss | setting from Krynki 2 | on Krynki 1 | best on Krynki 1 | loss |", "|---|---|---|---|---|---|---|---|---|"]
for alg in NAMES:
    a, b = by.get(("krynki1", alg)), by.get(("krynki2", alg))
    if not a or not b:
        continue
    pa = max(a.values(), key=lambda r: r["patch_iou_mean"])["param"]
    pb = max(b.values(), key=lambda r: r["patch_iou_mean"])["param"]
    best_a = max(r["patch_iou_mean"] for r in a.values())
    best_b = max(r["patch_iou_mean"] for r in b.values())
    on_b = b[pa]["patch_iou_mean"] if pa in b else float("nan")
    on_a = a[pb]["patch_iou_mean"] if pb in a else float("nan")
    lines.append(f"| {NAMES[alg]} | {pa:g} | {on_b:.3f} | {best_b:.3f} | {best_b - on_b:.3f} | {pb:g} | {on_a:.3f} | {best_a:.3f} | {best_a - on_a:.3f} |")
lines += ["", "## Run time", "",
          "Krynki 1, 2400 x 2400 px, 4 bands, 5.4 M valid pixels, measured alone on the machine (bench/timing.py).", "",
          "| method | seconds |", "|---|---|",
          "| prototype, stabilised + smoothed | 23.1 (19.0 noise units + 4.1 segmentation) |",
          "| prototype, whitened + smoothed | 16.8 (12.7 + 4.1) |",
          "| prototype, whitened only | 15.0 (11.2 + 3.8) |",
          "| adaptels | 4.0 |",
          "| MRS, shape 0.5 / compactness 0.7 | 16.6 |"]
open(OUT + "summary.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
