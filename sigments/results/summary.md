# Results on the two Krynki windows

Mean IoU of a reference object with the segment that overlaps it most. Crowns: the reference crowns.
Patches: touching reference crowns joined into one object. Every method at its own best setting per scene.

## krynki1

| method | best setting | crown IoU | crowns with IoU >= 0.5 | patch IoU | patches with IoU >= 0.5 | segments |
|---|---|---|---|---|---|---|
| prototype, default (stabilised + smoothed) | 9 | 0.450 | 0.46 | 0.484 | 0.49 | 33548 |
| prototype, whitened + smoothed | 9 | 0.412 | 0.36 | 0.431 | 0.40 | 31129 |
| adaptels | 200 | 0.478 | 0.48 | 0.476 | 0.48 | 49901 |
| MRS, shape 0.5 / compactness 0.7 | 27 | 0.455 | 0.40 | 0.461 | 0.41 | 34534 |
| MRS, shape 0.1 / compactness 0.5 | 33 (crowns), 40 (patches) | 0.427 | 0.36 | 0.434 | 0.39 | 25465 |

## krynki2

| method | best setting | crown IoU | crowns with IoU >= 0.5 | patch IoU | patches with IoU >= 0.5 | segments |
|---|---|---|---|---|---|---|
| prototype, default (stabilised + smoothed) | 11 | 0.324 | 0.12 | 0.318 | 0.10 | 29234 |
| prototype, whitened + smoothed | 10 | 0.312 | 0.11 | 0.317 | 0.11 | 35178 |
| adaptels | 260 (crowns), 340 (patches) | 0.456 | 0.39 | 0.449 | 0.41 | 22465 |
| MRS, shape 0.5 / compactness 0.7 | 33 | 0.417 | 0.30 | 0.413 | 0.27 | 24063 |
| MRS, shape 0.1 / compactness 0.5 | 40 | 0.389 | 0.25 | 0.379 | 0.23 | 25127 |

## Transfer: the setting chosen on one scene, applied to the other

Patch IoU. "Loss" is the distance to the best the method can do on the target scene.

| method | setting from Krynki 1 | on Krynki 2 | best on Krynki 2 | loss | setting from Krynki 2 | on Krynki 1 | best on Krynki 1 | loss |
|---|---|---|---|---|---|---|---|---|
| prototype, default (stabilised + smoothed) | 9 | 0.302 | 0.318 | 0.017 | 11 | 0.460 | 0.484 | 0.025 |
| prototype, whitened + smoothed | 9 | 0.307 | 0.317 | 0.010 | 10 | 0.429 | 0.431 | 0.002 |
| adaptels | 200 | 0.420 | 0.449 | 0.029 | 340 | 0.388 | 0.476 | 0.088 |
| MRS, shape 0.5 / compactness 0.7 | 27 | 0.385 | 0.413 | 0.028 | 33 | 0.459 | 0.461 | 0.003 |
| MRS, shape 0.1 / compactness 0.5 | 40 | 0.379 | 0.379 | 0.000 | 40 | 0.434 | 0.434 | 0.000 |

## Run time

Krynki 1, 2400 x 2400 px, 4 bands, 5.4 M valid pixels, measured alone on the machine (bench/timing.py).

| method | seconds |
|---|---|
| prototype, stabilised + smoothed | 23.1 (19.0 noise units + 4.1 segmentation) |
| prototype, whitened + smoothed | 16.8 (12.7 + 4.1) |
| prototype, whitened only | 15.0 (11.2 + 3.8) |
| adaptels | 4.0 |
| MRS, shape 0.5 / compactness 0.7 | 16.6 |
