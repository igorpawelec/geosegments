"""Why is the number of segments not monotone in k on the synthetic scene?"""
import numpy as np
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import noise_units, tessellate

rng = np.random.default_rng(7)
H = W = 240
truth = np.zeros((H, W), dtype=np.int32)
truth[:, 120:] = 1
truth[150:, :] += 2
means = rng.normal(0, 12, size=(4, 3))
data = means[truth].transpose(2, 0, 1) + rng.normal(0, 3.0, size=(3, H, W))
z, valid = noise_units(data, None, smooth=0.0, stabilised=False)
print("pixel sd per band in noise units:", np.round([z[i][truth == 0].std() for i in range(3)], 3))
for k in (2, 3, 3.5, 4, 5, 6, 8):
    for compete in (False, True):
        lab, n, seeded = tessellate(z, valid, k=k, compete=compete, return_stages=True)
        sizes = np.bincount(lab.ravel())
        print(f"k={k:4}: compete={compete!s:5}  seeds {n:5d}  labels used {len(np.unique(lab)):5d}  free after seeding {(seeded == -1).mean():.3f}  "
              f"largest segments {np.sort(sizes)[-4:][::-1]}  1-px {int((sizes == 1).sum())}")
