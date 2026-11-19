"""Are the reference objects of Krynki 2 more heterogeneous than those of Krynki 1?

For every patch, in noise units: the spread of its own pixels around their median colour,
and the distance of that median to the median of a ring around the patch. A tolerance
against one colour can hold an object only while its spread is well below its contrast.
"""
import numpy as np
from scipy import ndimage
import paths  # noqa: F401  puts the repository and its sibling packages on sys.path
from sigments import noise_units
from data import load

for scene in ("krynki1", "krynki2"):
    sc = load(scene)
    z, valid = noise_units(sc["data"], sc["valid"], smooth=0.7, stabilised=True)
    patches, n = ndimage.label(sc["crowns"] > 0, structure=np.ones((3, 3)))
    objs = ndimage.find_objects(patches)
    spread, contrast, size = [], [], []
    for i, sl in enumerate(objs, start=1):
        r0, r1 = max(sl[0].start - 8, 0), min(sl[0].stop + 8, patches.shape[0])
        c0, c1 = max(sl[1].start - 8, 0), min(sl[1].stop + 8, patches.shape[1])
        m = patches[r0:r1, c0:c1] == i
        zz = z[:, r0:r1, c0:c1]
        ring = ndimage.binary_dilation(m, iterations=6) & ~ndimage.binary_dilation(m, iterations=2) & valid[r0:r1, c0:c1] & (patches[r0:r1, c0:c1] == 0)
        if m.sum() < 8 or ring.sum() < 8:
            continue
        med = np.median(zz[:, m], axis=1)
        dev = np.sqrt(((zz[:, m] - med[:, None]) ** 2).sum(axis=0))
        spread.append(np.percentile(dev, 90))
        contrast.append(np.linalg.norm(med - np.median(zz[:, ring], axis=1)))
        size.append(int(m.sum()))
    spread, contrast, size = np.array(spread), np.array(contrast), np.array(size)
    ratio = spread / contrast
    print(f"{scene}: {len(size)} patches, median size {int(np.median(size))} px | 90th percentile deviation inside the patch: median {np.median(spread):.1f} | "
          f"contrast to the surroundings: median {np.median(contrast):.1f} | spread / contrast: median {np.median(ratio):.2f}, share of patches above 1: {(ratio > 1).mean():.2f}")
