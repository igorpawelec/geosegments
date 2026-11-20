"""Where things are, on any of the computers this repository is worked on from.

Importing this module puts the repository root and the sibling packages the baselines need
(pygeoadaptels, obiastudio) on ``sys.path``, so a script runs with nothing but the right
interpreter. Two environment variables override the defaults:

``SIGMENTS_APPS``  folder that holds pygeoadaptels and obiastudio
                   (default: ``Apps`` beside the folder this repository's parent, e.g. E:/Apps)
``SIGMENTS_DATA``  the Krynki exercise with its orthophotos and reference crowns
                   (default: ``OneDrive/5_DYDAKTYKA/_B4/B4_OBIA/ZADANIE_KRYNKI`` under the home folder)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "results").as_posix() + "/"
APPS = Path(os.environ.get("SIGMENTS_APPS", ROOT.parents[1] / "Apps")).as_posix() + "/"
DATA = Path(os.environ.get("SIGMENTS_DATA", Path.home() / "OneDrive/5_DYDAKTYKA/_B4/B4_OBIA/ZADANIE_KRYNKI")).as_posix() + "/"

for _p in (ROOT, Path(APPS) / "pygeoadaptels", Path(APPS) / "obiastudio"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.makedirs(OUT, exist_ok=True)
