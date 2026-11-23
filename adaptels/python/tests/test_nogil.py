"""The compiled kernels release the GIL while they run.

A desktop application runs these functions in a worker thread so that its
window keeps answering. That only works if the kernel lets go of the GIL: a
numba function compiled without ``nogil=True`` holds it from entry to return,
and the main thread waits the whole time. Measured on the test orthophoto
tiled 5 x 5 (2000 x 2000 px), a main thread sleeping 1 ms per iteration
stalled 2.2-2.5 s of 2.3-2.5 s for ``adaptels_from_array`` under 0.10.5 and
2.2 s of 4.3 s for ``sicle_from_array``; with the kernels declared ``nogil``
the longest stall is 5 ms for either.

The declaration is checked on every compiled function found in the package's
modules, so a kernel added later without it fails here rather than in an
application's frozen window.

Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""

import importlib
import os
import pkgutil
import threading
import time

import numpy as np
import pytest

import pygeoadaptels

numba_dispatcher = pytest.importorskip("numba.core.dispatcher")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "test_data", "SNP_21_2020_1.tif")


def _kernels():
    """Every numba-compiled function defined in a module of the package."""
    found = {}
    for info in pkgutil.walk_packages(pygeoadaptels.__path__,
                                      pygeoadaptels.__name__ + "."):
        if info.name.endswith(".__main__"):
            continue                       # runs the CLI on import
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if isinstance(obj, numba_dispatcher.Dispatcher):
                # sicle imports the heap from core; count each kernel once,
                # under the module that defines it
                found[f"{obj.py_func.__module__}.{obj.py_func.__name__}"] = obj
    return found


def test_every_kernel_is_declared_nogil():
    kernels = _kernels()
    # Guards the walk itself: an empty result must not read as a pass.
    for expected in ("pygeoadaptels.core._create_adaptels",
                     "pygeoadaptels.sicle._ift_fmax",
                     "pygeoadaptels.sicle._compute_seed_relevance"):
        assert expected in kernels, f"{expected} not found by the module walk"
    # targetoptions holds the options the decorator was given, before any
    # compilation, which is what a kernel that is never called needs.
    holding = sorted(name for name, k in kernels.items()
                     if not k.targetoptions.get("nogil", False))
    assert not holding, (
        f"compiled without nogil=True, so they hold the GIL while they run "
        f"and freeze the thread that waits for them: {holding}"
    )


def test_main_thread_runs_while_a_kernel_works():
    rasterio = pytest.importorskip("rasterio")
    from pygeoadaptels import adaptels_from_array

    with rasterio.open(DATA) as src:
        small = src.read().astype(np.float64)
    large = np.tile(small, (1, 3, 3))      # 1200 x 1200 px, most of a second
    adaptels_from_array(small[:, :50, :50], threshold=60.0)   # compile first

    done = threading.Event()

    def work():
        adaptels_from_array(large, threshold=60.0)
        done.set()

    worker = threading.Thread(target=work)
    t0 = time.perf_counter()
    worker.start()
    last = time.perf_counter()
    longest, loops = 0.0, 0
    while not done.is_set():
        time.sleep(0.001)
        now = time.perf_counter()
        longest = max(longest, now - last)
        last = now
        loops += 1
    worker.join()
    elapsed = time.perf_counter() - t0

    # One kernel call does nearly all the work here, so a kernel holding the
    # GIL stalls the main thread for nearly the whole run. Released, the
    # stall is a few milliseconds; half the run is the generous margin.
    assert loops >= 20, (
        f"the main thread ran {loops} times in {elapsed:.2f} s")
    assert longest < 0.5 * elapsed, (
        f"the main thread stalled {longest:.2f} s of {elapsed:.2f} s")
