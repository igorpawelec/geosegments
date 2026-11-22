"""Shared helpers for the algorithm classes: the dependency gate, the JIT
warm-up, the wrapper that turns a package's refusal into a message, and the
parameter-flag helper that keeps the dialogs from overwhelming an operator."""
import contextlib
import logging

from qgis.core import QgsProcessingException

from ..deps import ensure_dependencies, manual_hint

_WARMED = False


@contextlib.contextmanager
def package_errors():
    """Show what a package refused as its message, not as a Python traceback.

    The packages validate their input and say what is wrong with it in the
    exception text: a threshold on the wrong scale, two weights for three
    bands, a seed point outside the raster, an output folder that does not
    exist. Raised bare inside ``processAlgorithm`` that text reaches the
    operator at the bottom of a traceback through this plugin's and the
    package's source, which reads as the plugin having crashed.
    ``QgsProcessingException`` is the error Processing prints as a message.

    Converted: the exception types the packages use for bad input and
    unusable files, and whatever rasterio and fiona raise of their own. The
    traceback is not thrown away with that: it goes to the Log Messages panel
    under the plugin's name, because some of those types are also what a
    defect of the plugin would raise, and then it is the part that matters.
    Anything else (a TypeError, a KeyError, numba failing to compile) is
    left alone and arrives with its traceback, as a defect should.
    """
    try:
        with library_logs():
            yield
    except QgsProcessingException:
        raise
    except Exception as exc:
        ours = isinstance(exc, (ValueError, OSError, IndexError, ImportError, RuntimeError))
        theirs = type(exc).__module__.split(".")[0] in ("rasterio", "fiona")
        if not (ours or theirs):
            raise
        _log_traceback()
        text = str(exc).strip() or type(exc).__name__
        raise QgsProcessingException(text) from exc


class _ToMessageLog(logging.Handler):
    """What a library logs, into the Log Messages panel under the plugin's name."""

    def emit(self, record):
        try:
            from qgis.core import QgsMessageLog
            QgsMessageLog.logMessage(f"{record.name}: {record.getMessage()}", "GeoAdaptels + GeoPalette")
        except Exception:
            pass


@contextlib.contextmanager
def library_logs(names=("rasterio", "fiona")):
    """Keep what rasterio and fiona log during a run away from the root logger.

    GDAL's warnings reach Python as log records of rasterio; one arrives with
    the first polygonisation of every session ("'Memory' driver is deprecated
    since GDAL 3.11"). The two libraries have no handler of their own, so the
    record travels up to the root logger, and in a QGIS on Windows the root
    logger may hold a handler that cannot write: a plugin that calls
    ``logging.basicConfig()`` while QGIS starts gets a stream handler on
    ``sys.stderr``, which is None at that moment and stays None in the handler.
    Every record that reaches it raises inside ``logging``, which prints
    "--- Logging error ---" and a traceback through this plugin's source into
    the Python console. The run itself is fine; it reads as a crash.

    For the length of a run the two loggers get a handler that writes to the
    Log Messages panel and stop propagating. Both are put back afterwards, so
    nothing changes for anything else that uses them.
    """
    handler = _ToMessageLog()
    saved = []
    for name in names:
        logger = logging.getLogger(name)
        saved.append((logger, logger.propagate))
        logger.addHandler(handler)
        logger.propagate = False
    try:
        yield
    finally:
        for logger, propagate in saved:
            logger.removeHandler(handler)
            logger.propagate = propagate


def _log_traceback():
    """The traceback of the exception being handled, to the Log Messages panel."""
    try:
        import traceback

        from qgis.core import QgsMessageLog
        QgsMessageLog.logMessage(traceback.format_exc(), "GeoAdaptels + GeoPalette")
    except Exception:
        pass


def warm_jit(feedback=None):
    """Compile the numba kernels here, on the main thread.

    **Call this from prepareAlgorithm, never from processAlgorithm.** The
    packages compute inside ``@njit(cache=True)`` functions, and the first call
    against a cold cache has to compile them. Compiling on the worker thread
    Processing runs algorithms on takes QGIS down with an access violation --
    not an exception, so no amount of try/except in the algorithm can catch it.
    ``prepareAlgorithm`` runs on the main thread before the task starts, which
    is where that compilation is safe.

    The cache goes cold at exactly the wrong moment: installing the plugin from
    a zip replaces the vendored package directory, ``__pycache__`` and the
    ``.nbc`` files with it. So every update armed the crash for the operator's
    next run, and a single successful run from the Python console -- main
    thread -- disarmed it again. That is the whole reason a run could fail from
    the toolbox and then succeed, unchanged, minutes later.

    Both kernel modules are warmed: ``core`` backs Adaptels, and ``sicle``
    backs both SICLE and Grow Seeds, which borrows ``_ift_fmax`` from it.
    ``distance_type`` and ``connectivity`` reach the kernels as int32 rather
    than as strings, so one warm-up covers every parameter combination the
    dialogs can produce.

    Runs once per QGIS session and is never fatal: if the packages are missing,
    ``require_packages`` reports that properly during the run.
    """
    global _WARMED
    if _WARMED:
        return
    try:
        import contextlib
        import io

        import numpy as np

        from pygeoadaptels import adaptels_from_array
        from pygeoadaptels.sicle import sicle_from_array
    except Exception:
        return
    if feedback is not None:
        feedback.pushInfo(
            "Preparing the compute kernels (first run after an update only)...")
    data = np.zeros((1, 8, 8), dtype=np.float64)
    data[0, 2:5, 2:5] = 100.0
    for call in (lambda: adaptels_from_array(data, threshold=10.0),
                 lambda: sicle_from_array(data, n_segments=2,
                                          n_oversampling=6, n_iterations=1)):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                call()
        except Exception as exc:
            if feedback is not None:
                feedback.pushInfo(f"Kernel warm-up skipped: {exc}")
    _WARMED = True


def band_hint(raster_path, rgb, feedback):
    """Say so when the band picked as red looks like infrared.

    Vegetation is dark in red and bright in near-infrared, so over a forested
    sample the ratio of the two separates cleanly: measured against known files,
    a CIR raster gives about 2.1 and a true-colour one about 0.86. Anything past
    1.25 means the band being fed to the conversion as red is almost certainly
    the infrared one.

    **This only reports.** Guessing the band order for the operator is how the
    silent version of this mistake happens, and the database's own inventory
    report already got two rasters wrong this way. A wrong band assignment is
    not an error either -- running a colour space over (NIR, R, G) deliberately
    is a real technique -- so it must not stop the run.
    """
    if feedback is None:
        return
    try:
        import numpy as np
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(raster_path) as src:
            w = min(1024, src.width)
            h = min(1024, src.height)
            win = Window((src.width - w) // 2, (src.height - h) // 2, w, h)
            r = src.read(int(rgb[0]), window=win).astype("float64")
            g = src.read(int(rgb[1]), window=win).astype("float64")
        ok = np.isfinite(r) & np.isfinite(g) & (r != 0) & (g != 0)
        if ok.sum() < 1000:
            return
        ratio = float(np.median(r[ok]) / max(np.median(g[ok]), 1e-6))
        if ratio > 1.25:
            feedback.pushInfo(
                f"Note: band {rgb[0]} is {ratio:.2f}x brighter than band "
                f"{rgb[1]} over this sample, which is what near-infrared looks "
                f"like, not red. If this is a CIR or NRGB raster, set the band "
                f"numbers to match it -- the conversion treats them as R, G, B "
                f"exactly as given.")
    except Exception:
        pass


def require_packages(feedback):
    ok, missing = ensure_dependencies(feedback)
    if not ok:
        raise QgsProcessingException(
            "This algorithm needs the Python packages: "
            + ", ".join(missing)
            + ".\nInstall them into the plugin's libs folder and restart QGIS:\n  "
            + manual_hint())


def advanced(param):
    """Collapse a parameter under the dialog's 'Advanced' section.

    Grow Seeds alone has ten parameters; showing all of them at once buries
    the three that are actually tuned per scene. Nothing is hidden -- the
    section is one click away -- and the defaults are the values the guide
    recommends.

    Wrapped: if a future QGIS moves the flag enum, the parameter simply stays
    visible rather than the whole dialog failing to build.
    """
    try:
        from qgis.core import QgsProcessingParameterDefinition
        param.setFlags(
            param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
    except Exception:
        pass
    return param
