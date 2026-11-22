"""The plugin's own logic that needs no QGIS and no packages: the error wrapper,
the reference that keeps a post-processor alive, the help markup, the
Grow seeds limits and the pruning of libs/. Runs against tests/qgis_stub.
"""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (os.path.join(HERE, "qgis_stub"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qgis.core import QgsProcessingException  # noqa: E402

from geoadaptels_palette import deps, styling  # noqa: E402
from geoadaptels_palette.algorithms import _base  # noqa: E402

ALGORITHMS = {
    "adaptels": "AdaptelsAlgorithm",
    "convert_colourspace": "ConvertColourSpaceAlgorithm",
    "enforce_connectivity": "EnforceConnectivityAlgorithm",
    "grow_seeds": "GrowSeedsAlgorithm",
    "sicle": "SicleAlgorithm",
    "vectorize": "VectorizeAlgorithm",
}


def _algorithm(module):
    mod = importlib.import_module(f"geoadaptels_palette.algorithms.{module}")
    return getattr(mod, ALGORITHMS[module])()


# -- what a package refuses reaches the operator as its message ----------------

@pytest.mark.parametrize("exc", [
    ValueError("threshold=60.0 cannot be reached with distance='cosine'"),
    OSError("out/labels.tif: No such file or directory"),
    IndexError("band index 2 out of range (not in (1,))"),
    RuntimeError("convert_raster produced no GeoTIFF for space 'lab'"),
    ImportError("fill_holes=True needs scipy."),
    type("WindowError", (Exception,), {"__module__": "rasterio.errors"})("window outside the raster"),
    type("UnsupportedGeometryTypeError", (Exception,), {"__module__": "fiona.errors"})("3D"),
])
def test_a_refusal_becomes_a_processing_exception_with_the_same_text(exc):
    with pytest.raises(QgsProcessingException) as caught:
        with _base.package_errors():
            raise exc
    assert str(caught.value) == str(exc)
    assert caught.value.__cause__ is exc


def test_a_defect_keeps_its_traceback():
    # a KeyError or a TypeError is not something the operator did, nor is numba failing to compile
    typing_error = type("TypingError", (Exception,), {"__module__": "numba.core.errors"})("cannot type")
    for exc in (KeyError("n_labels"), TypeError("unsupported operand"), AttributeError("x"), typing_error):
        with pytest.raises(type(exc)):
            with _base.package_errors():
                raise exc


def test_a_processing_exception_passes_through_unchanged():
    original = QgsProcessingException("band_weights must be numbers")
    with pytest.raises(QgsProcessingException) as caught:
        with _base.package_errors():
            raise original
    assert caught.value is original


# -- the post-processor stays alive until QGIS has used it -----------------------

class _Details:
    def __init__(self):
        self.processor = None

    def setPostProcessor(self, processor):
        self.processor = processor


class _Context:
    """The two calls styling._register makes on a QgsProcessingContext."""

    def __init__(self):
        self.details = {}

    def willLoadLayerOnCompletion(self, dest_id):
        return True

    def layerToLoadOnCompletionDetails(self, dest_id):
        return self.details.setdefault(dest_id, _Details())


def test_every_registered_processor_is_kept_alive():
    """QGIS takes the C++ half and nothing of the Python half: without a
    reference held here the subclass is collected when the algorithm returns
    and the layer loads with QGIS's default style. A run registers up to two,
    and the second must not push out the first."""
    before = len(styling._KEEP_ALIVE)
    ctx = _Context()
    assert styling.style_label_raster(ctx, "labels.tif", 36) is True
    assert styling.style_polygons(ctx, "crowns.gpkg") is True
    kept = list(styling._KEEP_ALIVE)[before:]
    assert len(kept) == 2
    assert ctx.details["labels.tif"].processor is kept[0]
    assert ctx.details["crowns.gpkg"].processor is kept[1]
    assert isinstance(kept[0], styling.LabelRasterPostProcessor) and kept[0].n_labels == 36
    assert isinstance(kept[1], styling.PolygonPostProcessor)


def test_nothing_is_registered_for_an_output_that_is_not_loaded():
    class NotLoaded(_Context):
        def willLoadLayerOnCompletion(self, dest_id):
            return False
    before = len(styling._KEEP_ALIVE)
    assert styling.style_polygons(NotLoaded(), "crowns.gpkg") is False
    assert styling.style_polygons(_Context(), "") is False
    assert len(styling._KEEP_ALIVE) == before


# -- help texts ------------------------------------------------------------------

@pytest.mark.parametrize("module", sorted(ALGORITHMS))
def test_help_does_not_use_the_b_tag(module):
    """QGIS gives the help panel of a Processing dialog its own style sheet with
    ``b { color: #333333; }``. In a dark theme that is dark grey on a dark
    background, and the lead-in of every paragraph vanished. ``<strong>`` is
    as bold and keeps the paragraph's colour."""
    text = _algorithm(module).shortHelpString()
    assert "<b>" not in text and "</b>" not in text
    assert text.count("<strong>") == text.count("</strong>")


# -- Grow seeds: the dialog's defaults do not reach a script that names no limit --

def test_grow_seeds_limits_apply_only_when_given():
    alg = _algorithm("grow_seeds")
    assert alg._optional_set({alg.MAX_RADIUS: 20}, alg.MAX_RADIUS) is True
    assert alg._optional_set({alg.MAX_COST: 15.0}, alg.MAX_COST) is True
    assert alg._optional_set({alg.MAX_RADIUS: None}, alg.MAX_RADIUS) is False     # the field was emptied
    assert alg._optional_set({alg.MAX_COST: ""}, alg.MAX_COST) is False
    # a script that leaves the key out grew without the limit before the dialog
    # had defaults, and still does
    assert alg._optional_set({}, alg.MAX_RADIUS) is False
    assert alg._optional_set({}, alg.MAX_COST) is False


# -- Enforce connectivity: which nodata value the output declares ----------------

@pytest.mark.parametrize("given, declared", [
    (-1, -1), (-1.0, -1), (-9999, -9999), (-9999.0, -9999),      # negative: the pixels still hold it
    (None, -9999),                                               # the input declared none
    (0, -9999), (255, -9999), (65535.0, -9999),                  # a label to the package: renumbered
    (float("nan"), -9999), (-3.4028235e38, -9999), (-1.5, -9999),  # cannot be the nodata of int32 labels
])
def test_label_nodata(given, declared):
    from geoadaptels_palette import core
    assert core.label_nodata(given) == declared


# -- libs/: the numpy pip drags in beside rasterio is removed ----------------------

def _package(folder, name):
    d = folder / name
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("", encoding="utf-8")
    importlib.invalidate_caches()
    return d


@pytest.fixture
def libs(tmp_path, monkeypatch):
    folder = tmp_path / "libs"
    folder.mkdir()
    monkeypatch.setattr(deps, "LIBS_DIR", str(folder))
    monkeypatch.syspath_prepend(str(tmp_path / "unused"))        # keeps monkeypatch restoring sys.path
    sys.path.append(str(folder))
    yield folder
    for name in ("fakeshadow", "fakeown"):
        sys.modules.pop(name, None)


def test_copies_of_what_the_interpreter_provides_are_removed(libs, tmp_path, monkeypatch):
    interpreter = tmp_path / "interpreter"
    _package(interpreter / "site-packages", "fakeshadow")
    monkeypatch.setattr(deps, "_interpreter_roots", lambda: [str(interpreter)])
    monkeypatch.syspath_prepend(str(interpreter / "site-packages"))
    _package(libs, "fakeshadow")
    (libs / "fakeshadow.libs").mkdir()
    (libs / "fakeshadow-2.5.3.dist-info").mkdir()
    _package(libs, "fakeown")                                    # only libs/ has this one
    (libs / "fakeown-1.0.dist-info").mkdir()

    assert deps.prune_shadowed(names=("fakeshadow", "fakeown")) == ["fakeshadow"]
    assert sorted(p.name for p in libs.iterdir()) == ["fakeown", "fakeown-1.0.dist-info"]
    assert deps.prune_shadowed(names=("fakeshadow", "fakeown")) == []      # nothing left to do


def test_a_copy_shadowed_by_another_plugin_stays(libs, tmp_path, monkeypatch):
    # what another plugin provides can go away with that plugin; only the interpreter's own counts
    other = tmp_path / "other_plugin" / "libs"
    _package(other, "fakeshadow")
    monkeypatch.setattr(deps, "_interpreter_roots", lambda: [str(tmp_path / "interpreter")])
    monkeypatch.syspath_prepend(str(other))
    _package(libs, "fakeshadow")
    assert deps.prune_shadowed(names=("fakeshadow",)) == []
    assert (libs / "fakeshadow").is_dir()


def test_numpy_is_what_gets_pruned():
    assert deps.SHADOWED == ("numpy",)


# -- a library's log record does not reach a root logger that cannot write ---------

def test_library_records_stay_away_from_the_root_logger_during_a_run(capsys):
    """What happens in a QGIS on Windows with a plugin that called
    ``logging.basicConfig()`` at start-up: the root logger holds a stream
    handler whose stream is None, and every record that reaches it makes
    ``logging`` print "--- Logging error ---" with a traceback through the
    code that was running, which was this plugin."""
    import logging

    root = logging.getLogger()
    broken = logging.StreamHandler()
    broken.stream = None                              # sys.stderr was None when the handler was made
    root.addHandler(broken)
    rasterio_env = logging.getLogger("rasterio._env")
    before = logging.getLogger("rasterio").propagate
    handlers = [list(logging.getLogger(n).handlers) for n in ("rasterio", "fiona")]   # rasterio adds a NullHandler on import
    try:
        with _base.package_errors():
            rasterio_env.warning("%s in %s", "CPLE_AppDefined", "'Memory' driver is deprecated")
            assert logging.getLogger("rasterio").propagate is False
        inside = capsys.readouterr().err
        assert logging.getLogger("rasterio").propagate is before
        assert [list(logging.getLogger(n).handlers) for n in ("rasterio", "fiona")] == handlers
        rasterio_env.warning("%s in %s", "CPLE_AppDefined", "the same record, outside a run")
        outside = capsys.readouterr().err
    finally:
        root.removeHandler(broken)
    assert "Logging error" not in inside
    assert "Logging error" in outside                 # the test set-up does reproduce the fault


def test_library_logging_is_put_back_when_the_run_fails():
    import logging
    before = [(logging.getLogger(n).propagate, list(logging.getLogger(n).handlers)) for n in ("rasterio", "fiona")]
    with pytest.raises(QgsProcessingException):
        with _base.package_errors():
            raise ValueError("no")
    assert [(logging.getLogger(n).propagate, list(logging.getLogger(n).handlers)) for n in ("rasterio", "fiona")] == before


def test_overlapping_runs_leave_the_loggers_as_they_found_them():
    """Two runs at once, the first to start being the first to finish. Each used
    to save and restore the propagation on its own: the first switched it back
    on under the second, and the second then left it off for good."""
    import logging
    names = ("rasterio", "fiona")
    state = [(logging.getLogger(n).propagate, list(logging.getLogger(n).handlers)) for n in names]
    first, second = _base.library_logs(), _base.library_logs()
    first.__enter__()
    second.__enter__()
    assert all(logging.getLogger(n).propagate is False for n in names)
    first.__exit__(None, None, None)
    assert all(logging.getLogger(n).propagate is False for n in names)      # the second is still running
    second.__exit__(None, None, None)
    assert [(logging.getLogger(n).propagate, list(logging.getLogger(n).handlers)) for n in names] == state
    # and once more, to see that nothing is left counted
    with _base.library_logs():
        assert all(logging.getLogger(n).propagate is False for n in names)
    assert [(logging.getLogger(n).propagate, list(logging.getLogger(n).handlers)) for n in names] == state
