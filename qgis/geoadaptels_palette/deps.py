"""Dependency bootstrap -- the single highest-risk piece of this plugin.

QGIS ships its own Python, and things have to be importable *from that
interpreter*. Since 0.3.0 the two pure-Python packages ride along inside the
plugin (see ``vendor_loader``), so this module only ever has to fetch the
three that carry compiled binaries: ``numba``, ``rasterio`` and ``fiona``.
That removes the dependency on ``git`` being present, which was the most
common way a clean machine failed.

**Where they go: a plugin-private folder, never the user site.** Up to 0.5.4
the install was ``pip install --user``, which writes into
``%APPDATA%/Python/Python312/site-packages`` -- a folder *every* Python 3.12
on the machine reads before its own site-packages, conda environments
included. A scikit-learn installed that way by the GeoSnag plugin shadowed a
conda environment's and broke it, and this plugin's numba, rasterio and fiona
had been shadowing that environment's for months without anyone noticing. So
the binary dependencies are installed with ``pip --target`` into
``<plugin>/libs``, which only this plugin puts on ``sys.path``. Reinstalling
the plugin zip keeps the folder (QGIS replaces the files it ships, not the
folder), and deleting it just triggers a fresh install. Packages already
present in QGIS's Python or in the user site still win: pip is only asked for
what is missing.

Other lessons kept, learned against QGIS 3.40/3.44 and 4.2 (Python 3.12) on
Windows:

1. **``sys.executable`` is a launcher, not the interpreter.** On Windows QGIS
   it is ``bin/python3.exe``, which does not initialise as a subprocess ("No
   module named 'encodings'") and makes pip exit 1 with no useful output. The
   real interpreter sits at ``sys.prefix`` (``apps/PythonXY`` on Windows,
   ``prefix/bin`` on Linux). ``_qgis_python()`` resolves that.

2. **rasterio/fiona past the numpy-2 line drag in numpy 2.x, which breaks
   numba and scipy on the numpy 1.26 QGIS 3 ships.** On an interpreter still
   on numpy 1 the installs are pinned (``rasterio<1.4``, ``fiona<1.10``,
   ``numpy<2``) so pip does not put a numpy 2 into ``libs/`` underneath numba.

3. **pip's real error was swallowed.** Output is captured and reported now, so
   a failure shows why rather than "returned non-zero exit status 1".

4. **PEP 668.** Debian, Ubuntu and Fedora mark the system interpreter
   "externally managed". A failure carrying that phrase is retried with
   ``--break-system-packages``, which is what those distributions document.

Policy: auto-install on first use, with the exact manual command in the log if
it fails (locked-down machines). Copyright (C) 2026 Igor Pawelec. Licence: GPLv3.
"""
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys

from .vendor_loader import LIBS_DIR

# Everything an algorithm needs to be importable. pygeoadaptels and pygeopalette
# are pure Python and ride along in vendor/ (see vendor_loader), so they are
# checked here but never handed to pip.
REQUIRED = ["pygeoadaptels", "pygeopalette", "numba", "rasterio", "fiona"]

# Only the packages that carry compiled binaries, which cannot be vendored.
#
# **numba is listed explicitly.** It used to arrive as a dependency of
# pygeoadaptels when that was pip-installed; now that pygeoadaptels is
# vendored, pip is never told about it and would not install it otherwise.
BINARY_DEPS = ["numba", "rasterio", "fiona"]

# What pip drags into libs/ beside them although the interpreter has its own.
# ``pip --target`` does not look at the interpreter's packages, so rasterio and
# numba bring a numpy with them; libs/ is appended to ``sys.path``, so QGIS's
# numpy is the one imported and the copy is 54 MB that nothing ever reads.
SHADOWED = ("numpy",)


def _numpy_major():
    try:
        import numpy
        return int(numpy.__version__.split(".")[0])
    except Exception:
        return 1


def _install_specs(modules=None):
    """pip specs for the binary dependencies, pinned to the interpreter's numpy line.

    Current rasterio/fiona pull in numpy 2.x, which breaks numba and scipy on
    the numpy 1.26 that QGIS 3 ships -- installing without the pins would put
    a numpy 2 into libs/ and leave numba dead. On an interpreter already on
    numpy 2 (QGIS 4, some Linux distributions) the pins would be wrong, hence
    the check rather than a hardcoded list.
    """
    wanted = [m for m in (modules or BINARY_DEPS) if m in BINARY_DEPS]
    if _numpy_major() < 2:
        pins = {"rasterio": "rasterio<1.4", "fiona": "fiona<1.10"}
        specs = [pins.get(m, m) for m in wanted]
        if specs:
            specs.append("numpy<2")
        return specs
    return list(wanted)


INSTALL_SPECS = _install_specs()


def _qgis_python():
    """The real QGIS interpreter, not the ``bin/`` launcher.

    Inside QGIS ``sys.prefix`` is the interpreter home. The launcher at
    ``sys.executable`` fails to run as a subprocess on Windows, so it is only
    the last resort.
    """
    names = ("python.exe", "python3.exe", "python3", "python")
    for base in (sys.prefix, os.path.join(sys.prefix, "bin")):
        for name in names:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    return sys.executable


def missing_packages():
    out = []
    for m in REQUIRED:
        try:
            if importlib.util.find_spec(m) is None:
                out.append(m)
        except (ImportError, ValueError):
            out.append(m)
    return out


def _same_or_under(path, root):
    """True when ``path`` is ``root`` or lies inside it. Real paths are compared:
    QGIS on Windows reports its prefix in the 8.3 short form (C:\\PROGRA~1\\...)."""
    try:
        p = os.path.normcase(os.path.realpath(path))
        r = os.path.normcase(os.path.realpath(root))
    except OSError:
        return False
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def _interpreter_roots():
    return [r for r in {sys.prefix, sys.base_prefix, sys.exec_prefix} if r]


def _origin(name):
    """The file ``name`` is imported from, or would be, without importing it; None when not found."""
    loaded = getattr(sys.modules.get(name), "__file__", None)
    if loaded and os.path.isfile(loaded):
        return loaded                             # find_spec trusts a loaded module's __spec__, which may be absent
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    origin = getattr(spec, "origin", None)
    return origin if origin and os.path.isfile(origin) else None


def prune_shadowed(feedback=None, names=SHADOWED):
    """Remove from libs/ the copies of packages the interpreter provides itself.

    A copy goes only when Python would import the package from the
    interpreter's own folders, which come before libs/ on ``sys.path``: the
    copy is then never the one in use, and removing it changes nothing but
    the size of the folder. Returns the names removed. The same routine as in
    the GeoSnag plugin.
    """
    removed = []
    if not os.path.isdir(LIBS_DIR):
        return removed
    importlib.invalidate_caches()                 # a folder pip filled a moment ago is otherwise not seen
    for name in names:
        if not os.path.isdir(os.path.join(LIBS_DIR, name)):
            continue
        origin = _origin(name)
        if origin is None or _same_or_under(origin, LIBS_DIR):
            continue                              # the copy in libs/ is the one in use
        if not any(_same_or_under(origin, root) for root in _interpreter_roots()):
            continue                              # provided by something that can go away again
        for entry in os.listdir(LIBS_DIR):
            low = entry.lower()
            if low in (name, name + ".libs") or (low.startswith(name + "-") and low.endswith(".dist-info")):
                shutil.rmtree(os.path.join(LIBS_DIR, entry), ignore_errors=True)
        removed.append(name)
    if removed and feedback is not None:
        feedback.pushInfo("Removed from the plugin's libs folder the copies of " + ", ".join(removed)
                          + " that pip put there: QGIS has its own and those are the ones in use.")
    return removed


def _pip(py, specs, extra=()):
    os.makedirs(LIBS_DIR, exist_ok=True)
    cmd = [py, "-m", "pip", "install", "--target", LIBS_DIR, "--upgrade", *extra, *specs]
    return subprocess.run(cmd, capture_output=True, text=True)


def ensure_dependencies(feedback=None, auto_install=True):
    """Return ``(ok, missing)``. Never raises; never hard-crashes the plugin.

    Order matters: the vendored pure-Python packages and the plugin's own
    libs/ are put on the path *first*, so pip is only ever asked for the
    binary dependencies that are genuinely missing.
    """
    from . import vendor_loader
    vendor_loader.activate(feedback)

    missing = missing_packages()
    if not missing:
        prune_shadowed(feedback)                  # what an earlier version of the plugin left there
        return True, []
    if not auto_install:
        return False, missing

    # Only the binary ones are pip's business; if a vendored package is
    # missing the vendor directory is broken and pip cannot help.
    to_install = [m for m in missing if m in BINARY_DEPS]
    if not to_install:
        if feedback is not None:
            feedback.reportError(
                "These are bundled with the plugin but did not import: "
                + ", ".join(missing)
                + ". The plugin's vendor/ folder looks incomplete -- "
                  "reinstall the plugin zip.")
        return False, missing

    py = _qgis_python()
    specs = _install_specs(to_install)
    if feedback is not None:
        feedback.pushInfo(f"Installing {', '.join(specs)} into {LIBS_DIR} using {py} (first run only)")
    try:
        r = _pip(py, specs)
        # PEP 668: Debian/Ubuntu/Fedora mark the system interpreter
        # "externally managed". The override is exactly what those
        # distributions document for this case.
        if r.returncode != 0 and "externally-managed-environment" in (
                (r.stderr or "") + (r.stdout or "")):
            if feedback is not None:
                feedback.pushInfo(
                    "Interpreter is externally managed; retrying with "
                    "--break-system-packages.")
            r = _pip(py, specs, extra=("--break-system-packages",))
    except Exception as e:  # pragma: no cover - environment dependent
        if feedback is not None:
            feedback.reportError(f"Could not launch pip: {e}\n{manual_hint()}")
        return False, missing

    if r.returncode != 0:
        if feedback is not None:
            tail = (r.stdout or "")[-2000:] + "\n" + (r.stderr or "")[-2000:]
            feedback.reportError(
                "pip failed:\n" + tail.strip()
                + "\n\nInstall by hand and restart QGIS:\n  "
                + manual_hint())
        return False, missing

    importlib.invalidate_caches()
    prune_shadowed(feedback)
    vendor_loader.activate()
    still = missing_packages()
    if still and feedback is not None:
        feedback.reportError(
            "pip reported success but these are still not importable: "
            + ", ".join(still) + ".\nRestart QGIS, then try again.")
    return (not still), still


def manual_hint():
    """The exact one-line install command, for the log.

    Only the binary dependencies -- the two pure-Python packages ship inside
    the plugin, so an operator never has to fetch them.
    """
    py = _qgis_python()
    return f'"{py}" -m pip install --target "{LIBS_DIR}" --upgrade ' + " ".join(_install_specs())
