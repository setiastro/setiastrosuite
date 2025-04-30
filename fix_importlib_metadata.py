# fix_importlib_metadata.py
import sys

if getattr(sys, 'frozen', False):
    # Try stdlib first, then backport
    try:
        import importlib.metadata as _md
    except ImportError:
        import importlib_metadata as _md

    # Keep originals
    _orig_version      = _md.version
    _orig_distribution = _md.distribution

    # Return a dummy version string so .split() works
    def safe_version(pkg, *args, **kwargs):
        try:
            return _orig_version(pkg, *args, **kwargs)
        except Exception:
            return "0.0.0"

    # Return a minimal object for distribution() if needed
    class DummyDist:
        version = "0.0.0"
        metadata = {}

    def safe_distribution(pkg, *args, **kwargs):
        try:
            return _orig_distribution(pkg, *args, **kwargs)
        except Exception:
            return DummyDist()

    # Patch
    _md.version      = safe_version
    _md.distribution = safe_distribution
