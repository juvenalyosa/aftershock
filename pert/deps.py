"""
Optional heavy dependencies, imported once and shared.

OpenMM, scikit-learn and SciPy are only needed by some stages.  Importing them
here (and tolerating their absence) lets the light-weight parts of the package
(sequence mutation, alignment, benchmark normalisation) work in a minimal
environment.  Stages that do need them call ``ensure_openmm()`` /
``ensure_sklearn()`` to fail early with an actionable message.

MODELLER is deliberately *not* imported here: it is loaded lazily by
``pert.structure.prepare.prepare_structures`` because it needs a licence key.
"""

from __future__ import annotations

try:
    from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, Vec3, XmlSerializer, app, unit
except Exception:  # pragma: no cover - validated at runtime by ensure_openmm()
    XmlSerializer = None
    MonteCarloBarostat = None
    Platform = None
    Vec3 = None
    unit = None
    LangevinMiddleIntegrator = None
    app = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - validated at runtime by ensure_sklearn()
    LogisticRegression = None
    GaussianMixture = None

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


def ensure_openmm() -> None:
    """Raise a helpful ImportError when OpenMM is not installed."""
    if app is None or unit is None:
        raise ImportError(
            "OpenMM is required but not available. Install OpenMM 8.x "
            "(e.g., `conda install -c conda-forge openmm`)."
        )


def ensure_sklearn() -> None:
    """Raise a helpful ImportError when scikit-learn is not installed."""
    if GaussianMixture is None or LogisticRegression is None:
        raise ImportError(
            "scikit-learn is required for GMM/JS components. "
            "Install with `pip install scikit-learn`."
        )
