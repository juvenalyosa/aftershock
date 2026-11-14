#!/usr/bin/env bash
# Install what PERT needs on a Google Colab runtime (Ubuntu, NVIDIA GPU, no conda):
# OpenMM with the CUDA platform (pip wheels) and MODELLER (Debian package).
#
# Usage:  KEY_MODELLER=<licence key> bash scripts/colab_install.sh
#
# Environment overrides: PERT_PYTHON (interpreter to install into, default
# python3), OPENMM_VERSION, OPENMM_CUDA_EXTRA (cuda12 | cuda13), MODELLER_VERSION.
# Safe to re-run: it only installs what is missing and rewrites the licence key.
set -euo pipefail

PY="${PERT_PYTHON:-python3}"
OPENMM_VERSION="${OPENMM_VERSION:-8.6.1}"
OPENMM_CUDA_EXTRA="${OPENMM_CUDA_EXTRA:-cuda12}"
MODELLER_VERSION="${MODELLER_VERSION:-10.8}"
MODELLER_HOME="/usr/lib/modeller${MODELLER_VERSION}"

if [[ -z "${KEY_MODELLER:-}" ]]; then
  echo "KEY_MODELLER is not set. MODELLER needs a licence key (free for academic use):" >&2
  echo "https://salilab.org/modeller/registration.html" >&2
  exit 1
fi

echo "[1/3] OpenMM ${OPENMM_VERSION} (${OPENMM_CUDA_EXTRA})..."
if command -v nvidia-smi >/dev/null 2>&1; then
  "${PY}" -m pip install -q "openmm[${OPENMM_CUDA_EXTRA}]==${OPENMM_VERSION}"
else
  echo "  no NVIDIA GPU found: installing OpenMM without CUDA (simulations will be slow)." >&2
  "${PY}" -m pip install -q "openmm==${OPENMM_VERSION}"
fi
"${PY}" -m pip install -q numpy scipy scikit-learn matplotlib

echo "[2/3] MODELLER ${MODELLER_VERSION}..."
if [[ ! -d "${MODELLER_HOME}" ]]; then
  DEB="$(mktemp -d)/modeller.deb"
  wget -q "https://salilab.org/modeller/${MODELLER_VERSION}/modeller_${MODELLER_VERSION}-1_amd64.deb" -O "${DEB}"
  dpkg -i "${DEB}" >/dev/null
fi
# The key is written here rather than left to the package script, so that
# re-running with a corrected key fixes an installation made with a wrong one.
printf "install_dir = r'%s'\nlicense = r'%s'\n" "${MODELLER_HOME}" "${KEY_MODELLER}" \
  > "${MODELLER_HOME}/modlib/modeller/config.py"
# The package links itself into /usr/lib/pythonX.Y/dist-packages for the Python
# versions it knows; add the links for a newer one (the extension module uses
# the stable ABI) and make sure the chosen interpreter looks there.
PYVER="$("${PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
DIST="/usr/lib/python${PYVER}/dist-packages"
mkdir -p "${DIST}"
[[ -e "${DIST}/modeller" ]] || ln -s "${MODELLER_HOME}/modlib/modeller" "${DIST}/modeller"
[[ -e "${DIST}/_modeller.so" ]] || ln -s "${MODELLER_HOME}/lib/x86_64-intel8/python3.3/_modeller.so" "${DIST}/_modeller.so"
if ! "${PY}" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('modeller') else 1)"; then
  SITE="$("${PY}" -c 'import site; print(site.getsitepackages()[0])')"
  echo "${DIST}" > "${SITE}/modeller_dist_packages.pth"
fi

echo "[3/3] Checking the installation..."
if ! "${PY}" -c "import modeller" >/dev/null 2>&1; then
  echo "MODELLER does not start. The usual cause is a wrong licence key; the message was:" >&2
  "${PY}" -c "import modeller" 2>&1 | tail -n 5 >&2 || true
  exit 1
fi
"${PY}" - <<'PYEOF'
import openmm
import sklearn
import numpy

names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
print(f"  OpenMM {openmm.__version__}, platforms: {', '.join(names)}")
print(f"  NumPy {numpy.__version__}, scikit-learn {sklearn.__version__}, MODELLER ok")
if "CUDA" not in names:
    print("  WARNING: the CUDA platform is not available; simulations will run on a slower platform.")
    for error in openmm.Platform.getPluginLoadFailures():
        print("   ", error)
PYEOF
echo "Installation ready."
