#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:/home/yhzhu/AI/NVlabs/curobo:${PYTHONPATH:-}"
if [[ "${TERM:-}" == "" || "${TERM:-}" == "dumb" ]]; then
  export TERM="xterm"
fi
ISAACLAB_PY="/home/yhzhu/isaaclab/isaaclab.sh -p"
OUT_DIR="${ROOT_DIR}/outputs/quickmove_demo"

mkdir -p "${OUT_DIR}"

printf '
== Environment ==
'
${ISAACLAB_PY} "${ROOT_DIR}/scripts/check_env.py"

printf '
== Baseline vs QuickMove ==
'
${ISAACLAB_PY} "${ROOT_DIR}/examples/compare_cspace_profiles.py" --json | tee "${OUT_DIR}/comparison.json"

printf '
== Visualize QuickMove trajectory ==
'
${ISAACLAB_PY} "${ROOT_DIR}/scripts/visualize_cspace_motion.py" --output-dir "${OUT_DIR}"

printf '
== Render MuJoCo video ==
'
${ISAACLAB_PY} "${ROOT_DIR}/scripts/visualize_mujoco_motion.py" --mode video --no-warmup --output "${OUT_DIR}/quickmove_mujoco.mp4"

printf '
Artifacts written to: %s
' "${OUT_DIR}"
printf 'Open %s/quickmove_motion.html in a browser to inspect the motion.
' "${OUT_DIR}"
