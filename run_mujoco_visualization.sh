#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:/home/yhzhu/AI/NVlabs/curobo:${PYTHONPATH:-}"
if [[ "${TERM:-}" == "" || "${TERM:-}" == "dumb" ]]; then
  export TERM="xterm"
fi
MODE="${1:-video}"
shift || true
/home/yhzhu/isaaclab/isaaclab.sh -p "${ROOT_DIR}/scripts/visualize_mujoco_motion.py" --mode "${MODE}" "$@"
