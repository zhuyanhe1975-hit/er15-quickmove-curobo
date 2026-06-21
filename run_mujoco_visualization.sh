#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}/src:/home/yhzhu/AI/NVlabs/curobo:${PYTHONPATH:-}"
if [[ "${TERM:-}" == "" || "${TERM:-}" == "dumb" ]]; then
  export TERM="xterm"
fi
MODE="video"
if [[ "${1:-}" == "video" || "${1:-}" == "viewer" ]]; then
  MODE="$1"
  shift
elif [[ "${1:-}" == "--mode" ]]; then
  shift
  MODE="${1:?--mode requires video or viewer}"
  shift
fi

/home/yhzhu/isaaclab/isaaclab.sh -p "${ROOT_DIR}/scripts/visualize_mujoco_motion.py" --mode "${MODE}" "$@"
