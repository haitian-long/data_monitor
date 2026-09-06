#!/bin/bash
set -euo pipefail

# ===== user config (edit these) =====
DATASET_ROOT="${HOME}/disk/dataset/xiaohongshan/hesai_jt128"
SEQ="xiaohongshan_jt128_2026-07-08-17-23-41"
MAIN_ALGO="fast_lio2"
OURS_ALGO="ours"
# Optional legend overrides for remaining algorithms only.
# Unlisted names stay as the TUM suffix (e.g. fast_lio2). Avoid commas.
declare -A LEGEND_NAMES=(
  # [fast_lio2]="FAST-LIO2"
  # [kiss_icp]="KISS-ICP"
)
# ====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${WS_ROOT}/devel/setup.bash"

legend_name_for() {
  local algo="$1"
  if [[ -n "${LEGEND_NAMES[$algo]+x}" ]]; then
    printf '%s' "${LEGEND_NAMES[$algo]}"
  else
    printf '%s' "${algo}"
  fi
}

MAPS_DIR="${DATASET_ROOT}/maps"
ODOM_DIR="${DATASET_ROOT}/result_odoms"
PCD_PATH="${MAPS_DIR}/${SEQ}_${MAIN_ALGO}.pcd"

if [[ ! -f "${PCD_PATH}" ]]; then
  echo "PCD not found: ${PCD_PATH}" >&2
  exit 1
fi

shopt -s nullglob
tum_files=("${ODOM_DIR}/${SEQ}_"*.tum)
if [[ ${#tum_files[@]} -eq 0 ]]; then
  echo "No TUM trajectories found for ${SEQ} in ${ODOM_DIR}" >&2
  exit 1
fi

benchmark_path=""
ours_path=""
other_paths=()
other_names=()

for tum_path in "${tum_files[@]}"; do
  stem="$(basename "${tum_path}" .tum)"
  algo="${stem#${SEQ}_}"
  if [[ "${algo}" == "${MAIN_ALGO}" ]]; then
    benchmark_path="${tum_path}"
  elif [[ -n "${OURS_ALGO}" && "${algo}" == "${OURS_ALGO}" ]]; then
    ours_path="${tum_path}"
  else
    other_paths+=("${tum_path}")
    other_names+=("$(legend_name_for "${algo}")")
  fi
done

if [[ -z "${benchmark_path}" ]]; then
  echo "Benchmark TUM not found: ${ODOM_DIR}/${SEQ}_${MAIN_ALGO}.tum" >&2
  exit 1
fi
if [[ -n "${OURS_ALGO}" && -z "${ours_path}" ]]; then
  echo "Ours TUM not found: ${ODOM_DIR}/${SEQ}_${OURS_ALGO}.tum" >&2
  exit 1
fi

other_paths_arg=""
other_names_arg=""
if [[ ${#other_paths[@]} -gt 0 ]]; then
  IFS=","
  other_paths_arg="${other_paths[*]}"
  other_names_arg="${other_names[*]}"
  unset IFS
fi

echo "Sequence:  ${SEQ}"
echo "Map:       ${PCD_PATH}"
echo "Benchmark: ${MAIN_ALGO} -> ${benchmark_path}"
if [[ -n "${ours_path}" ]]; then
  echo "Ours:      ${OURS_ALGO} -> ${ours_path}"
fi
if [[ ${#other_names[@]} -gt 0 ]]; then
  echo "Others:    ${other_names[*]}"
fi

launch_args=(
  "pcd_path:=${PCD_PATH}"
  "title:=${SEQ}"
  "benchmark_trajectory_path:=${benchmark_path}"
)
if [[ -n "${ours_path}" ]]; then
  launch_args+=("ours_trajectory_path:=${ours_path}")
fi
if [[ -n "${other_paths_arg}" ]]; then
  launch_args+=(
    "other_trajectory_paths:=${other_paths_arg}"
    "other_trajectory_names:=${other_names_arg}"
  )
fi

roslaunch data_monitor traj_monitor.launch "${launch_args[@]}"
