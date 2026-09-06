#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${WS_ROOT}/devel/setup.bash"

roslaunch data_monitor pcd_monitor.launch \
  pcd_path:=~/disk/dataset/shangqi/hesai_xt32/maps/shangqi_xt32_2026-04-29-13-00-12_dlio.pcd \
  title:="shangqi_xt32_2026-04-29-13-00-12_dlio" \
  trajectory_path:=~/disk/dataset/shangqi/hesai_xt32/result_odoms/shangqi_xt32_2026-04-29-13-00-12_dlio.tum \
  voxel_size:=0.30
