#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UR_TYPE="${UR_TYPE:-ur7e}"
ROBOT_IP="${ROBOT_IP:-192.168.1.103}"

if pgrep -fa "ur_robot_driver.*ur_control.launch.py|ur_control.launch.py.*ur_type:=${UR_TYPE}" >/dev/null; then
  echo "A UR driver launch process already appears to be running."
  echo "Stop the old driver first, then run this script again."
  pgrep -fa "ur_robot_driver.*ur_control.launch.py|ur_control.launch.py.*ur_type:=${UR_TYPE}" || true
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
source "${WS_DIR}/install/setup.bash"
set -u

exec ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:="${UR_TYPE}" \
  robot_ip:="${ROBOT_IP}" \
  launch_rviz:=false \
  "$@"
