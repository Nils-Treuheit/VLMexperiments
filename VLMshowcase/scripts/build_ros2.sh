#!/bin/bash
# Build the ROS2 webcam demo workspace
set -e

source /opt/ros/humble/setup.bash
WS="/mnt/HDD1/Project_Code/VLMshowcase/ros2_ws"

cd "$WS"
colcon build --packages-select vlm_webcam_demo

echo ""
echo "ROS2 package built. To run:"
echo "  source $WS/install/setup.bash"
echo "  ros2 run vlm_webcam_demo vlm_webcam_node"
