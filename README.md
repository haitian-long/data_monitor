# odometry_monitor

ROS Noetic 下的 odometry 实时监控工具。节点订阅 `nav_msgs/Odometry`，打开 Matplotlib 窗口，将 odometry 的 `x`、`y` 坐标绘制到二维直角坐标系中，并实时显示相对当前起点的 pose 和 `x`、`y`、`z` 相对位置波动范围。

## 功能

- 订阅 odometry 数据，默认话题为 `/odom`
- 实时绘制 `x-y` 运动轨迹
- 坐标系范围会根据当前已记录的 `x`、`y` 范围动态扩展
- 显示当前相对 pose
  - `position(m): [x, y, z]`
  - `rotation(rpy degree): [roll, pitch, yaw]`
- 显示 `x`、`y`、`z` 相对位置波动范围
- 提供 `Reset` 按钮
  - 点击后以当前 odometry 位置作为新的起点
  - 清空旧轨迹并重新绘制
  - 重新统计 `x`、`y`、`z` 波动范围

## 依赖

```bash
sudo apt install ros-noetic-nav-msgs python3-matplotlib
```

## 编译

在工作空间根目录执行：

```bash
catkin_make
source devel/setup.bash
```

## 运行

使用默认 `/odom` 话题：

```bash
roslaunch odometry_monitor odometry_monitor.launch
```

指定其他 odometry 话题：

```bash
roslaunch odometry_monitor odometry_monitor.launch odom_topic:=/your/odom/topic
```

也可以直接运行节点：

```bash
rosrun odometry_monitor odometry_monitor_node.py _odom_topic:=/odom
```

## 说明

窗口中的 pose 和波动范围均为相对值。首次收到 odometry 时，当前 position 和 orientation 会被自动作为起点；点击 `Reset` 后，当前 odometry pose 会成为新的起点。
