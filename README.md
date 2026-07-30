# data_monitor

ROS Noetic 下的数据可视化工具，包含 odometry 实时监控和 PCD 点云 BEV 查看两个节点。

## Odom Monitor

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
- 提供 `Pause` 按钮
  - 点击后暂停追加轨迹点，图和波动范围保持不变
  - 暂停后可使用 Matplotlib 工具栏缩放、平移查看轨迹
  - 再次点击 `Resume` 后继续绘图

## 依赖

```bash
sudo apt install ros-noetic-nav-msgs python3-matplotlib python3-numpy
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
roslaunch data_monitor odom_monitor.launch
```

指定其他 odometry 话题：

```bash
roslaunch data_monitor odom_monitor.launch odom_topic:=/your/odom/topic
```

也可以直接运行节点：

```bash
rosrun data_monitor odom_monitor_node.py _odom_topic:=/odom
```

## 说明

窗口中的 pose 和波动范围均为相对值。首次收到 odometry 时，当前 position 和 orientation 会被自动作为起点；点击 `Reset` 后，当前 odometry pose 会成为新的起点。

## PCD BEV 查看

`pcd_monitor` 会读取指定目录内的所有 `.pcd` 文件并合并为一幅鸟瞰图：

- 横轴为 X，纵轴为 Y，坐标单位为米
- 每个 XY 像素的数值为落入该像素的所有点中最大的 Z
- 支持 `ascii`、`binary`、`binary_compressed` PCD
- 鼠标滚轮以光标为中心缩放
- 鼠标左键拖拽平移
- 双击、按 `R` 或点击 `Reset view` 恢复完整视图
- 也可以使用 Matplotlib 窗口自带的缩放和平移工具

运行：

```bash
roslaunch data_monitor pcd_monitor.launch \
  pcd_dir:=/absolute/path/to/pcd_folder \
  resolution:=0.10
```

递归读取子目录：

```bash
roslaunch data_monitor pcd_monitor.launch \
  pcd_dir:=/absolute/path/to/pcd_folder \
  recursive:=true
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pcd_dir` | 空 | PCD 文件目录，必须指定 |
| `resolution` | `0.10` | BEV 分辨率，单位 m/pixel |
| `recursive` | `false` | 是否递归搜索子目录 |
| `colormap` | `turbo` | Matplotlib 色图名称 |
| `max_grid_cells` | `16000000` | 最大 BEV 像素数，防止误用过多内存 |

如果点云 XY 范围很大且分辨率过细，节点会提示建议的 `resolution`，以避免创建过大的 BEV 数组。
