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

`pcd_monitor` 读取一个指定的 `.pcd` 文件并生成鸟瞰图：

- 横轴为 X，纵轴为 Y，坐标单位为米
- 每个 XY 像素的数值为落入该像素的所有点中最大的 Z
- 启动时一次性读取完整 PCD，然后执行一次全局三维体素下采样
- 每个体素保留 Z 最高点
- 启动时缓存下采样后的 `float32` XYZ，后续重建不再读取 PCD
- 启动时按缓存点云 Z 值的 `1%～99%` 百分位固定色标范围
- 初始 BEV 为 `2560 × 1440`、16:9，窗口以 `1280 × 720` 启动并可手动调整大小
- 图像及窗口标题由 `title` 参数指定，运行统计只输出到终端
- 支持 `ascii`、`binary`、`binary_compressed` PCD
- 鼠标滚轮以光标为中心缩放
- 鼠标左键拖拽平移
- 拖拽、连续滚轮缩放和窗口尺寸调整期间使用最高 `1280 × 720` 的最大 Z
  预览，停止交互后自动恢复当前完整 BEV
- 缩放范围限制在有效点云尺度内，避免无限放大或缩小
- 点击 `Resolution` 可选择 `2560*1440`、`1280*720` 或 `640*360`
- 点击 `Rebuild BEV`，根据当前视野和已选分辨率重新生成 BEV
- 点击 `Save Image`，将当前视野保存为只包含 BEV、标题、坐标轴和色条的 PNG
- 双击、按 `R` 或点击 `Reset view` 恢复完整视图
- 也可以使用 Matplotlib 窗口自带的缩放和平移工具

运行：

```bash
roslaunch data_monitor pcd_monitor.launch \
  pcd_path:=/absolute/path/to/cloud.pcd \
  voxel_size:=0.10 \
  title:="My PCD Map"
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pcd_path` | 无，必填 | 单个 PCD 文件路径，不支持目录 |
| `voxel_size` | `0.10` | 三维体素边长，单位为米，必须大于 0 |
| `title` | `PCD BEV Monitor` | 图像和窗口标题 |
| `colormap` | `turbo` | Matplotlib 色图名称 |

节点启动时先将完整 PCD 的 XYZ 读取到内存，再按 `voxel_size` 执行一次全局下采样，之后只缓存 `float32` XYZ 下采样点云。初次启动时，节点使用缓存点云的 XY 范围建立 16:9 视野，并自动计算米/像素。放大或平移后，可以先选择 BEV 分辨率，再点击 `Rebuild BEV`，直接从内存缓存筛选当前视野并重新栅格化，不再读取 PCD 或重复体素滤波；如果当前视野不是 16:9，节点会扩展其中一个方向，以保持 X/Y 分辨率一致。`Reset view` 会立即恢复初始完整 BEV。

`voxel_size` 越大，保留的点越少，读取和重建所需内存越低，但点云细节也会相应减少。

色图的 `vmin/vmax` 在启动时由缓存点云 Z 值的第 1 和第 99 百分位确定，之后全景、局部重建和复位始终使用相同范围。低于或高于该范围的少量极端值会显示为色图两端颜色。

原始/下采样点数、当前视野点数、BEV 分辨率、体素尺寸、XYZ 范围和固定色标范围通过 ROS 日志输出，不会附加到图像标题。

保存图片时，PNG 会写入 PCD 文件所在目录。文件名使用标题参数，例如
`My PCD Map.png`。
如果文件已经存在，则依次使用 `My PCD Map1.png`、`My PCD Map2.png`。
导出的图片不会包含操作按钮和底部操作说明。

节点会针对 Tk、Qt 和 GTK Matplotlib 后端显式启用窗口调整大小，并解除 GUI 画布可能继承的固定尺寸限制；窗口大小变化不会改变缓存点云或 BEV 栅格分辨率。
