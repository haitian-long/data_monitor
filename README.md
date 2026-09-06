# data_monitor

ROS Noetic 下的数据可视化工具，包含三个节点：

- `odom_monitor`：odometry 实时轨迹监控
- `pcd_monitor`：PCD 点云彩色 BEV 查看
- `traj_monitor`：灰色 BEV 上叠加多条轨迹对比

## 依赖

```bash
sudo apt install ros-noetic-nav-msgs python3-matplotlib python3-numpy python3-pil
```

## 编译

在工作空间根目录执行：

```bash
catkin_make
source devel/setup.bash
```

## Odom Monitor

### 功能

- 订阅 odometry 数据，默认话题为 `/robot/dlio/odom_node/odom`
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

### 运行

使用 launch 默认话题：

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

窗口中的 pose 和波动范围均为相对值。首次收到 odometry 时，当前 position 和 orientation 会被自动作为起点；点击 `Reset` 后，当前 odometry pose 会成为新的起点。

## PCD BEV 查看

`pcd_monitor` 读取一个指定的 `.pcd` 文件并生成按高度着色的鸟瞰图：

- 横轴为 X，纵轴为 Y，坐标单位为米
- 每个 XY 像素的数值为落入该像素的所有点中最大的 Z
- 空像素背景为白色，点云使用 `turbo` 色图按固定 Z 范围着色
- 启动时一次性读取完整 PCD，然后执行一次全局三维体素下采样
- 每个体素保留 Z 最高点
- 启动时缓存下采样后的 `float32` XYZ，后续重建不再读取 PCD
- 启动时按缓存点云 Z 值的 `1%～99%` 百分位固定色标范围
- 可选读取一条 TUM 格式轨迹，按 X-Y 以红色粗实线叠加到 BEV
- 初始 BEV 为 `2560 × 1440`、16:9，窗口以 `1280 × 720` 启动并可手动调整大小
- 图像及窗口标题由 `title` 参数指定，运行统计只输出到终端
- 支持 `ascii`、`binary`、`binary_compressed` PCD
- 鼠标滚轮以光标为中心缩放
- 点击 `Zoom` 可显示横向缩放滑条，并以当前视野中心连续缩放；点击滑条外的位置即可关闭
- 鼠标左键拖拽平移
- 拖拽、连续滚轮缩放和窗口尺寸调整期间使用最高 `1280 × 720` 的最大 Z 预览，停止交互后自动恢复当前完整 BEV
- 缩放范围限制在有效点云尺度内，避免无限放大或缩小
- 点击 `Resolution` 可选择 `2560*1440`、`1280*720` 或 `640*360`；点击选项外的位置即可关闭
- 点击 `Rebuild BEV`，根据当前视野和已选分辨率重新生成 BEV
- 点击 `Save PNG` 或 `Save TIFF`，将当前视野保存为只包含 BEV、标题、坐标轴、色条和轨迹的图片
- 双击、按 `R` 或点击 `Reset view` 恢复完整视图

运行：

```bash
roslaunch data_monitor pcd_monitor.launch \
  pcd_path:=/absolute/path/to/cloud.pcd \
  trajectory_path:=/absolute/path/to/trajectory.tum \
  voxel_size:=0.10 \
  title:="My PCD Map"
```

也可以使用包内脚本（会自动 source 工作空间）：

```bash
src/data_monitor/scripts/bev.sh
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pcd_path` | 无，必填 | 单个 PCD 文件路径，不支持目录 |
| `trajectory_path` | 空 | 可选 TUM 轨迹文件路径；为空或格式错误时不加载 |
| `voxel_size` | launch 为 `0.30` | 三维体素边长，单位为米，必须大于 0 |
| `title` | `PCD BEV Monitor` | 图像和窗口标题 |
| `colormap` | `turbo` | Matplotlib 色图名称 |

节点启动时先将完整 PCD 的 XYZ 读取到内存，再按 `voxel_size` 执行一次全局下采样，之后只缓存 `float32` XYZ 下采样点云。初次启动时，节点使用缓存点云的 XY 范围建立 16:9 视野，并自动计算米/像素。放大或平移后，可以先选择 BEV 分辨率，再点击 `Rebuild BEV`，直接从内存缓存筛选当前视野并重新栅格化，不再读取 PCD 或重复体素滤波；如果当前视野不是 16:9，节点会扩展其中一个方向，以保持 X/Y 分辨率一致。`Reset view` 会立即恢复初始完整 BEV。

`voxel_size` 越大，保留的点越少，读取和重建所需内存越低，但点云细节也会相应减少。

色图的 `vmin/vmax` 在启动时由缓存点云 Z 值的第 1 和第 99 百分位确定，之后全景、局部重建和复位始终使用相同范围。低于或高于该范围的少量极端值会显示为色图两端颜色。

原始/下采样点数、当前视野点数、BEV 分辨率、体素尺寸、XYZ 范围和固定色标范围通过 ROS 日志输出，不会附加到图像标题。

保存图片时，PNG 或 TIFF 会写入 PCD 文件所在目录。文件名使用读取的 PCD 文件名，例如读取 `merged_filtered.pcd` 时保存为 `merged_filtered.png` 或 `merged_filtered.tiff`。如果文件已经存在，则依次使用 `_1`、`_2` 后缀。导出的图片会包含已加载的轨迹，不会包含操作按钮。TIFF 使用 LZW 压缩。

节点会针对 Tk、Qt 和 GTK Matplotlib 后端显式启用窗口调整大小，并解除 GUI 画布可能继承的固定尺寸限制；窗口大小变化不会改变缓存点云或 BEV 栅格分辨率。

## Trajectory BEV 对比

`traj_monitor` 与 `pcd_monitor` 共用同一套 PCD 读取、体素下采样和 BEV 交互，但只显示灰色点云，并叠加多条轨迹做对比：

- 背景为近白色，点云为中灰色，不按高度着色，也没有色条
- `benchmark` 为红色实线，`ours` 为红色虚线，其余轨迹为不同颜色虚线
- 其它轨迹会按 benchmark 起点位置和起步约 10 m 的行驶方向做 SE(2) 对齐；TUM 文件开头 `t=0` 的占位位姿会跳过
- 对齐后的起点用黄色小三角标出，图例中标注为 `start`
- 图例放在图内右上角，随窗口缩放
- 缩放、平移、分辨率、`Filter`、`Color`、`Rebuild BEV`、`Save PNG` / `Save TIFF` 与 `pcd_monitor` 相同；`Filter` 可勾选要显示的轨迹，`Color` 可改每条轨迹的颜色（色值或调色盘）

运行：

```bash
roslaunch data_monitor traj_monitor.launch \
  pcd_path:=/absolute/path/to/cloud.pcd \
  benchmark_trajectory_path:=/absolute/path/to/benchmark.tum \
  ours_trajectory_path:=/absolute/path/to/ours.tum \
  other_trajectory_paths:=/abs/a.tum,/abs/b.tum \
  other_trajectory_names:="FAST-LIO2,KISS-ICP" \
  voxel_size:=0.30 \
  title:="Trajectory Comparison"
```

`other_trajectory_paths` 和 `other_trajectory_names` 用逗号分隔，按顺序对应。未给出名称时，图例使用 TUM 文件名（去掉扩展名）。图例顺序为 `benchmark`、`ours`，再是其它算法。`benchmark` / `ours` 的图例名固定，不需要再传 name。

也可以使用包内脚本，按序列一键启动：

```bash
src/data_monitor/scripts/traj.sh
```

脚本开头可改 `DATASET_ROOT`、`SEQ`、`MAIN_ALGO`、`OURS_ALGO` 和可选的 `LEGEND_NAMES`。脚本会把 `${SEQ}_${MAIN_ALGO}.tum` 填到 `benchmark_trajectory_path`，把 `${SEQ}_${OURS_ALGO}.tum` 填到 `ours_trajectory_path`，其余 `.tum` 填到 `other_*`。底图使用 `${SEQ}_${MAIN_ALGO}.pcd`。`LEGEND_NAMES` 只改写其余算法的图例名。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pcd_path` | 无，必填 | 单个 PCD 文件路径，不支持目录 |
| `benchmark_trajectory_path` | 空 | benchmark 的 TUM 路径，红色实线，图例固定为 `benchmark` |
| `ours_trajectory_path` | 空 | ours 的 TUM 路径，红色虚线，图例固定为 `ours` |
| `other_trajectory_paths` | 空 | 其余 TUM 路径，逗号分隔，彩色虚线 |
| `other_trajectory_names` | 空 | 其余轨迹图例名称，逗号分隔，与路径一一对应 |
| `voxel_size` | launch 为 `0.30` | 三维体素边长，单位为米 |
| `title` | `Trajectory BEV Monitor` | 图像和窗口标题 |
