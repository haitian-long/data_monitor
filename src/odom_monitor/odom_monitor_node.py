#!/usr/bin/env python3

import math
import threading

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from nav_msgs.msg import Odometry
import rospy


class OdomMonitor:
    def __init__(self):
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")

        # ROS 订阅回调和 Matplotlib 动画刷新会从不同路径访问状态，
        # 因此用 lock 保护 pose 和历史数据。
        self.lock = threading.Lock()

        # pose 存储格式为 ((x, y, z), (qx, qy, qz, qw))。
        # origin_pose 是 Reset 后的参考原点；latest_pose 是 odometry 原始 pose。
        self.latest_pose = None
        self.origin_pose = None
        self.is_paused = False

        # 自上次 Reset 以来的相对 pose 历史。
        # position 单位为 m；rotation 为 roll/pitch/yaw，单位为 degree。
        self.x_values = []
        self.y_values = []
        self.z_values = []
        self.roll_values = []
        self.pitch_values = []
        self.yaw_values = []

        self.subscriber = rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self._odometry_callback,
            queue_size=50,
        )

        self.figure, self.axis = plt.subplots(figsize=(13.5, 7))
        self.figure.canvas.manager.set_window_title("Odom Monitor")

        # 给右侧信息面板预留空间。
        self.figure.subplots_adjust(left=0.06, bottom=0.18, right=0.58)

        # 主图中的 X-Y 轨迹对象，FuncAnimation 会原地更新它们。
        (self.path_line,) = self.axis.plot([], [], color="#2563eb", linewidth=1.2)
        (self.current_point,) = self.axis.plot([], [], "o", color="#dc2626", markersize=4)

        self.axis.set_title("Odometry X-Y Trace")
        self.axis.set_xlabel("X relative to reset origin (m)")
        self.axis.set_ylabel("Y relative to reset origin (m)")
        self.axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        self.axis.axhline(0.0, color="#111827", linewidth=0.8)
        self.axis.axvline(0.0, color="#111827", linewidth=0.8)
        self.axis.set_aspect("equal", adjustable="box")

        # 右侧信息面板：[left, bottom, width, height]，坐标基于整个 figure。
        self.info_axis = self.figure.add_axes([0.555, 0.13, 0.375, 0.78])
        self.info_axis.set_axis_off()
        self.info_axis.set_xlim(0.0, 1.0)
        self.info_axis.set_ylim(0.0, 1.0)
        self._render_waiting_panel()

        # 控制按钮：[left, bottom, width, height]，坐标基于整个 figure。
        reset_axis = self.figure.add_axes([0.555, 0.04, 0.18, 0.06])
        self.reset_button = Button(
            reset_axis,
            "Reset",
            color="#2563eb",
            hovercolor="#1d4ed8",
        )
        self._style_button(reset_axis, self.reset_button, "#2563eb", "#1e40af")
        self.reset_button.on_clicked(self._reset_clicked)

        pause_axis = self.figure.add_axes([0.75, 0.04, 0.18, 0.06])
        self.pause_button = Button(
            pause_axis,
            "Pause",
            color="#f59e0b",
            hovercolor="#d97706",
        )
        self._style_button(pause_axis, self.pause_button, "#f59e0b", "#b45309")
        self.pause_button.on_clicked(self._pause_clicked)

        self.animation = FuncAnimation(
            self.figure,
            self._update_plot,
            interval=100,
            cache_frame_data=False,
        )

    def _odometry_callback(self, message):
        absolute_pose = self._pose_from_message(message)

        with self.lock:
            self.latest_pose = absolute_pose
            if self.origin_pose is None:
                # 第一次收到 odometry 时，自动把当前 pose 作为相对 pose 原点。
                self.origin_pose = absolute_pose

            if self.is_paused:
                return

            relative_position = self._relative_position(absolute_pose[0])
            relative_rpy = self._relative_rpy_degrees(absolute_pose[1])
            self.x_values.append(relative_position[0])
            self.y_values.append(relative_position[1])
            self.z_values.append(relative_position[2])
            self.roll_values.append(relative_rpy[0])
            self.pitch_values.append(relative_rpy[1])
            self.yaw_values.append(relative_rpy[2])

    def _pose_from_message(self, message):
        # nav_msgs/Odometry 的 pose 位于 message.pose.pose。
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        return (
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w),
        )

    def _relative_position(self, absolute_position):
        # 相对平移 = 当前 position - 原点 position。
        origin_position = self.origin_pose[0]
        return (
            absolute_position[0] - origin_position[0],
            absolute_position[1] - origin_position[1],
            absolute_position[2] - origin_position[2],
        )

    def _relative_rpy_degrees(self, absolute_orientation):
        origin_orientation = self.origin_pose[1]

        # 相对旋转：q_relative = inverse(q_origin) * q_current。
        relative_orientation = self._quaternion_multiply(
            self._quaternion_inverse(origin_orientation),
            absolute_orientation,
        )

        # 转欧拉角前先归一化，减少四元数数值误差带来的漂移。
        relative_orientation = self._quaternion_normalize(relative_orientation)
        return tuple(math.degrees(value) for value in self._quaternion_to_rpy(relative_orientation))

    def _quaternion_normalize(self, quaternion):
        x, y, z, w = quaternion
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm == 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm, w / norm)

    def _quaternion_inverse(self, quaternion):
        x, y, z, w = quaternion
        norm_squared = x * x + y * y + z * z + w * w
        if norm_squared == 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        return (-x / norm_squared, -y / norm_squared, -z / norm_squared, w / norm_squared)

    def _quaternion_multiply(self, left, right):
        left_x, left_y, left_z, left_w = left
        right_x, right_y, right_z, right_w = right
        return (
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        )

    def _quaternion_to_rpy(self, quaternion):
        x, y, z, w = quaternion

        # 按 ROS 常见的 xyzw 四元数顺序转换为 roll/pitch/yaw。
        roll = math.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )

        pitch_sin = 2.0 * (w * y - z * x)
        pitch_sin = max(-1.0, min(1.0, pitch_sin))
        pitch = math.asin(pitch_sin)

        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return (roll, pitch, yaw)

    def _reset_clicked(self, _event):
        with self.lock:
            if self.latest_pose is None:
                return

            # Reset 后，当前绝对 pose 会成为新的相对 pose 原点。
            self.origin_pose = self.latest_pose
            self.x_values = [0.0]
            self.y_values = [0.0]
            self.z_values = [0.0]
            self.roll_values = [0.0]
            self.pitch_values = [0.0]
            self.yaw_values = [0.0]

    def _pause_clicked(self, _event):
        with self.lock:
            self.is_paused = not self.is_paused
            is_paused = self.is_paused

        if is_paused:
            self.pause_button.label.set_text("Resume")
        else:
            self.pause_button.label.set_text("Pause")
        self.figure.canvas.draw_idle()

    def _snapshot(self):
        with self.lock:
            # 返回列表副本，避免绘图过程中长时间持有 lock。
            return (
                list(self.x_values),
                list(self.y_values),
                list(self.z_values),
                list(self.roll_values),
                list(self.pitch_values),
                list(self.yaw_values),
                self.latest_pose,
                self.is_paused,
            )

    def _update_plot(self, _frame):
        (
            x_values,
            y_values,
            z_values,
            roll_values,
            pitch_values,
            yaw_values,
            latest_pose,
            is_paused,
        ) = self._snapshot()

        if rospy.is_shutdown():
            plt.close(self.figure)
            return self.path_line, self.current_point, self.info_axis

        if not x_values or not y_values:
            self._render_waiting_panel()
            return self.path_line, self.current_point, self.info_axis

        # 使用快照更新轨迹，并重绘右侧信息面板。
        self.path_line.set_data(x_values, y_values)
        self.current_point.set_data([x_values[-1]], [y_values[-1]])
        if not is_paused:
            self._rescale_axis(x_values, y_values)
        self._render_info_panel(
            x_values,
            y_values,
            z_values,
            roll_values,
            pitch_values,
            yaw_values,
            latest_pose,
            is_paused,
        )

        return self.path_line, self.current_point, self.info_axis

    def _rescale_axis(self, x_values, y_values):
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        # 保持坐标图为正方形，并添加边距，避免当前点贴在边界上。
        x_span = max(max_x - min_x, 1.0)
        y_span = max(max_y - min_y, 1.0)
        span = max(x_span, y_span)
        padding = span * 0.12

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_span = (span / 2.0) + padding

        self.axis.set_xlim(center_x - half_span, center_x + half_span)
        self.axis.set_ylim(center_y - half_span, center_y + half_span)

    def _render_waiting_panel(self):
        # 右侧信息面板由轻量 Matplotlib artist 组成，每次刷新时重绘。
        self.info_axis.clear()
        self.info_axis.set_axis_off()
        self.info_axis.set_xlim(0.0, 1.0)
        self.info_axis.set_ylim(0.0, 1.0)
        self._draw_cards(
            [
                (
                    "Odom Monitor",
                    [
                        ("Status", "Waiting for odometry"),
                        ("Topic", self.odom_topic),
                    ],
                ),
            ]
        )

    def _render_info_panel(
        self,
        x_values,
        y_values,
        z_values,
        roll_values,
        pitch_values,
        yaw_values,
        latest_pose,
        is_paused,
    ):
        self.info_axis.clear()
        self.info_axis.set_axis_off()
        self.info_axis.set_xlim(0.0, 1.0)
        self.info_axis.set_ylim(0.0, 1.0)

        current_position = (x_values[-1], y_values[-1], z_values[-1])
        current_rpy = (roll_values[-1], pitch_values[-1], yaw_values[-1])
        latest_position = latest_pose[0]
        latest_rpy = tuple(
            math.degrees(value) for value in self._quaternion_to_rpy(
                self._quaternion_normalize(latest_pose[1])
            )
        )

        # 这里只定义每张卡片的标题和内容行；卡片位置和高度由 _draw_cards() 自动计算。
        self._draw_cards(
            [
                (
                    "Source",
                    [
                        ("Topic", self.odom_topic),
                        ("Status", "Paused" if is_paused else "Running"),
                        ("Samples", "{}".format(len(x_values))),
                    ],
                ),
                (
                    "Relative Pose",
                    [
                        ("position (m)", self._format_vector(current_position, 3)),
                        ("rotation (deg)", self._format_vector(current_rpy, 2)),
                    ],
                ),
                (
                    "Fluctuation Range",
                    [
                        ("x (m)", self._format_range(x_values, 3)),
                        ("y (m)", self._format_range(y_values, 3)),
                        ("z (m)", self._format_range(z_values, 3)),
                        ("roll (deg)", self._format_range(roll_values, 2)),
                        ("pitch (deg)", self._format_range(pitch_values, 2)),
                        ("yaw (deg)", self._format_range(yaw_values, 2)),
                    ],
                ),
                (
                    "Latest Absolute Pose",
                    [
                        ("position (m)", self._format_vector(latest_position, 3)),
                        ("rotation (deg)", self._format_vector(latest_rpy, 2)),
                    ],
                ),
            ]
        )

    def _style_button(self, axis, button, facecolor, spine_color):
        axis.set_facecolor(facecolor)
        for spine in axis.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(2.0)
        button.label.set_color("#ffffff")
        button.label.set_fontsize(10)
        button.label.set_fontweight("bold")

    def _draw_cards(self, cards):
        # 自动布局参数均为右侧信息面板的归一化坐标。
        top = 0.98
        bottom_limit = 0.00
        card_gap = 0.04

        heights = [self._card_height(rows) for _title, rows in cards]
        total_height = sum(heights) + (card_gap * max(len(cards) - 1, 0))

        # 如果内容总高度超过面板，就按比例压缩卡片高度，避免溢出面板。
        available_height = top - bottom_limit
        if total_height > available_height:
            scale = (available_height - card_gap * max(len(cards) - 1, 0)) / sum(heights)
            heights = [height * scale for height in heights]

        current_top = top
        for (title, rows), height in zip(cards, heights):
            bottom = current_top - height
            self._draw_card(bottom, height, title, rows)
            current_top = bottom - card_gap

    def _card_height(self, rows):
        # 卡片高度由内容行数自动决定：
        title_and_top_padding = 0.075  # 第一行内容距离卡片顶部距离，第一行内容距标题0.05+标题距离顶部0.025。
        row_gap = 0.045  # 行距
        bottom_padding = 0.045  # 卡片底部留白
        return title_and_top_padding + row_gap * max(len(rows) - 1, 0) + bottom_padding

    def _draw_card(self, bottom, height, title, rows):

        # 绘制卡片
        card = FancyBboxPatch(
            (0.0, bottom),  # 卡片左下角坐标
            1.0,  # 卡片宽度
            height,  # 卡片高度
            boxstyle="round,pad=0.012,rounding_size=0.018",  # 圆角矩形
            linewidth=0.8,  # 边框线宽
            edgecolor="#d1d5db",  # 边框颜色
            facecolor="#f9fafb",  # 填充颜色
            transform=self.info_axis.transAxes,  # 使用信息面板坐标系
            clip_on=False,  # 允许卡片边界超出坐标轴范围，避免被裁剪掉圆角部分
        )
        self.info_axis.add_patch(card)

        # 绘制标题
        self.info_axis.text(
            0.04,  # 信息面板中标题左起始坐标
            bottom + height - 0.025,  # 信息面板中标题的 y 坐标
            title,  # 标题文本
            transform=self.info_axis.transAxes,  # 使用信息面板坐标系
            ha="left",  # 水平左对齐
            va="top",  # 垂直顶部对齐
            fontsize=10,  # 标题字体大小
            fontweight="bold",  # 标题字体加粗
            color="#111827",  # 标题颜色
        )

        # 第一行内容位置由卡片顶部向下偏移得到，不需要为每张卡片单独手动设置。
        row_top = bottom + height - 0.075

        # 绘制内容
        row_gap = 0.045  # 行距
        for index, (label, value) in enumerate(rows):
            y = row_top - (index * row_gap)
            # 内容名称
            self.info_axis.text(
                0.05,  # 标签列的 x 坐标。
                y,
                "{}:".format(label),
                transform=self.info_axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.6,
                color="#6b7280",
            )
            # 内容值
            self.info_axis.text(
                0.25,  # 数值列的 x 坐标。
                y,
                value,
                transform=self.info_axis.transAxes,
                ha="left",
                va="top",
                family="monospace",
                fontsize=8.8,
                color="#111827",
            )

    def _format_vector(self, values, precision):
        template = "[{}]".format(", ".join(["{:+.%df}" % precision] * len(values)))
        return template.format(*values)

    def _format_range(self, values, precision):
        template = "{:+.%df} .. {:+.%df}" % (precision, precision)
        return template.format(min(values), max(values))

    def show(self):
        rospy.loginfo("Listening to odometry topic: %s", self.odom_topic)
        plt.show()


def main():
    rospy.init_node("odom_monitor")
    monitor = OdomMonitor()
    monitor.show()


if __name__ == "__main__":
    main()
