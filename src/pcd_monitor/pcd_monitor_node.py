#!/usr/bin/env python3

import copy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
import numpy as np
import rospy

from pcd_monitor.pcd import PcdError, build_bev, find_pcd_files


class PcdMonitor:
    def __init__(self):
        pcd_dir_param = str(rospy.get_param("~pcd_dir", "")).strip()
        if not pcd_dir_param:
            raise PcdError("~pcd_dir is empty; set it to a folder containing PCD files")
        self.pcd_dir = Path(pcd_dir_param).expanduser()
        self.resolution = float(rospy.get_param("~resolution", 0.10))
        self.recursive = bool(rospy.get_param("~recursive", False))
        self.colormap = rospy.get_param("~colormap", "turbo")
        self.max_grid_cells = int(rospy.get_param("~max_grid_cells", 16000000))

        try:
            plt.get_cmap(self.colormap)
        except ValueError as error:
            raise PcdError("Unknown Matplotlib colormap: {}".format(self.colormap)) from error

        rospy.loginfo("Scanning PCD directory: %s", self.pcd_dir)
        self.pcd_files = find_pcd_files(self.pcd_dir, self.recursive)
        rospy.loginfo("Found %d PCD file(s), building BEV...", len(self.pcd_files))
        self.bev = build_bev(
            self.pcd_files,
            self.resolution,
            self.max_grid_cells,
        )
        rospy.loginfo(
            "BEV ready: %d finite points, %d occupied pixels",
            self.bev["finite_points"],
            self.bev["occupied_pixels"],
        )

        self._drag_state = None
        self._create_figure()

    def _create_figure(self):
        self.figure, self.axis = plt.subplots(figsize=(12.8, 8.2))
        self.figure.canvas.manager.set_window_title("PCD BEV Monitor")
        self.figure.subplots_adjust(left=0.09, bottom=0.14, right=0.88, top=0.90)

        grid = np.ma.masked_where(~self.bev["occupied"], self.bev["grid"])
        occupied_values = self.bev["grid"][self.bev["occupied"]]
        value_min = float(np.min(occupied_values))
        value_max = float(np.max(occupied_values))
        if value_min == value_max:
            # Normalize needs a non-zero range for a meaningful colorbar.
            padding = max(abs(value_min) * 0.01, 0.01)
            value_min -= padding
            value_max += padding

        colormap = copy.copy(plt.get_cmap(self.colormap))
        colormap.set_bad(color="#f3f4f6", alpha=1.0)
        self.image = self.axis.imshow(
            grid,
            origin="lower",
            extent=self.bev["extent"],
            interpolation="nearest",
            cmap=colormap,
            norm=Normalize(vmin=value_min, vmax=value_max),
            aspect="equal",
        )
        colorbar = self.figure.colorbar(self.image, ax=self.axis, pad=0.02)
        colorbar.set_label("Maximum Z in pixel (m)")

        self.axis.set_title(self._title())
        self.axis.set_xlabel("X (m)")
        self.axis.set_ylabel("Y (m)")
        self.axis.grid(True, color="#ffffff", linestyle="--", linewidth=0.45, alpha=0.35)
        self._set_full_view()

        reset_axis = self.figure.add_axes([0.09, 0.035, 0.13, 0.055])
        self.reset_button = Button(
            reset_axis,
            "Reset view",
            color="#2563eb",
            hovercolor="#1d4ed8",
        )
        self.reset_button.label.set_color("white")
        self.reset_button.label.set_fontweight("bold")
        self.reset_button.on_clicked(self._reset_clicked)

        self.figure.text(
            0.24,
            0.059,
            "Wheel: zoom    Left drag: pan    Double click / R: reset",
            ha="left",
            va="center",
            fontsize=9,
            color="#4b5563",
        )

        canvas = self.figure.canvas
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_button_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_button_release)
        canvas.mpl_connect("key_press_event", self._on_key_press)
        canvas.mpl_connect("close_event", self._on_close)
        rospy.on_shutdown(self._close_figure)

    def _title(self):
        bounds = self.bev["xyz_bounds"]
        return (
            "PCD BEV — {files} files, {points:,} points, {resolution:g} m/pixel\n"
            "X [{xmin:.3f}, {xmax:.3f}] m   "
            "Y [{ymin:.3f}, {ymax:.3f}] m   "
            "Z [{zmin:.3f}, {zmax:.3f}] m"
        ).format(
            files=self.bev["file_count"],
            points=self.bev["finite_points"],
            resolution=self.bev["resolution"],
            xmin=bounds[0][0],
            xmax=bounds[0][1],
            ymin=bounds[1][0],
            ymax=bounds[1][1],
            zmin=bounds[2][0],
            zmax=bounds[2][1],
        )

    def _toolbar_is_idle(self):
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return toolbar is None or not getattr(toolbar, "mode", "")

    def _on_scroll(self, event):
        if (
            event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
            or not self._toolbar_is_idle()
        ):
            return

        zoom_factor = 1.0 / 1.25 if event.button == "up" else 1.25
        x_min, x_max = self.axis.get_xlim()
        y_min, y_max = self.axis.get_ylim()
        relative_x = (event.xdata - x_min) / (x_max - x_min)
        relative_y = (event.ydata - y_min) / (y_max - y_min)
        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor
        self.axis.set_xlim(
            event.xdata - relative_x * new_width,
            event.xdata + (1.0 - relative_x) * new_width,
        )
        self.axis.set_ylim(
            event.ydata - relative_y * new_height,
            event.ydata + (1.0 - relative_y) * new_height,
        )
        self.figure.canvas.draw_idle()

    def _on_button_press(self, event):
        if event.inaxes is not self.axis:
            return
        if event.dblclick and event.button == 1:
            self._set_full_view()
            self.figure.canvas.draw_idle()
            return
        if (
            event.button == 1
            and event.xdata is not None
            and event.ydata is not None
            and self._toolbar_is_idle()
        ):
            self._drag_state = (
                event.xdata,
                event.ydata,
                self.axis.get_xlim(),
                self.axis.get_ylim(),
            )

    def _on_motion(self, event):
        if (
            self._drag_state is None
            or event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        start_x, start_y, x_limits, y_limits = self._drag_state
        delta_x = event.xdata - start_x
        delta_y = event.ydata - start_y
        self.axis.set_xlim(x_limits[0] - delta_x, x_limits[1] - delta_x)
        self.axis.set_ylim(y_limits[0] - delta_y, y_limits[1] - delta_y)
        self.figure.canvas.draw_idle()

    def _on_button_release(self, _event):
        self._drag_state = None

    def _on_key_press(self, event):
        if event.key and event.key.lower() == "r":
            self._set_full_view()
            self.figure.canvas.draw_idle()

    def _reset_clicked(self, _event):
        self._set_full_view()
        self.figure.canvas.draw_idle()

    def _set_full_view(self):
        x_min, x_max, y_min, y_max = self.bev["extent"]
        x_padding = max((x_max - x_min) * 0.02, self.resolution)
        y_padding = max((y_max - y_min) * 0.02, self.resolution)
        self.axis.set_xlim(x_min - x_padding, x_max + x_padding)
        self.axis.set_ylim(y_min - y_padding, y_max + y_padding)

    def _on_close(self, _event):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("PCD monitor window closed")

    def _close_figure(self):
        if hasattr(self, "figure"):
            plt.close(self.figure)

    def show(self):
        plt.show()


def main():
    rospy.init_node("pcd_monitor")
    try:
        monitor = PcdMonitor()
    except (PcdError, ValueError) as error:
        rospy.logfatal("Could not start pcd_monitor: %s", error)
        return 1
    monitor.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
