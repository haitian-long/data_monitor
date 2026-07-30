#!/usr/bin/env python3

import copy
import os
from pathlib import Path
import sys
import time


def _prefer_xcb_on_wslg():
    """Avoid Qt/Wayland window-decoration limitations under WSLg."""
    if os.environ.get("QT_QPA_PLATFORM") or not os.environ.get("DISPLAY"):
        return False

    try:
        kernel_release = Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False

    if "microsoft" not in kernel_release and "wsl" not in kernel_release:
        return False

    # This must be set before importing pyplot, which may create QApplication.
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    return True


WSLG_XCB_ENABLED = _prefer_xcb_on_wslg()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import rospy

from pcd_monitor.pcd import (
    PcdError,
    build_bev,
    extent_for_bounds,
    find_pcd_files,
    load_point_clouds,
)


class PcdMonitor:
    BEV_WIDTH = 2560
    BEV_HEIGHT = 1440
    BEV_ASPECT_RATIO = float(BEV_WIDTH) / float(BEV_HEIGHT)
    INITIAL_FIGURE_SIZE = (12.8, 7.2)
    FIGURE_DPI = 100
    INTERACTION_PREVIEW_WIDTH = 1280
    INTERACTION_PREVIEW_HEIGHT = 720
    DRAG_FRAME_INTERVAL = 1.0 / 30.0
    INTERACTION_RESTORE_DELAY_MS = 180
    MIN_VIEW_VOXELS = 8.0
    MAX_FULL_VIEW_SCALE = 4.0
    TITLE_PAD_POINTS = 15.0

    def __init__(self):
        pcd_dir_param = str(rospy.get_param("~pcd_dir", "")).strip()
        if not pcd_dir_param:
            raise PcdError("~pcd_dir is empty; set it to a folder containing PCD files")
        self.pcd_dir = Path(pcd_dir_param).expanduser()
        self.recursive = bool(rospy.get_param("~recursive", False))
        self.colormap = rospy.get_param("~colormap", "turbo")
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.10))
        self.title = str(rospy.get_param("~title", "PCD BEV Monitor"))

        try:
            plt.get_cmap(self.colormap)
        except ValueError as error:
            raise PcdError("Unknown Matplotlib colormap: {}".format(self.colormap)) from error

        rospy.loginfo("Scanning PCD directory: %s", self.pcd_dir)
        self.pcd_files = find_pcd_files(self.pcd_dir, self.recursive)
        rospy.loginfo(
            "Found %d PCD file(s), applying %.6g m voxel filter...",
            len(self.pcd_files),
            self.voxel_size,
        )
        self.sampled_xyz, self.dataset_stats = load_point_clouds(
            self.pcd_files,
            self.voxel_size,
        )
        rospy.loginfo(
            "Voxel filter: %d finite source points -> %d cached sampled points "
            "(%.2f MiB)",
            self.dataset_stats["finite_points"],
            self.dataset_stats["sampled_points"],
            self.dataset_stats["cache_bytes"] / (1024.0 * 1024.0),
        )
        display_percentiles = self.dataset_stats["display_z_percentiles"]
        display_z_range = self.dataset_stats["display_z_range"]
        rospy.loginfo(
            "Fixed color range: Z percentile %.0f%%..%.0f%% = %.6g..%.6g m",
            display_percentiles[0],
            display_percentiles[1],
            display_z_range[0],
            display_z_range[1],
        )
        bounds = self.dataset_stats["xyz_bounds"]
        self.full_extent = extent_for_bounds(
            bounds[0],
            bounds[1],
            aspect_ratio=self.BEV_ASPECT_RATIO,
        )
        full_view_width = self.full_extent[1] - self.full_extent[0]
        self.min_view_width = min(
            full_view_width,
            self.voxel_size * self.MIN_VIEW_VOXELS,
        )
        self.max_view_width = full_view_width * self.MAX_FULL_VIEW_SCALE

        rospy.loginfo(
            "Building initial %d x %d BEV...",
            self.BEV_WIDTH,
            self.BEV_HEIGHT,
        )
        self.bev = build_bev(
            self.sampled_xyz,
            self.full_extent,
            grid_width=self.BEV_WIDTH,
            grid_height=self.BEV_HEIGHT,
            dataset_stats=self.dataset_stats,
        )
        self.full_bev = self.bev
        self._log_bev_info("Initial BEV", self.bev)

        self._drag_state = None
        self._create_figure()

    def _create_figure(self):
        self.figure, self.axis = plt.subplots(
            figsize=self.INITIAL_FIGURE_SIZE,
            dpi=self.FIGURE_DPI,
        )
        self.figure.canvas.manager.set_window_title(self.title)
        self._configure_resizable_window()
        self.figure.subplots_adjust(left=0.09, bottom=0.17, right=0.88, top=0.93)
        plot_position = self.axis.get_position()
        self.axis.set_position(
            (
                (1.0 - plot_position.width) / 2.0,
                plot_position.y0,
                plot_position.width,
                plot_position.height,
            )
        )

        self._prepare_image_grids(self.bev)
        preview_height, preview_width = self._interaction_preview_grid.shape
        rospy.loginfo(
            "Interaction preview: %d x %d max-Z pooled BEV, "
            "drag refresh limited to %.0f FPS",
            preview_width,
            preview_height,
            1.0 / self.DRAG_FRAME_INTERVAL,
        )

        colormap = copy.copy(plt.get_cmap(self.colormap))
        colormap.set_bad(color="#f3f4f6", alpha=1.0)
        self.image = self.axis.imshow(
            self._full_image_grid,
            origin="lower",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
            cmap=colormap,
            norm=self._normalization_for_bev(self.bev),
            aspect="equal",
        )
        self._using_interaction_preview = False
        # Keep the BEV axes centered in the Figure. A normal colorbar with
        # ax=self.axis shrinks the main axes toward the left.
        self.axis.set_anchor("C")
        self.colorbar_axis = inset_axes(
            self.axis,
            width="2.5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=self.axis.transAxes,
            borderpad=0.0,
        )
        self.colorbar = self.figure.colorbar(self.image, cax=self.colorbar_axis)
        self.colorbar.set_label("Maximum Z in pixel (m), fixed P1-P99")

        self._set_plot_title()
        self.axis.set_xlabel("X (m)")
        self.axis.set_ylabel("Y (m)")
        self.axis.grid(True, color="#ffffff", linestyle="--", linewidth=0.45, alpha=0.35)
        self._set_axis_extent(self.bev["extent"])

        rebuild_axis = self.figure.add_axes([0.09, 0.035, 0.18, 0.06])
        self.rebuild_button = Button(
            rebuild_axis,
            "Rebuild BEV",
            color="#f59e0b",
            hovercolor="#d97706",
        )
        self._style_button(
            rebuild_axis,
            self.rebuild_button,
            "#f59e0b",
            "#b45309",
        )
        self.rebuild_button.on_clicked(self._rebuild_clicked)

        reset_axis = self.figure.add_axes([0.285, 0.035, 0.18, 0.06])
        self.reset_button = Button(
            reset_axis,
            "Reset view",
            color="#2563eb",
            hovercolor="#1d4ed8",
        )
        self._style_button(
            reset_axis,
            self.reset_button,
            "#2563eb",
            "#1e40af",
        )
        self.reset_button.on_clicked(self._reset_clicked)

        self.figure.text(
            0.49,
            0.059,
            "Wheel: zoom    Left drag: pan    Rebuild: rerasterize current view",
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
        canvas.mpl_connect("resize_event", self._on_resize)
        canvas.mpl_connect("close_event", self._on_close)
        self._interaction_restore_timer = canvas.new_timer(
            interval=self.INTERACTION_RESTORE_DELAY_MS,
        )
        self._interaction_restore_timer.single_shot = True
        self._interaction_restore_timer.add_callback(
            self._restore_full_resolution_after_interaction
        )
        rospy.loginfo(
            "Zoom X-span limits: %.6g..%.6g m; full-resolution restore delay: %d ms",
            self.min_view_width,
            self.max_view_width,
            self.INTERACTION_RESTORE_DELAY_MS,
        )
        rospy.on_shutdown(self._close_figure)

    def _configure_resizable_window(self):
        manager = self.figure.canvas.manager
        window = getattr(manager, "window", None)
        configured_backends = []
        qt_platform = None

        # Tk/TkAgg top-level window.
        if window is not None and callable(getattr(window, "resizable", None)):
            window.resizable(True, True)
            if callable(getattr(window, "minsize", None)):
                window.minsize(640, 360)
            configured_backends.append("Tk")

        # Qt widgets. Reset constraints that can turn resize handles into a
        # fixed-size window, including constraints inherited by the canvas.
        if window is not None and callable(getattr(window, "setMinimumSize", None)):
            from matplotlib.backends.qt_compat import QtCore, QtWidgets

            qt_namespace = QtCore.Qt
            window_type = getattr(qt_namespace, "WindowType", qt_namespace)

            def qt_window_flag(name):
                return getattr(window_type, name, getattr(qt_namespace, name, None))

            flags = window.windowFlags()
            for flag_name in ("MSWindowsFixedSizeDialogHint", "FramelessWindowHint"):
                flag = qt_window_flag(flag_name)
                if flag is not None:
                    flags &= ~flag
            for flag_name in (
                "Window",
                "WindowTitleHint",
                "WindowSystemMenuHint",
                "WindowMinMaxButtonsHint",
                "WindowCloseButtonHint",
            ):
                flag = qt_window_flag(flag_name)
                if flag is not None:
                    flags |= flag
            window.setWindowFlags(flags)

            policy_namespace = getattr(
                QtWidgets.QSizePolicy,
                "Policy",
                QtWidgets.QSizePolicy,
            )
            expanding = policy_namespace.Expanding
            size_policy = QtWidgets.QSizePolicy(expanding, expanding)

            window.setMinimumSize(640, 360)
            window.setMaximumSize(16777215, 16777215)
            window.setSizePolicy(size_policy)

            canvas = self.figure.canvas
            if callable(getattr(canvas, "setMinimumSize", None)):
                canvas.setMinimumSize(1, 1)
                canvas.setMaximumSize(16777215, 16777215)
                canvas.setSizePolicy(size_policy)

            central_widget = window.centralWidget()
            if central_widget is not None:
                central_widget.setMinimumSize(1, 1)
                central_widget.setMaximumSize(16777215, 16777215)
                central_widget.setSizePolicy(size_policy)

            layout = window.layout()
            if layout is not None:
                constraint_namespace = getattr(
                    QtWidgets.QLayout,
                    "SizeConstraint",
                    QtWidgets.QLayout,
                )
                layout.setSizeConstraint(constraint_namespace.SetDefaultConstraint)

            application = QtWidgets.QApplication.instance()
            if application is not None:
                qt_platform = application.platformName()
            configured_backends.append(
                "Qt ({})".format(qt_platform or os.environ.get("QT_QPA_PLATFORM", "auto"))
            )

        # GTK exposes a direct resizable property.
        if window is not None and callable(getattr(window, "set_resizable", None)):
            window.set_resizable(True)
            if callable(getattr(window, "set_default_size", None)):
                window.set_default_size(1280, 720)
            configured_backends.append("GTK")

        # Matplotlib's manager resize is implemented by interactive backends
        # and is harmless for non-interactive Agg validation.
        if callable(getattr(manager, "resize", None)):
            manager.resize(1280, 720)

        rospy.loginfo(
            "Matplotlib window backend: %s; explicit resize support: %s%s",
            type(manager).__name__,
            ", ".join(configured_backends) if configured_backends else "manager default",
            "; WSLg XCB workaround enabled" if WSLG_XCB_ENABLED else "",
        )

    def _toolbar_is_idle(self):
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return toolbar is None or not getattr(toolbar, "mode", "")

    def _on_scroll(self, event):
        if (
            event.inaxes is not self.axis
            or event.xdata is None
            or event.ydata is None
            or self._drag_state is not None
            or not self._toolbar_is_idle()
        ):
            return

        if event.button == "up":
            zoom_factor = 1.0 / 1.25
        elif event.button == "down":
            zoom_factor = 1.25
        else:
            return

        x_min, x_max = self.axis.get_xlim()
        y_min, y_max = self.axis.get_ylim()
        relative_x = (event.xdata - x_min) / (x_max - x_min)
        relative_y = (event.ydata - y_min) / (y_max - y_min)
        current_width = x_max - x_min
        current_height = y_max - y_min
        new_width = min(
            self.max_view_width,
            max(self.min_view_width, current_width * zoom_factor),
        )
        if new_width == current_width:
            return
        applied_zoom_factor = new_width / current_width
        new_height = current_height * applied_zoom_factor

        self._show_interaction_preview()
        self.axis.set_xlim(
            event.xdata - relative_x * new_width,
            event.xdata + (1.0 - relative_x) * new_width,
        )
        self.axis.set_ylim(
            event.ydata - relative_y * new_height,
            event.ydata + (1.0 - relative_y) * new_height,
        )
        self.figure.canvas.draw_idle()
        self._schedule_full_resolution_restore()

    def _on_resize(self, _event):
        self._show_interaction_preview()
        self._schedule_full_resolution_restore()

    def _on_button_press(self, event):
        if event.inaxes is not self.axis:
            return
        if event.dblclick and event.button == 1:
            self._restore_full_bev()
            return
        if (
            event.button == 1
            and event.x is not None
            and event.y is not None
            and self._toolbar_is_idle()
        ):
            x_limits = self.axis.get_xlim()
            y_limits = self.axis.get_ylim()
            self._drag_state = {
                "start_pixel": (float(event.x), float(event.y)),
                "x_limits": x_limits,
                "y_limits": y_limits,
                "axis_pixels": (
                    max(float(self.axis.bbox.width), 1.0),
                    max(float(self.axis.bbox.height), 1.0),
                ),
                "last_draw_time": 0.0,
            }
            self._interaction_restore_timer.stop()
            self._show_interaction_preview()

    def _on_motion(self, event):
        if (
            self._drag_state is None
            or event.x is None
            or event.y is None
        ):
            return

        now = time.monotonic()
        if now - self._drag_state["last_draw_time"] < self.DRAG_FRAME_INTERVAL:
            return

        x_limits, y_limits = self._drag_limits_for_event(event)
        self._set_axis_limits(x_limits, y_limits)
        self._drag_state["last_draw_time"] = now
        self.figure.canvas.draw_idle()

    def _on_button_release(self, event):
        if self._drag_state is None:
            return

        if event.x is not None and event.y is not None:
            x_limits, y_limits = self._drag_limits_for_event(event)
            self._set_axis_limits(x_limits, y_limits)

        self._drag_state = None
        self._show_full_resolution()
        self.figure.canvas.draw_idle()

    def _drag_limits_for_event(self, event):
        start_x, start_y = self._drag_state["start_pixel"]
        axis_width, axis_height = self._drag_state["axis_pixels"]
        x_limits = self._drag_state["x_limits"]
        y_limits = self._drag_state["y_limits"]

        delta_x = (
            (float(event.x) - start_x)
            * (x_limits[1] - x_limits[0])
            / axis_width
        )
        delta_y = (
            (float(event.y) - start_y)
            * (y_limits[1] - y_limits[0])
            / axis_height
        )
        return (
            (x_limits[0] - delta_x, x_limits[1] - delta_x),
            (y_limits[0] - delta_y, y_limits[1] - delta_y),
        )

    def _set_axis_limits(self, x_limits, y_limits):
        self.axis.set_xlim(x_limits)
        self.axis.set_ylim(y_limits)

    def _set_plot_title(self):
        self.axis.set_title(
            self.title,
            pad=self.TITLE_PAD_POINTS,
        )

    def _style_button(self, axis, button, facecolor, spine_color):
        axis.set_facecolor(facecolor)
        for spine in axis.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(2.0)
        button.label.set_color("#ffffff")
        button.label.set_fontsize(10)
        button.label.set_fontweight("bold")

    def _show_interaction_preview(self):
        if (
            self._using_interaction_preview
            or self._interaction_preview_grid is self._full_image_grid
        ):
            return
        self.image.set_data(self._interaction_preview_grid)
        self._using_interaction_preview = True

    def _show_full_resolution(self):
        if not self._using_interaction_preview:
            return
        self.image.set_data(self._full_image_grid)
        self._using_interaction_preview = False

    def _schedule_full_resolution_restore(self):
        self._interaction_restore_timer.stop()
        self._interaction_restore_timer.start()

    def _restore_full_resolution_after_interaction(self):
        if self._drag_state is not None:
            return
        self._show_full_resolution()
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key and event.key.lower() == "r":
            self._restore_full_bev()

    def _reset_clicked(self, _event):
        self._restore_full_bev()

    def _rebuild_clicked(self, _event):
        x_limits = self.axis.get_xlim()
        y_limits = self.axis.get_ylim()
        extent = extent_for_bounds(
            x_limits,
            y_limits,
            aspect_ratio=self.BEV_ASPECT_RATIO,
            padding_ratio=0.0,
        )

        self.rebuild_button.label.set_text("Rebuilding...")
        self.figure.canvas.draw()
        try:
            bev = build_bev(
                self.sampled_xyz,
                extent,
                grid_width=self.BEV_WIDTH,
                grid_height=self.BEV_HEIGHT,
                dataset_stats=self.dataset_stats,
            )
        except PcdError as error:
            rospy.logerr("Could not rebuild PCD BEV: %s", error)
        else:
            self._apply_bev(bev)
            self._log_bev_info("Rebuilt BEV", bev)
        finally:
            self.rebuild_button.label.set_text("Rebuild BEV")
            self.figure.canvas.draw_idle()

    def _restore_full_bev(self):
        self._apply_bev(self.full_bev)
        self._log_bev_info("Restored full BEV", self.full_bev)
        self.figure.canvas.draw_idle()

    def _apply_bev(self, bev):
        self._interaction_restore_timer.stop()
        self._drag_state = None
        self.bev = bev
        self._prepare_image_grids(bev)
        self.image.set_data(self._full_image_grid)
        self._using_interaction_preview = False
        self.image.set_extent(bev["extent"])
        self.image.set_norm(self._normalization_for_bev(bev))
        self.colorbar.update_normal(self.image)
        self._set_plot_title()
        self._set_axis_extent(bev["extent"])

    def _log_bev_info(self, label, bev):
        min_x, max_x, min_y, max_y = bev["extent"]
        min_z, max_z = bev["xyz_bounds"][2]
        color_min_z, color_max_z = bev["display_z_range"]
        rospy.loginfo(
            "%s: %d x %d, %.6g m/pixel, voxel %.6g m, "
            "%d/%d sampled points in view, %d occupied pixels",
            label,
            bev["grid_width"],
            bev["grid_height"],
            bev["resolution"],
            bev["voxel_size"],
            bev["visible_points"],
            bev["sampled_points"],
            bev["occupied_pixels"],
        )
        rospy.loginfo(
            "%s ranges: X [%.6g, %.6g] m, Y [%.6g, %.6g] m, "
            "source Z [%.6g, %.6g] m, fixed color Z [%.6g, %.6g] m",
            label,
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            color_min_z,
            color_max_z,
        )

    def _normalization_for_bev(self, bev):
        value_min, value_max = bev["display_z_range"]
        if value_min == value_max:
            padding = max(abs(value_min) * 0.01, 0.01)
            value_min -= padding
            value_max += padding
        return Normalize(vmin=value_min, vmax=value_max, clip=True)

    def _prepare_image_grids(self, bev):
        self._full_image_grid = bev["grid"]
        if "_interaction_preview_grid" in bev:
            self._interaction_preview_grid = bev["_interaction_preview_grid"]
            return

        grid = self._full_image_grid
        grid_height, grid_width = grid.shape
        preview_scale = max(
            1,
            (grid_width + self.INTERACTION_PREVIEW_WIDTH - 1)
            // self.INTERACTION_PREVIEW_WIDTH,
            (grid_height + self.INTERACTION_PREVIEW_HEIGHT - 1)
            // self.INTERACTION_PREVIEW_HEIGHT,
        )
        if preview_scale == 1:
            self._interaction_preview_grid = self._full_image_grid
            bev["_interaction_preview_grid"] = self._interaction_preview_grid
            return

        preview_height = (grid_height + preview_scale - 1) // preview_scale
        preview_width = (grid_width + preview_scale - 1) // preview_scale
        preview_grid = np.full(
            (preview_height, preview_width),
            -np.inf,
            dtype=grid.dtype,
        )

        # Max-pooling preserves the highest Z point in every preview pixel.
        # Sliced reductions avoid allocating a full-size temporary array.
        for row_offset in range(preview_scale):
            for column_offset in range(preview_scale):
                source = grid[
                    row_offset::preview_scale,
                    column_offset::preview_scale,
                ]
                destination = preview_grid[: source.shape[0], : source.shape[1]]
                np.maximum(destination, source, out=destination)

        self._interaction_preview_grid = preview_grid
        bev["_interaction_preview_grid"] = self._interaction_preview_grid

    def _set_axis_extent(self, extent):
        min_x, max_x, min_y, max_y = extent
        self.axis.set_xlim(min_x, max_x)
        self.axis.set_ylim(min_y, max_y)

    def _on_close(self, _event):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("PCD monitor window closed")

    def _close_figure(self):
        if hasattr(self, "figure"):
            if hasattr(self, "_interaction_restore_timer"):
                self._interaction_restore_timer.stop()
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
