#!/usr/bin/env python3

import copy
import math
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import rospy

from pcd_monitor.pcd import (
    PcdError,
    build_bev,
    extent_for_bounds,
    load_point_cloud,
    read_tum_trajectory_positions,
    validate_pcd_file,
)


class PcdMonitor:
    BEV_WIDTH = 2560
    BEV_HEIGHT = 1440
    BEV_RESOLUTIONS = (
        (2560, 1440),
        (1280, 720),
        (640, 360),
    )
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
    EXPORT_DPI = 200

    def __init__(self):
        pcd_path_param = str(rospy.get_param("~pcd_path", "")).strip()
        if not pcd_path_param:
            raise PcdError(
                "~pcd_path is empty; set it to a single PCD file"
            )
        self.pcd_path = validate_pcd_file(pcd_path_param)
        self.colormap = rospy.get_param("~colormap", "turbo")
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.10))
        self.title = str(rospy.get_param("~title", "PCD BEV Monitor"))
        trajectory_path_param = str(
            rospy.get_param("~trajectory_path", "")
        ).strip()
        self.trajectory_path = None
        self.trajectory_positions = None
        if trajectory_path_param:
            self.trajectory_path = Path(trajectory_path_param).expanduser()
            try:
                self.trajectory_positions = read_tum_trajectory_positions(
                    self.trajectory_path
                )
            except PcdError as error:
                rospy.logwarn("Skipping TUM trajectory: %s", error)
                self.trajectory_path = None
            else:
                rospy.loginfo(
                    "Loaded TUM trajectory: %s (%d poses)",
                    self.trajectory_path,
                    self.trajectory_positions.shape[0],
                )
        self.selected_bev_width = self.BEV_WIDTH
        self.selected_bev_height = self.BEV_HEIGHT

        try:
            plt.get_cmap(self.colormap)
        except ValueError as error:
            raise PcdError("Unknown Matplotlib colormap: {}".format(self.colormap)) from error

        rospy.loginfo("Using PCD file: %s", self.pcd_path)
        self.export_dir = self.pcd_path.parent
        rospy.loginfo(
            "Reading the complete PCD file, then applying one %.6g m "
            "voxel filter...",
            self.voxel_size,
        )
        self.sampled_xyz, self.dataset_stats = load_point_cloud(
            self.pcd_path,
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
        self.full_view_width = full_view_width
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
        self.trajectory_line = self._draw_trajectory(self.axis)
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

        button_width = 0.16
        button_gap = 0.015
        button_height = 0.06
        button_y = 0.035
        button_count = 5
        buttons_width = (
            button_count * button_width
            + (button_count - 1) * button_gap
        )
        first_button_x = (1.0 - buttons_width) / 2.0
        button_x_positions = [
            first_button_x + index * (button_width + button_gap)
            for index in range(button_count)
        ]

        rebuild_axis = self.figure.add_axes(
            [button_x_positions[3], button_y, button_width, button_height]
        )
        self.rebuild_button = Button(
            rebuild_axis,
            "Rebuild BEV",
            color="#2563eb",
            hovercolor="#1d4ed8",
        )
        self._style_button(
            rebuild_axis,
            self.rebuild_button,
            "#2563eb",
            "#1e40af",
        )
        self.rebuild_button.on_clicked(self._rebuild_clicked)

        reset_axis = self.figure.add_axes(
            [button_x_positions[0], button_y, button_width, button_height]
        )
        self.reset_button = Button(
            reset_axis,
            "Reset view",
            color="#f59e0b",
            hovercolor="#d97706",
        )
        self._style_button(
            reset_axis,
            self.reset_button,
            "#f59e0b",
            "#b45309",
        )
        self.reset_button.on_clicked(self._reset_clicked)

        save_axis = self.figure.add_axes(
            [button_x_positions[4], button_y, button_width, button_height]
        )
        self.save_button = Button(
            save_axis,
            "Save Image",
            color="#7c3aed",
            hovercolor="#6d28d9",
        )
        self._style_button(
            save_axis,
            self.save_button,
            "#7c3aed",
            "#5b21b6",
        )
        self.save_button.on_clicked(self._save_clicked)

        resolution_axis = self.figure.add_axes(
            [button_x_positions[2], button_y, button_width, button_height]
        )
        self.resolution_button = Button(
            resolution_axis,
            self._resolution_button_label(),
            color="#0891b2",
            hovercolor="#0e7490",
        )
        self._style_button(
            resolution_axis,
            self.resolution_button,
            "#0891b2",
            "#155e75",
        )
        self.resolution_button.on_clicked(self._toggle_resolution_popup)
        self._create_resolution_popup()

        zoom_axis = self.figure.add_axes(
            [button_x_positions[1], button_y, button_width, button_height]
        )
        self.zoom_button = Button(
            zoom_axis,
            "Zoom",
            color="#059669",
            hovercolor="#047857",
        )
        self._style_button(
            zoom_axis,
            self.zoom_button,
            "#059669",
            "#065f46",
        )
        self.zoom_button.on_clicked(self._toggle_zoom_slider)
        self._create_zoom_slider()
        self._control_button_axes = (
            reset_axis,
            zoom_axis,
            resolution_axis,
            rebuild_axis,
            save_axis,
        )
        self._control_buttons = (
            self.reset_button,
            self.zoom_button,
            self.resolution_button,
            self.rebuild_button,
            self.save_button,
        )
        self._layout_controls()

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
        self._sync_zoom_slider_to_current_view()
        self.figure.canvas.draw_idle()
        self._schedule_full_resolution_restore()

    def _on_resize(self, _event):
        self._layout_controls()
        self._show_interaction_preview()
        self._schedule_full_resolution_restore()

    def _on_button_press(self, event):
        self._dismiss_transient_controls_for_event(event)
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
            if (
                hasattr(self, "zoom_slider_axis")
                and event.inaxes is self.zoom_slider_axis
            ):
                self._schedule_full_resolution_restore()
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

    def _dismiss_transient_controls_for_event(self, event):
        if (
            self._zoom_slider_visible
            and event.inaxes not in (
                self.zoom_slider_axis,
                self.zoom_button.ax,
            )
        ):
            self._set_zoom_slider_visible(False)

        resolution_axes = [self.resolution_button.ax]
        resolution_axes.extend(self._resolution_option_axes)
        if (
            self._resolution_popup_visible
            and event.inaxes not in resolution_axes
        ):
            self._set_resolution_popup_visible(False)

    def _layout_controls(self):
        figure_width = max(float(self.figure.bbox.width), 1.0)
        figure_height = max(float(self.figure.bbox.height), 1.0)
        button_y = 0.035
        button_height = max(0.06, 30.0 / figure_height)
        slider_gap = 8.0 / figure_height
        slider_height = 16.0 / figure_height

        for button_axis in self._control_button_axes:
            position = button_axis.get_position()
            button_axis.set_position(
                [position.x0, button_y, position.width, button_height]
            )

        button_top = button_y + button_height
        slider_y = button_top + slider_gap
        self.zoom_slider_axis.set_position(
            [0.18, slider_y, 0.64, slider_height]
        )

        resolution_position = self.resolution_button.ax.get_position()
        option_height_pixels = min(
            48.0,
            max(34.0, 0.042 * figure_height),
        )
        option_height = option_height_pixels / figure_height
        popup_width = max(
            200.0 / figure_width,
            min(resolution_position.width, 280.0 / figure_width),
        )
        popup_padding_x = 6.0 / figure_width
        popup_padding_y = 6.0 / figure_height
        option_width = popup_width - 2.0 * popup_padding_x
        popup_height = (
            len(self._resolution_option_axes) * option_height
            + 2.0 * popup_padding_y
        )
        popup_x = min(
            1.0 - popup_width - 4.0 / figure_width,
            max(
                4.0 / figure_width,
                resolution_position.x0
                + 0.5 * (resolution_position.width - popup_width),
            ),
        )
        popup_y = resolution_position.y1 + slider_gap
        for option_index, option_axis in enumerate(
            self._resolution_option_axes
        ):
            option_axis.set_position(
                [
                    popup_x + popup_padding_x,
                    popup_y
                    + popup_padding_y
                    + (
                        len(self._resolution_option_axes)
                        - option_index
                        - 1
                    ) * option_height,
                    option_width,
                    option_height,
                ]
            )
        self._resolution_popup_panel.set_bounds(
            popup_x,
            popup_y,
            popup_width,
            popup_height,
        )
        self._resolution_popup_shadow.set_bounds(
            popup_x + 2.0 / figure_width,
            popup_y - 2.0 / figure_height,
            popup_width,
            popup_height,
        )

        plot_top = min(0.93, 1.0 - 35.0 / figure_height)
        self.figure.subplots_adjust(
            left=0.09,
            bottom=0.17,
            right=0.88,
            top=plot_top,
        )
        plot_position = self.axis.get_position()
        self.axis.set_position(
            (
                (1.0 - plot_position.width) / 2.0,
                plot_position.y0,
                plot_position.width,
                plot_position.height,
            )
        )
        self._apply_adaptive_button_fonts()

    def _apply_adaptive_button_fonts(self):
        figure_height = max(float(self.figure.bbox.height), 1.0)
        default_figure_height = (
            self.INITIAL_FIGURE_SIZE[1] * self.FIGURE_DPI
        )

        button_height_pixels = max(0.06 * figure_height, 30.0)
        default_button_height_pixels = 0.06 * default_figure_height
        button_font_size = min(
            13.0,
            max(
                8.0,
                10.0 * button_height_pixels / default_button_height_pixels,
            ),
        )
        for button in self._control_buttons:
            button.label.set_fontsize(button_font_size)

        option_height_pixels = max(0.042 * figure_height, 24.0)
        default_option_height_pixels = 0.042 * default_figure_height
        option_font_size = min(
            13.0,
            max(
                8.0,
                10.0 * option_height_pixels / default_option_height_pixels,
            ),
        )
        for (
            option_button,
            _width,
            _height,
            selection_indicator,
        ) in self._resolution_option_buttons:
            option_button.label.set_fontsize(option_font_size)
            selection_indicator.set_fontsize(option_font_size + 1.0)

    def _resolution_button_label(self):
        return "Resolution"

    def _style_resolution_option(
        self,
        option_button,
        selection_indicator,
        is_selected,
    ):
        if is_selected:
            facecolor = "#ecfeff"
            hovercolor = "#cffafe"
            text_color = "#0e7490"
        else:
            facecolor = "#ffffff"
            hovercolor = "#f1f5f9"
            text_color = "#334155"

        option_button.color = facecolor
        option_button.hovercolor = hovercolor
        option_button.ax.set_facecolor(facecolor)
        for spine in option_button.ax.spines.values():
            spine.set_visible(False)
        option_button.label.set_color(text_color)
        option_button.label.set_fontweight(
            "bold" if is_selected else "normal"
        )
        option_button.label.set_horizontalalignment("left")
        option_button.label.set_position((0.08, 0.5))
        selection_indicator.set_visible(is_selected)

    def _create_resolution_popup(self):
        self._resolution_popup_visible = False
        self._resolution_option_axes = []
        self._resolution_option_buttons = []

        self._resolution_popup_shadow = FancyBboxPatch(
            (0.0, 0.0),
            0.0,
            0.0,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=self.figure.transFigure,
            facecolor="#0f172a",
            edgecolor="none",
            alpha=0.14,
            zorder=18,
            visible=False,
        )
        self.figure.add_artist(self._resolution_popup_shadow)
        self._resolution_popup_panel = FancyBboxPatch(
            (0.0, 0.0),
            0.0,
            0.0,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=self.figure.transFigure,
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=0.8,
            zorder=19,
            visible=False,
        )
        self.figure.add_artist(self._resolution_popup_panel)

        for width, height in self.BEV_RESOLUTIONS:
            option_axis = self.figure.add_axes(
                [0.0, 0.0, 0.1, 0.042]
            )
            option_axis.set_zorder(20)
            is_selected = (
                width == self.selected_bev_width
                and height == self.selected_bev_height
            )
            option_button = Button(
                option_axis,
                "{} × {}".format(width, height),
                color="#ffffff",
                hovercolor="#f1f5f9",
            )
            selection_indicator = option_axis.text(
                0.90,
                0.5,
                "✓",
                transform=option_axis.transAxes,
                color="#0891b2",
                fontsize=11,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
            )
            self._style_resolution_option(
                option_button,
                selection_indicator,
                is_selected,
            )
            option_button.on_clicked(
                lambda _event, selected_width=width, selected_height=height:
                self._select_bev_resolution(selected_width, selected_height)
            )
            option_axis.set_visible(False)
            self._resolution_option_axes.append(option_axis)
            self._resolution_option_buttons.append(
                (
                    option_button,
                    width,
                    height,
                    selection_indicator,
                )
            )

    def _toggle_resolution_popup(self, _event):
        visible = not self._resolution_popup_visible
        if visible:
            self._set_zoom_slider_visible(False)
        self._set_resolution_popup_visible(visible)

    def _set_resolution_popup_visible(self, visible):
        self._resolution_popup_visible = bool(visible)
        self._resolution_popup_shadow.set_visible(
            self._resolution_popup_visible
        )
        self._resolution_popup_panel.set_visible(
            self._resolution_popup_visible
        )
        for option_axis in self._resolution_option_axes:
            option_axis.set_visible(self._resolution_popup_visible)
        self.figure.canvas.draw_idle()

    def _select_bev_resolution(self, width, height):
        self.selected_bev_width = width
        self.selected_bev_height = height
        self._update_resolution_option_styles()
        self._set_resolution_popup_visible(False)
        rospy.loginfo(
            "Selected BEV rebuild resolution: %d x %d; "
            "click Rebuild BEV to apply",
            width,
            height,
        )

    def _update_resolution_option_styles(self):
        for (
            option_button,
            width,
            height,
            selection_indicator,
        ) in self._resolution_option_buttons:
            is_selected = (
                width == self.selected_bev_width
                and height == self.selected_bev_height
            )
            self._style_resolution_option(
                option_button,
                selection_indicator,
                is_selected,
            )
        self._apply_adaptive_button_fonts()

    def _create_zoom_slider(self):
        self._zoom_slider_visible = False
        self._updating_zoom_slider = False
        self.zoom_slider_axis = self.figure.add_axes(
            [0.18, 0.085, 0.64, 0.022]
        )
        self.zoom_slider_axis.set_zorder(20)
        self.zoom_slider = Slider(
            self.zoom_slider_axis,
            "Zoom",
            0.0,
            1.0,
            valinit=self._view_width_to_slider_value(self.full_view_width),
            valfmt="%1.2f",
            color="#059669",
            track_color="#d1d5db",
            initcolor="none",
            handle_style={
                "facecolor": "#059669",
                "edgecolor": "#065f46",
                "size": 6,
            },
        )
        self.zoom_slider.track.set_visible(False)
        self.zoom_slider.poly.set_visible(False)
        self.zoom_slider.vline.set_visible(False)
        (self._zoom_slider_track_line,) = self.zoom_slider_axis.plot(
            [0.0, 1.0],
            [0.5, 0.5],
            transform=self.zoom_slider_axis.transAxes,
            color="#d1d5db",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
        (self._zoom_slider_fill_line,) = self.zoom_slider_axis.plot(
            [0.0, self.zoom_slider.val],
            [0.5, 0.5],
            transform=self.zoom_slider_axis.transAxes,
            color="#059669",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=2,
        )
        self.zoom_slider._handle.set_zorder(3)
        self.zoom_slider_axis.set_visible(False)
        self._sync_zoom_slider_to_current_view()
        self.zoom_slider.on_changed(self._zoom_slider_changed)

    def _toggle_zoom_slider(self, _event):
        visible = not self._zoom_slider_visible
        if visible:
            self._set_resolution_popup_visible(False)
        self._set_zoom_slider_visible(visible)

    def _set_zoom_slider_visible(self, visible):
        self._zoom_slider_visible = bool(visible)
        self.zoom_slider_axis.set_visible(self._zoom_slider_visible)
        if self._zoom_slider_visible:
            self._sync_zoom_slider_to_current_view()
        self.figure.canvas.draw_idle()

    def _view_width_to_slider_value(self, view_width):
        bounded_width = min(
            self.max_view_width,
            max(self.min_view_width, float(view_width)),
        )
        logarithmic_range = math.log(
            self.max_view_width / self.min_view_width
        )
        return math.log(self.max_view_width / bounded_width) / logarithmic_range

    def _slider_value_to_view_width(self, slider_value):
        value = min(1.0, max(0.0, float(slider_value)))
        logarithmic_range = math.log(
            self.max_view_width / self.min_view_width
        )
        return self.max_view_width * math.exp(-value * logarithmic_range)

    def _zoom_slider_changed(self, slider_value):
        if self._updating_zoom_slider:
            return

        self._update_zoom_slider_fill(slider_value)

        x_min, x_max = self.axis.get_xlim()
        y_min, y_max = self.axis.get_ylim()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        new_width = self._slider_value_to_view_width(slider_value)
        new_height = new_width / self.BEV_ASPECT_RATIO

        self._show_interaction_preview()
        self._set_axis_limits(
            (
                center_x - new_width / 2.0,
                center_x + new_width / 2.0,
            ),
            (
                center_y - new_height / 2.0,
                center_y + new_height / 2.0,
            ),
        )
        self._set_zoom_slider_value_text(new_width)
        self.figure.canvas.draw_idle()
        self._schedule_full_resolution_restore()

    def _sync_zoom_slider_to_current_view(self):
        if not hasattr(self, "zoom_slider"):
            return

        x_min, x_max = self.axis.get_xlim()
        view_width = abs(x_max - x_min)
        slider_value = self._view_width_to_slider_value(view_width)
        if not math.isclose(
            self.zoom_slider.val,
            slider_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            self._updating_zoom_slider = True
            try:
                self.zoom_slider.set_val(slider_value)
            finally:
                self._updating_zoom_slider = False
        self._update_zoom_slider_fill(slider_value)
        self._set_zoom_slider_value_text(view_width)

    def _update_zoom_slider_fill(self, slider_value):
        value = min(1.0, max(0.0, float(slider_value)))
        self._zoom_slider_fill_line.set_xdata([0.0, value])

    def _set_zoom_slider_value_text(self, view_width):
        zoom_factor = self.full_view_width / float(view_width)
        self.zoom_slider.valtext.set_text("{:.2f}x".format(zoom_factor))

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
        if (
            self._drag_state is not None
            or (
                hasattr(self, "zoom_slider")
                and self.zoom_slider.drag_active
            )
        ):
            return
        self._show_full_resolution()
        self.figure.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key and event.key.lower() == "r":
            self._restore_full_bev()

    def _reset_clicked(self, _event):
        self._restore_full_bev()

    def _save_clicked(self, _event):
        self.save_button.label.set_text("Saving...")
        self.figure.canvas.draw()
        try:
            output_path = self._next_export_path()
            self._save_bev_image(output_path)
        except (OSError, ValueError) as error:
            rospy.logerr("Could not save PCD BEV image: %s", error)
        else:
            rospy.loginfo("Saved PCD BEV image: %s", output_path)
        finally:
            self.save_button.label.set_text("Save Image")
            self.figure.canvas.draw_idle()

    def _next_export_path(self):
        filename_stem = self.pcd_path.stem

        candidate = self.export_dir / "{}.png".format(filename_stem)
        suffix = 1
        while candidate.exists():
            candidate = self.export_dir / "{}_{}.png".format(
                filename_stem,
                suffix,
            )
            suffix += 1
        return candidate

    def _save_bev_image(self, output_path):
        export_figure = self._create_export_figure()
        try:
            export_figure.savefig(
                str(output_path),
                dpi=self.EXPORT_DPI,
                format="png",
                facecolor="#ffffff",
            )
        finally:
            export_figure.clear()

    def _create_export_figure(self):
        export_figure = Figure(
            figsize=self.INITIAL_FIGURE_SIZE,
            dpi=self.EXPORT_DPI,
        )
        FigureCanvasAgg(export_figure)
        export_axis = export_figure.subplots()
        export_figure.subplots_adjust(
            left=0.07,
            bottom=0.09,
            right=0.93,
            top=0.91,
        )
        plot_position = export_axis.get_position()
        export_axis.set_position(
            (
                (1.0 - plot_position.width) / 2.0,
                plot_position.y0,
                plot_position.width,
                plot_position.height,
            )
        )

        colormap = copy.copy(plt.get_cmap(self.colormap))
        colormap.set_bad(color="#f3f4f6", alpha=1.0)
        export_image = export_axis.imshow(
            self._full_image_grid,
            origin="lower",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
            cmap=colormap,
            norm=self._normalization_for_bev(self.bev),
            aspect="equal",
        )
        self._draw_trajectory(export_axis)
        export_axis.set_anchor("C")
        export_colorbar_axis = inset_axes(
            export_axis,
            width="2.5%",
            height="100%",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
            bbox_transform=export_axis.transAxes,
            borderpad=0.0,
        )
        export_colorbar = export_figure.colorbar(
            export_image,
            cax=export_colorbar_axis,
        )
        export_colorbar.set_label("Maximum Z in pixel (m), fixed P1-P99")

        export_axis.set_title(self.title, pad=self.TITLE_PAD_POINTS)
        export_axis.set_xlabel("X (m)")
        export_axis.set_ylabel("Y (m)")
        export_axis.grid(
            True,
            color="#ffffff",
            linestyle="--",
            linewidth=0.45,
            alpha=0.35,
        )
        export_axis.set_xlim(self.axis.get_xlim())
        export_axis.set_ylim(self.axis.get_ylim())
        return export_figure

    def _draw_trajectory(self, axis):
        if self.trajectory_positions is None:
            return None

        (line,) = axis.plot(
            self.trajectory_positions[:, 0],
            self.trajectory_positions[:, 1],
            color="#FBBC05",
            linewidth=1.5,
            zorder=5,
        )
        return line

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
                grid_width=self.selected_bev_width,
                grid_height=self.selected_bev_height,
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
            "%d/%d sampled points in view, %d occupied pixels, "
            "%d raster threads",
            label,
            bev["grid_width"],
            bev["grid_height"],
            bev["resolution"],
            bev["voxel_size"],
            bev["visible_points"],
            bev["sampled_points"],
            bev["occupied_pixels"],
            bev["raster_workers"],
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
        self._sync_zoom_slider_to_current_view()

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
