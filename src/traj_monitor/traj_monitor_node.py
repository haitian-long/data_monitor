#!/usr/bin/env python3

import math
import os
from pathlib import Path
import sys


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
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import rospy

from pcd_monitor.pcd import (
    PcdError,
    build_bev,
    extent_for_bounds,
    load_point_cloud,
    read_tum_trajectory_poses,
    validate_pcd_file,
)
from traj_monitor.toolbar_menus import (
    attach_toolbar_dropdown,
    create_dropdown_content,
    create_dropdown_panel,
    hide_dropdown,
    raise_visible_dropdowns,
)


class TrajMonitor:
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
    INTERACTION_RESTORE_DELAY_MS = 180
    MIN_VIEW_VOXELS = 8.0
    MAX_FULL_VIEW_SCALE = 4.0
    ZOOM_SLIDER_STEPS = 10000
    TITLE_PAD_POINTS = 15.0
    EXPORT_DPI = 200
    UI_BACKGROUND = "#ffffff"
    UI_SURFACE = "#ffffff"
    UI_BORDER = "#cbd5e1"
    UI_TEXT = "#0f172a"
    UI_MUTED_TEXT = "#64748b"
    PLOT_EMPTY_COLOR = "#f3f4f6"
    PLOT_POINT_COLOR = "#a8a8a8"
    MAIN_TRAJECTORY_COLOR = "#DC2626"
    MAIN_TRAJECTORY_LINEWIDTH = 3.0
    OTHER_TRAJECTORY_LINEWIDTH = 2.2
    ALIGN_MOTION_START_M = 2.0
    ALIGN_WINDOW_M = 20.0
    ALIGN_MAX_TIME_DIFF = 0.05
    ALIGN_MIN_PAIRS = 8
    LEGEND_TITLE_FONTSIZE = 16.0
    LEGEND_ITEM_FONTSIZE = 14.0
    LEGEND_HANDLELENGTH = 2.4
    LEGEND_HANDLEHEIGHT = 0.7
    LEGEND_BORDERPAD = 0.35
    LEGEND_LABELSPACING = 0.18
    LEGEND_HANDLETEXTPAD = 0.45
    OTHER_TRAJECTORY_COLORS = (
        "#1D4ED8",
        "#15803D",
        "#EA580C",
        "#CA8A04",
        "#6D28D9",
        "#0891B2",
        "#9A3412",
        "#0F172A",
    )
    OURS_TRAJECTORY_COLOR = "#DC2626"
    GRID_COLOR = "#94a3b8"
    START_MARKER_COLOR = "#FACC15"
    START_MARKER_SIZE = 9.0
    START_MARKER_LABEL = "start"

    def __init__(self):
        pcd_path_param = str(rospy.get_param("~pcd_path", "")).strip()
        if not pcd_path_param:
            raise PcdError(
                "~pcd_path is empty; set it to a single PCD file"
            )
        self.pcd_path = validate_pcd_file(pcd_path_param)
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.10))
        self.title = str(rospy.get_param("~title", "Trajectory BEV Monitor"))
        self.trajectories = self._load_trajectories()
        self.selected_bev_width = self.BEV_WIDTH
        self.selected_bev_height = self.BEV_HEIGHT

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
            "Source Z percentile %.0f%%..%.0f%% = %.6g..%.6g m",
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
        self._build_bev_image(self.bev)
        self.full_bev = self.bev
        self._log_bev_info("Initial BEV", self.bev)

        self._drag_state = None
        self._overlay_view_limits = None
        self._interaction_active = False
        self._create_figure()

    def _create_figure(self):
        self.figure, self.axis = plt.subplots(
            figsize=self.INITIAL_FIGURE_SIZE,
            dpi=self.FIGURE_DPI,
        )
        self.figure.patch.set_facecolor(self.UI_BACKGROUND)
        self.figure.canvas.manager.set_window_title(self.title)
        self._configure_resizable_window()
        self.figure.subplots_adjust(left=0.08, bottom=0.10, right=0.90, top=0.93)

        self.image = self.axis.imshow(
            self.bev["rgba_image"],
            origin="upper",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
            aspect="equal",
        )
        self.trajectory_lines = self._draw_trajectories(self.axis)
        self.start_marker = self._draw_start_marker(self.axis)
        self.axis.set_anchor("C")
        self.legend = self._draw_legend(self.axis)

        self._set_plot_title()
        self.axis.set_xlabel("X (m)")
        self.axis.set_ylabel("Y (m)")
        self.axis.grid(True, color=self.GRID_COLOR, linestyle="--", linewidth=0.45, alpha=0.45)
        self._style_plot_chrome(self.axis, self.legend)
        self._set_axis_extent(self.bev["extent"])

        self._create_native_toolbar()
        self._layout_figure()
        self._create_interaction_overlay()
        self._bev_qpixmap(self.bev)

        canvas = self.figure.canvas
        self._install_canvas_draw_guards(canvas)
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
            self._finish_interaction
        )
        # Prepare the hidden Matplotlib buffer once, then keep the Qt-rendered
        # plot layer visible for the entire lifetime of the window.
        canvas.draw()
        self._show_persistent_overlay()
        rospy.loginfo(
            "Persistent Qt BEV renderer enabled; Matplotlib remains "
            "synchronized in the background for export"
        )
        rospy.loginfo(
            "Zoom X-span limits: %.6g..%.6g m; cached-image interaction "
            "commit delay: %d ms",
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

        data_xy = self._data_xy_from_event(event)
        if data_xy is None:
            return
        data_x, data_y = data_xy
        (x_min, x_max), (y_min, y_max) = self._current_view_limits()
        relative_x = (data_x - x_min) / (x_max - x_min)
        relative_y = (data_y - y_min) / (y_max - y_min)
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
        self._apply_overlay_view(
            (
                data_x - relative_x * new_width,
                data_x + (1.0 - relative_x) * new_width,
            ),
            (
                data_y - relative_y * new_height,
                data_y + (1.0 - relative_y) * new_height,
            ),
        )

    def _on_resize(self, _event):
        self._cancel_interaction()
        self._layout_figure()
        self.figure.canvas.draw_idle()
        self._update_interaction_overlay_geometry()
        self._interaction_overlay.raise_()
        self._raise_toolbar_dropdowns()
        self._interaction_overlay.update()

    def _on_button_press(self, event):
        if self._any_toolbar_menu_visible():
            self._hide_toolbar_menus()
            return
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
            x_limits, y_limits = self._current_view_limits()
            self._drag_state = {
                "start_pixel": (float(event.x), float(event.y)),
                "x_limits": x_limits,
                "y_limits": y_limits,
                "axis_pixels": (
                    max(float(self.axis.bbox.width), 1.0),
                    max(float(self.axis.bbox.height), 1.0),
                ),
            }
            self._interaction_restore_timer.stop()
            self._begin_interaction()

    def _on_motion(self, event):
        if (
            self._drag_state is None
            or event.x is None
            or event.y is None
        ):
            return

        self._overlay_view_limits = self._drag_limits_for_event(event)
        self._update_interaction_overlay()

    def _on_button_release(self, event):
        if self._drag_state is None:
            return

        if event.x is not None and event.y is not None:
            self._overlay_view_limits = self._drag_limits_for_event(event)
        self._drag_state = None
        self._finish_interaction(force=True)

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

    def _current_view_limits(self):
        overlay = getattr(self, "_overlay_view_limits", None)
        if overlay is not None:
            return overlay
        return (self.axis.get_xlim(), self.axis.get_ylim())

    def _data_xy_from_event(self, event):
        (x_min, x_max), (y_min, y_max) = self._current_view_limits()
        data_rect = getattr(self, "_interaction_data_rect", None)
        overlay = getattr(self, "_interaction_overlay", None)
        if (
            event.x is None
            or event.y is None
            or data_rect is None
            or overlay is None
            or data_rect.width() <= 0.0
            or data_rect.height() <= 0.0
        ):
            if event.xdata is None or event.ydata is None:
                return None
            return float(event.xdata), float(event.ydata)

        local_x = float(event.x) - float(overlay.x())
        local_y = (
            float(self.figure.canvas.height()) - float(event.y)
        ) - float(overlay.y())
        if not (
            data_rect.left() <= local_x <= data_rect.right()
            and data_rect.top() <= local_y <= data_rect.bottom()
        ):
            return None
        data_x = x_min + (
            (local_x - data_rect.left())
            / data_rect.width()
            * (x_max - x_min)
        )
        data_y = y_max - (
            (local_y - data_rect.top())
            / data_rect.height()
            * (y_max - y_min)
        )
        return data_x, data_y

    def _apply_overlay_view(self, x_limits, y_limits, sync_slider=True):
        self._begin_interaction()
        self._overlay_view_limits = (tuple(x_limits), tuple(y_limits))
        if sync_slider:
            self._sync_zoom_slider_to_current_view()
        self._update_interaction_overlay()
        self._schedule_interaction_finish()

    def _commit_overlay_view(self):
        if self._overlay_view_limits is None:
            return
        self._set_axis_limits(*self._overlay_view_limits)
        self._overlay_view_limits = None

    def _set_plot_title(self):
        self.axis.set_title(
            self.title,
            pad=self.TITLE_PAD_POINTS,
        )

    def _style_plot_chrome(self, axis, legend):
        for spine in axis.spines.values():
            spine.set_color(self.UI_BORDER)
            spine.set_linewidth(0.8)
        axis.tick_params(
            axis="both",
            colors=self.UI_MUTED_TEXT,
            labelsize=9,
            length=3.5,
            width=0.8,
        )
        axis.xaxis.label.set_color("#475569")
        axis.yaxis.label.set_color("#475569")
        axis.xaxis.label.set_fontsize(10)
        axis.yaxis.label.set_fontsize(10)
        axis.title.set_color(self.UI_TEXT)
        axis.title.set_fontsize(14)
        axis.title.set_fontweight("normal")
        self._style_legend(legend, figure=axis.figure)

    def _ui_scale(self, figure=None):
        figure = self.figure if figure is None else figure
        height_inches = float(figure.get_size_inches()[1])
        return max(height_inches / self.INITIAL_FIGURE_SIZE[1], 0.75)

    def _style_legend(self, legend, figure=None):
        if legend is None:
            return
        legend.set_zorder(20)
        scale = self._ui_scale(figure)
        frame = legend.get_frame()
        frame.set_facecolor("#ffffff")
        frame.set_edgecolor(self.UI_BORDER)
        frame.set_linewidth(0.8 * scale)
        title = legend.get_title()
        if title.get_text():
            title.set_color(self.UI_TEXT)
            title.set_fontsize(self.LEGEND_TITLE_FONTSIZE * scale)
            title.set_fontweight("bold")
        for text in legend.get_texts():
            text.set_color(self.UI_TEXT)
            text.set_fontsize(self.LEGEND_ITEM_FONTSIZE * scale)

    def _refresh_legend(self):
        if getattr(self, "legend", None) is not None:
            self.legend.remove()
            self.legend = None
        if not hasattr(self, "axis"):
            return
        self.legend = self._draw_legend(self.axis)
        self._style_legend(self.legend, figure=self.figure)

    def _create_native_toolbar(self):
        try:
            from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
        except ImportError as error:
            raise PcdError(
                "The native traj_monitor toolbar requires a Qt Matplotlib backend"
            ) from error

        manager = self.figure.canvas.manager
        toolbar = getattr(manager, "toolbar", None)
        window = getattr(manager, "window", None)
        if (
            toolbar is None
            or window is None
            or not isinstance(toolbar, QtWidgets.QToolBar)
        ):
            raise PcdError(
                "The native traj_monitor toolbar requires FigureManagerQT"
            )

        qt_namespace = QtCore.Qt
        tool_bar_area = getattr(
            qt_namespace,
            "ToolBarArea",
            qt_namespace,
        )
        top_tool_bar_area = getattr(
            tool_bar_area,
            "TopToolBarArea",
        )
        tool_button_style = getattr(
            qt_namespace,
            "ToolButtonStyle",
            qt_namespace,
        )
        text_only_style = getattr(
            tool_button_style,
            "ToolButtonTextOnly",
        )
        orientation = getattr(
            qt_namespace,
            "Orientation",
            qt_namespace,
        )
        horizontal_orientation = getattr(
            orientation,
            "Horizontal",
        )
        alignment = getattr(
            qt_namespace,
            "AlignmentFlag",
            qt_namespace,
        )
        right_alignment = (
            getattr(alignment, "AlignRight")
            | getattr(alignment, "AlignVCenter")
        )
        window.removeToolBar(toolbar)
        window.addToolBar(top_tool_bar_area, toolbar)
        toolbar.clear()
        toolbar.setObjectName("trajMonitorToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(text_only_style)

        self._reset_action = toolbar.addAction("Reset view")
        self._reset_action.setToolTip("Restore the initial full BEV view")
        self._reset_action.triggered.connect(
            lambda _checked=False: self._reset_clicked(None)
        )

        self._zoom_tool_button = QtWidgets.QToolButton(toolbar)
        self._zoom_tool_button.setObjectName("pcdMenuToolButton")
        self._zoom_tool_button.setText("Zoom")
        self._zoom_tool_button.setToolTip(
            "Adjust the BEV zoom factor; the mouse wheel remains available"
        )
        zoom_menu = create_dropdown_panel(window, QtCore, QtWidgets)
        zoom_content = create_dropdown_content(zoom_menu, QtWidgets)
        zoom_content.setObjectName("pcdZoomContent")
        zoom_layout = QtWidgets.QHBoxLayout(zoom_content)
        zoom_layout.setContentsMargins(16, 10, 16, 10)
        zoom_layout.setSpacing(10)

        mouse_buttons = getattr(
            qt_namespace,
            "MouseButton",
            qt_namespace,
        )
        left_mouse_button = getattr(mouse_buttons, "LeftButton")
        center_alignment = getattr(alignment, "AlignCenter")
        pen_styles = getattr(qt_namespace, "PenStyle", qt_namespace)
        no_pen = getattr(pen_styles, "NoPen")
        render_hints = getattr(
            QtGui.QPainter,
            "RenderHint",
            QtGui.QPainter,
        )
        antialiasing = getattr(render_hints, "Antialiasing")

        class RepeatingZoomLabel(QtWidgets.QLabel):
            """Keep QLabel rendering while adding click/hold behavior."""

            def __init__(label_self, text, callback, tooltip):
                super().__init__(text, zoom_content)
                label_self._callback = callback
                label_self._left_mouse_button = left_mouse_button
                label_self._hovered = False
                label_self._pressed = False
                label_self._repeat_delay = QtCore.QTimer(label_self)
                label_self._repeat_delay.setSingleShot(True)
                label_self._repeat_delay.setInterval(350)
                label_self._repeat_delay.timeout.connect(
                    label_self._start_repeat
                )
                label_self._repeat_timer = QtCore.QTimer(label_self)
                label_self._repeat_timer.setInterval(60)
                label_self._repeat_timer.timeout.connect(callback)
                label_self.setObjectName("pcdZoomBoundLabel")
                label_self.setToolTip(tooltip)
                label_self.setAlignment(center_alignment)
                label_self.setFixedSize(24, 24)

            def _start_repeat(label_self):
                label_self._callback()
                label_self._repeat_timer.start()

            def _stop_repeat(label_self):
                label_self._repeat_delay.stop()
                label_self._repeat_timer.stop()

            def mousePressEvent(label_self, event):
                if event.button() == label_self._left_mouse_button:
                    label_self._pressed = True
                    label_self.update()
                    label_self._callback()
                    label_self._repeat_delay.start()
                    event.accept()
                    return
                super().mousePressEvent(event)

            def mouseReleaseEvent(label_self, event):
                label_self._stop_repeat()
                label_self._pressed = False
                label_self.update()
                if event.button() == label_self._left_mouse_button:
                    event.accept()
                    return
                super().mouseReleaseEvent(event)

            def enterEvent(label_self, event):
                label_self._hovered = True
                label_self.update()
                super().enterEvent(event)

            def leaveEvent(label_self, event):
                label_self._stop_repeat()
                label_self._hovered = False
                label_self._pressed = False
                label_self.update()
                super().leaveEvent(event)

            def paintEvent(label_self, event):
                if label_self._hovered or label_self._pressed:
                    painter = QtGui.QPainter(label_self)
                    painter.setRenderHint(antialiasing, True)
                    painter.setPen(no_pen)
                    painter.setBrush(
                        QtGui.QColor(
                            "#dbe4ff"
                            if label_self._pressed
                            else "#eef2ff"
                        )
                    )
                    painter.drawEllipse(
                        label_self.rect().adjusted(1, 1, -1, -1)
                    )
                    painter.end()
                super().paintEvent(event)

        self._qt_zoom_out_button = RepeatingZoomLabel(
            "−",
            lambda: self._adjust_zoom_factor(-0.01),
            "Decrease zoom by 0.01×",
        )
        self._qt_zoom_in_button = RepeatingZoomLabel(
            "+",
            lambda: self._adjust_zoom_factor(0.01),
            "Increase zoom by 0.01×",
        )
        self._qt_zoom_slider = QtWidgets.QSlider(
            horizontal_orientation,
            zoom_content,
        )
        self._qt_zoom_slider.setObjectName("pcdZoomSlider")
        self._qt_zoom_slider.setRange(0, self.ZOOM_SLIDER_STEPS)
        self._qt_zoom_slider.setSingleStep(1)
        self._qt_zoom_slider.setPageStep(100)
        self._qt_zoom_slider.setMinimumWidth(420)
        self._qt_zoom_value_label = QtWidgets.QLabel(zoom_content)
        self._qt_zoom_value_label.setObjectName("pcdZoomValue")
        self._qt_zoom_value_label.setFixedWidth(58)
        self._qt_zoom_value_label.setAlignment(right_alignment)

        zoom_layout.addWidget(self._qt_zoom_out_button)
        zoom_layout.addWidget(self._qt_zoom_slider, 1)
        zoom_layout.addWidget(self._qt_zoom_in_button)
        zoom_layout.addWidget(self._qt_zoom_value_label)
        zoom_content.setMinimumWidth(560)

        zoom_menu.layout().addWidget(zoom_content)
        attach_toolbar_dropdown(
            self._zoom_tool_button,
            zoom_menu,
            on_show=self._sync_zoom_slider_to_current_view,
            on_hide=self._schedule_deferred_plot_chrome,
            QtCore=QtCore,
            sibling_panels=self._toolbar_menus,
        )
        toolbar.addWidget(self._zoom_tool_button)
        self._qt_zoom_menu = zoom_menu

        self._resolution_tool_button = QtWidgets.QToolButton(toolbar)
        self._resolution_tool_button.setObjectName("pcdMenuToolButton")
        self._resolution_tool_button.setText("Resolution")
        self._resolution_tool_button.setToolTip(
            "Select the resolution used by the next BEV rebuild"
        )
        resolution_menu = create_dropdown_panel(window, QtCore, QtWidgets)
        resolution_content = create_dropdown_content(resolution_menu, QtWidgets)
        resolution_content.setObjectName("pcdResolutionContent")
        resolution_layout = QtWidgets.QVBoxLayout(resolution_content)
        resolution_layout.setContentsMargins(8, 6, 8, 6)
        resolution_layout.setSpacing(2)
        self._resolution_actions = {}
        for width, height in self.BEV_RESOLUTIONS:
            choice = QtWidgets.QToolButton(resolution_content)
            choice.setObjectName("pcdResolutionChoice")
            choice.setText("{} × {}".format(width, height))
            choice.setCheckable(True)
            choice.setAutoExclusive(True)
            choice.setChecked(
                width == self.selected_bev_width
                and height == self.selected_bev_height
            )
            choice.clicked.connect(
                lambda _checked=False,
                selected_width=width,
                selected_height=height:
                self._select_bev_resolution(
                    selected_width,
                    selected_height,
                )
            )
            resolution_layout.addWidget(choice)
            self._resolution_actions[(width, height)] = choice
        resolution_menu.layout().addWidget(resolution_content)
        attach_toolbar_dropdown(
            self._resolution_tool_button,
            resolution_menu,
            QtCore=QtCore,
            sibling_panels=self._toolbar_menus,
        )
        toolbar.addWidget(self._resolution_tool_button)
        self._qt_resolution_menu = resolution_menu

        self._filter_tool_button = QtWidgets.QToolButton(toolbar)
        self._filter_tool_button.setObjectName("pcdMenuToolButton")
        self._filter_tool_button.setText("Filter")
        self._filter_tool_button.setToolTip(
            "Show or hide trajectories in the plot and legend"
        )
        filter_menu = create_dropdown_panel(window, QtCore, QtWidgets)
        filter_content = create_dropdown_content(filter_menu, QtWidgets)
        filter_content.setObjectName("pcdFilterContent")
        filter_layout = QtWidgets.QVBoxLayout(filter_content)
        filter_layout.setContentsMargins(12, 8, 16, 8)
        filter_layout.setSpacing(4)
        self._filter_checkboxes = []
        if self.trajectories:
            for traj in self.trajectories:
                checkbox = QtWidgets.QCheckBox(traj["label"], filter_content)
                checkbox.setObjectName("pcdFilterCheckBox")
                checkbox.setChecked(bool(traj.get("visible", True)))
                checkbox.toggled.connect(
                    lambda checked, target=traj: self._set_trajectory_visible(
                        target,
                        bool(checked),
                    )
                )
                filter_layout.addWidget(checkbox)
                self._filter_checkboxes.append(checkbox)
        else:
            empty_label = QtWidgets.QLabel("No trajectories", filter_content)
            filter_layout.addWidget(empty_label)
        filter_menu.layout().addWidget(filter_content)
        attach_toolbar_dropdown(
            self._filter_tool_button,
            filter_menu,
            on_hide=self._schedule_deferred_plot_chrome,
            QtCore=QtCore,
            sibling_panels=self._toolbar_menus,
        )
        toolbar.addWidget(self._filter_tool_button)
        self._qt_filter_menu = filter_menu

        self._color_tool_button = QtWidgets.QToolButton(toolbar)
        self._color_tool_button.setObjectName("pcdMenuToolButton")
        self._color_tool_button.setText("Color")
        self._color_tool_button.setToolTip(
            "Edit trajectory colors by hex value or color picker"
        )
        color_menu = create_dropdown_panel(window, QtCore, QtWidgets)
        color_content = create_dropdown_content(color_menu, QtWidgets)
        color_content.setObjectName("pcdColorContent")
        color_layout = QtWidgets.QVBoxLayout(color_content)
        color_layout.setContentsMargins(12, 8, 16, 8)
        color_layout.setSpacing(6)
        self._color_editors = []
        self._color_pick_target = None
        if self.trajectories:
            for traj in self.trajectories:
                row = QtWidgets.QWidget(color_content)
                row_layout = QtWidgets.QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                name_label = QtWidgets.QLabel(traj["label"], row)
                name_label.setObjectName("pcdColorName")
                name_label.setMinimumWidth(88)
                color_edit = QtWidgets.QLineEdit(traj["color"], row)
                color_edit.setObjectName("pcdColorEdit")
                color_edit.setFixedWidth(86)
                swatch = QtWidgets.QToolButton(row)
                swatch.setObjectName("pcdColorSwatch")
                swatch.setFixedSize(22, 22)
                swatch.setToolTip("Choose this trajectory, then pick a color below")
                self._style_color_swatch(swatch, traj["color"])
                color_edit.editingFinished.connect(
                    lambda editor=color_edit, target=traj, button=swatch:
                    self._set_trajectory_color(
                        target,
                        editor.text(),
                        editor,
                        button,
                    )
                )
                swatch.clicked.connect(
                    lambda _checked=False, target=traj, editor=color_edit, button=swatch:
                    self._select_color_target(target, editor, button)
                )
                row_layout.addWidget(name_label)
                row_layout.addWidget(color_edit)
                row_layout.addWidget(swatch)
                color_layout.addWidget(row)
                self._color_editors.append((traj, color_edit, swatch))
            first_traj, first_edit, first_swatch = self._color_editors[0]
            self._select_color_target(first_traj, first_edit, first_swatch)
            palette = self._build_inline_color_palette(color_content)
            color_layout.addWidget(palette)
        else:
            empty_label = QtWidgets.QLabel("No trajectories", color_content)
            color_layout.addWidget(empty_label)
        color_menu.layout().addWidget(color_content)
        attach_toolbar_dropdown(
            self._color_tool_button,
            color_menu,
            on_hide=self._schedule_deferred_plot_chrome,
            QtCore=QtCore,
            sibling_panels=self._toolbar_menus,
        )
        toolbar.addWidget(self._color_tool_button)
        self._qt_color_menu = color_menu

        toolbar.addSeparator()
        self._save_action = toolbar.addAction("Save PNG")
        self._save_action.setToolTip(
            "Save the current BEV, title and trajectory legend as a PNG beside the PCD file"
        )
        self._save_action.triggered.connect(
            lambda _checked=False: self._save_clicked(None)
        )

        self._save_tiff_action = toolbar.addAction("Save TIFF")
        self._save_tiff_action.setToolTip(
            "Save the current BEV, title and trajectory legend as a TIFF beside the PCD file"
        )
        self._save_tiff_action.triggered.connect(
            lambda _checked=False: self._save_tiff_clicked(None)
        )

        self._rebuild_action = toolbar.addAction("Rebuild BEV")
        self._rebuild_action.setToolTip(
            "Rebuild at the selected resolution for the current view"
        )
        self._rebuild_action.triggered.connect(
            lambda _checked=False: self._rebuild_clicked(None)
        )
        rebuild_widget = toolbar.widgetForAction(self._rebuild_action)
        if rebuild_widget is not None:
            rebuild_widget.setObjectName("pcdPrimaryToolButton")

        toolbar.setStyleSheet(
            """
            QToolBar#trajMonitorToolbar {
                background: #ffffff;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                spacing: 4px;
                padding: 6px 10px;
            }
            QToolBar#trajMonitorToolbar QToolButton {
                color: #334155;
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 7px 11px;
                font-weight: 600;
            }
            QToolBar#trajMonitorToolbar QToolButton:hover,
            QToolBar#trajMonitorToolbar QToolButton:pressed {
                color: #4338ca;
                background: #eef2ff;
            }
            QToolBar#trajMonitorToolbar
            QToolButton#pcdMenuToolButton::menu-indicator {
                image: none;
                width: 0px;
                height: 0px;
            }
            QToolBar#trajMonitorToolbar QToolButton#pcdPrimaryToolButton {
                color: #ffffff;
                background: #4f46e5;
                padding-left: 15px;
                padding-right: 15px;
            }
            QToolBar#trajMonitorToolbar
            QToolButton#pcdPrimaryToolButton:hover,
            QToolBar#trajMonitorToolbar
            QToolButton#pcdPrimaryToolButton:pressed {
                color: #ffffff;
                background: #4338ca;
            }
            QToolBar#trajMonitorToolbar QToolBarSeparator {
                background: #e2e8f0;
                width: 1px;
                margin: 7px 6px;
            }
            QMenu#pcdMonitorMenu {
                color: #334155;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 6px;
            }
            QMenu#pcdMonitorMenu::item {
                border-radius: 6px;
                padding: 8px 28px 8px 10px;
            }
            QMenu#pcdMonitorMenu::item:selected {
                color: #4338ca;
                background: #eef2ff;
            }
            QWidget#pcdZoomContent,
            QWidget#pcdFilterContent,
            QWidget#pcdColorContent {
                background: #ffffff;
            }
            QFrame#pcdColorPanel {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
            }
            QLabel#pcdColorName {
                color: #334155;
                font-weight: 600;
            }
            QLineEdit#pcdColorEdit {
                color: #334155;
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 3px 6px;
                font-family: monospace;
            }
            QToolButton#pcdColorSwatch {
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding: 0px;
            }
            QToolButton#pcdColorSwatch:hover {
                border: 1px solid #4338ca;
            }
            QCheckBox#pcdFilterCheckBox {
                color: #334155;
                spacing: 8px;
                padding: 3px 4px;
                font-weight: 600;
            }
            QCheckBox#pcdFilterCheckBox:hover {
                color: #4338ca;
            }
            QLabel#pcdZoomBoundLabel {
                color: #64748b;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#pcdZoomValue {
                color: #475569;
                font-weight: 600;
            }
            QSlider#pcdZoomSlider::groove:horizontal {
                height: 4px;
                background: #d8dee8;
                border-radius: 2px;
            }
            QSlider#pcdZoomSlider::sub-page:horizontal {
                background: #4f46e5;
                border-radius: 2px;
            }
            QSlider#pcdZoomSlider::add-page:horizontal {
                background: #d8dee8;
                border-radius: 2px;
            }
            QSlider#pcdZoomSlider::handle:horizontal {
                width: 14px;
                margin: -5px 0;
                background: #ffffff;
                border: 2px solid #4f46e5;
                border-radius: 7px;
            }
            """
        )
        toolbar.setVisible(True)
        self._native_toolbar = toolbar

        self._updating_zoom_slider = True
        try:
            slider_value = self._view_width_to_slider_value(
                abs(self.axis.get_xlim()[1] - self.axis.get_xlim()[0])
            )
            self._qt_zoom_slider.setValue(
                int(round(slider_value * self.ZOOM_SLIDER_STEPS))
            )
        finally:
            self._updating_zoom_slider = False
        self._set_zoom_slider_value_text(
            abs(self.axis.get_xlim()[1] - self.axis.get_xlim()[0])
        )
        self._qt_zoom_slider.valueChanged.connect(
            self._zoom_slider_changed
        )
        self._qt_zoom_slider.sliderReleased.connect(
            self._zoom_slider_released
        )

        self._install_window_escape_closes_menus(window, QtCore, QtWidgets)
        rospy.loginfo(
            "Installed native Qt toolbar: Reset | Zoom | Resolution | "
            "Filter | Color | Save PNG | Save TIFF | Rebuild"
        )

    def _install_window_escape_closes_menus(self, window, QtCore, QtWidgets):
        from matplotlib.backends.qt_compat import QtGui
        from traj_monitor.toolbar_menus import _escape_key

        escape = _escape_key(QtCore)
        key_sequence = getattr(QtGui, "QKeySequence", None) or getattr(
            QtWidgets,
            "QKeySequence",
            None,
        )
        shortcut_cls = getattr(QtWidgets, "QShortcut", None)
        if escape is None or key_sequence is None or shortcut_cls is None:
            return
        shortcut = shortcut_cls(key_sequence(escape), window)
        context = getattr(
            getattr(QtCore.Qt, "ShortcutContext", QtCore.Qt),
            "ApplicationShortcut",
            getattr(QtCore.Qt, "ApplicationShortcut", None),
        )
        if context is not None:
            shortcut.setContext(context)
        shortcut.activated.connect(self._hide_toolbar_menus)
        self._escape_menu_shortcut = shortcut

    def _hide_toolbar_menus(self):
        hidden = False
        for menu in self._toolbar_menus():
            if hide_dropdown(menu):
                hidden = True
        return hidden

    def _raise_toolbar_dropdowns(self):
        raise_visible_dropdowns(self._toolbar_menus())

    def _toolbar_menus(self):
        return (
            getattr(self, "_qt_zoom_menu", None),
            getattr(self, "_qt_filter_menu", None),
            getattr(self, "_qt_color_menu", None),
            getattr(self, "_qt_resolution_menu", None),
        )

    def _any_toolbar_menu_visible(self):
        return any(
            menu is not None and menu.isVisible()
            for menu in self._toolbar_menus()
        )

    def _request_plot_chrome_refresh(self):
        self._plot_chrome_refresh_pending = True
        if hasattr(self, "_interaction_overlay"):
            self._interaction_overlay.update()
        if self._any_toolbar_menu_visible():
            return
        self._schedule_deferred_plot_chrome()

    def _schedule_deferred_plot_chrome(self):
        from matplotlib.backends.qt_compat import QtCore

        if not getattr(self, "_plot_chrome_refresh_pending", False):
            return
        timer = getattr(self, "_plot_chrome_flush_timer", None)
        if timer is None:
            timer = QtCore.QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_deferred_plot_chrome)
            self._plot_chrome_flush_timer = timer
        timer.start(0)

    def _flush_deferred_plot_chrome(self):
        if not getattr(self, "_plot_chrome_refresh_pending", False):
            return
        if self._any_toolbar_menu_visible():
            return
        self._plot_chrome_refresh_pending = False
        if hasattr(self, "_interaction_overlay"):
            self._interaction_overlay.update()

    def _layout_figure(self):
        figure_height = max(float(self.figure.bbox.height), 1.0)
        plot_bottom = min(
            0.14,
            max(0.09, 76.0 / figure_height),
        )
        plot_top = min(0.93, 1.0 - 35.0 / figure_height)
        self.figure.subplots_adjust(
            left=0.08,
            bottom=plot_bottom,
            right=0.90,
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
        self._refresh_legend()

    def _create_interaction_overlay(self):
        from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets

        monitor = self

        class PersistentCachedBevOverlay(QtWidgets.QWidget):
            def __init__(self, parent):
                super().__init__(parent)
                widget_attributes = getattr(
                    QtCore.Qt,
                    "WidgetAttribute",
                    QtCore.Qt,
                )
                self.setAttribute(
                    getattr(widget_attributes, "WA_TransparentForMouseEvents"),
                    True,
                )
                self.setAttribute(
                    getattr(widget_attributes, "WA_OpaquePaintEvent"),
                    True,
                )
                self.hide()

            def paintEvent(self, _event):
                painter = QtGui.QPainter(self)
                monitor._paint_interaction_overlay(painter, self)
                painter.end()

        self._qt_core = QtCore
        self._qt_gui = QtGui
        self._interaction_overlay = PersistentCachedBevOverlay(
            self.figure.canvas
        )
        self._interaction_data_rect = QtCore.QRectF()
        self._trajectory_qitems = self._build_trajectory_qitems()

        alignments = getattr(
            QtCore.Qt,
            "AlignmentFlag",
            QtCore.Qt,
        )
        self._qt_align_center = getattr(alignments, "AlignCenter")
        self._qt_align_right_center = (
            getattr(alignments, "AlignRight")
            | getattr(alignments, "AlignVCenter")
        )
        self._qt_align_bottom_center = (
            getattr(alignments, "AlignHCenter")
            | getattr(alignments, "AlignBottom")
        )

    def _update_interaction_overlay_geometry(self):
        canvas = self.figure.canvas
        figure_width = max(float(self.figure.bbox.width), 1.0)
        figure_height = max(float(self.figure.bbox.height), 1.0)
        scale_x = float(canvas.width()) / figure_width
        scale_y = float(canvas.height()) / figure_height

        axis_left = float(self.axis.bbox.x0) * scale_x
        axis_right = float(self.axis.bbox.x1) * scale_x
        axis_top = (
            figure_height - float(self.axis.bbox.y1)
        ) * scale_y
        axis_bottom = (
            figure_height - float(self.axis.bbox.y0)
        ) * scale_y

        renderer = canvas.get_renderer()
        tight_bbox = self.axis.get_tightbbox(renderer)
        tight_left = float(tight_bbox.x0) * scale_x
        tight_right = float(tight_bbox.x1) * scale_x
        tight_top = (
            figure_height - float(tight_bbox.y1)
        ) * scale_y
        tight_bottom = (
            figure_height - float(tight_bbox.y0)
        ) * scale_y

        overlay_left = max(
            0.0,
            min(tight_left - 6.0, axis_left - 100.0),
        )
        overlay_top = max(
            0.0,
            min(tight_top - 6.0, axis_top - 50.0),
        )
        overlay_right = min(
            float(canvas.width()),
            max(tight_right + 6.0, axis_right + 4.0),
        )
        overlay_bottom = min(
            float(canvas.height()),
            max(tight_bottom + 6.0, axis_bottom + 64.0),
        )

        geometry = self._qt_core.QRect(
            int(math.floor(overlay_left)),
            int(math.floor(overlay_top)),
            max(1, int(math.ceil(overlay_right - overlay_left))),
            max(1, int(math.ceil(overlay_bottom - overlay_top))),
        )
        self._interaction_overlay.setGeometry(geometry)
        self._interaction_data_rect = self._qt_core.QRectF(
            axis_left - geometry.x(),
            axis_top - geometry.y(),
            axis_right - axis_left,
            axis_bottom - axis_top,
        )

    def _paint_interaction_overlay(self, painter, widget):
        painter.fillRect(
            widget.rect(),
            self._qt_gui.QColor(self.UI_BACKGROUND),
        )
        data_rect = self._interaction_data_rect
        if data_rect.width() <= 0.0 or data_rect.height() <= 0.0:
            return

        painter.fillRect(
            data_rect,
            self._qt_gui.QColor(self.PLOT_EMPTY_COLOR),
        )
        render_hints = getattr(
            self._qt_gui.QPainter,
            "RenderHint",
            self._qt_gui.QPainter,
        )
        painter.setRenderHint(
            getattr(render_hints, "SmoothPixmapTransform"),
            False,
        )
        self._draw_cached_bev_image(painter, data_rect)
        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )
        x_ticks, x_labels, x_offset = self._interaction_axis_ticks(
            self.axis.xaxis,
            view_min_x,
            view_max_x,
        )
        y_ticks, y_labels, y_offset = self._interaction_axis_ticks(
            self.axis.yaxis,
            view_min_y,
            view_max_y,
        )
        self._draw_interaction_grid(
            painter,
            data_rect,
            x_ticks,
            y_ticks,
        )
        self._draw_interaction_trajectories(painter, data_rect)
        self._draw_interaction_start_marker(painter, data_rect)
        if not self._interaction_active:
            self._draw_interaction_legend(painter, data_rect)
        self._draw_interaction_axes(
            painter,
            data_rect,
            x_ticks,
            x_labels,
            x_offset,
            y_ticks,
            y_labels,
            y_offset,
        )

    def _draw_cached_bev_image(self, painter, data_rect):
        pixmap = self._bev_qpixmap(self.bev)
        image_min_x, image_max_x, image_min_y, image_max_y = self.bev[
            "extent"
        ]
        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )
        visible_min_x = max(image_min_x, view_min_x)
        visible_max_x = min(image_max_x, view_max_x)
        visible_min_y = max(image_min_y, view_min_y)
        visible_max_y = min(image_max_y, view_max_y)
        if (
            visible_max_x <= visible_min_x
            or visible_max_y <= visible_min_y
        ):
            return

        view_width = view_max_x - view_min_x
        view_height = view_max_y - view_min_y
        image_width = image_max_x - image_min_x
        image_height = image_max_y - image_min_y
        target_rect = self._qt_core.QRectF(
            data_rect.left()
            + (visible_min_x - view_min_x)
            / view_width
            * data_rect.width(),
            data_rect.top()
            + (view_max_y - visible_max_y)
            / view_height
            * data_rect.height(),
            (visible_max_x - visible_min_x)
            / view_width
            * data_rect.width(),
            (visible_max_y - visible_min_y)
            / view_height
            * data_rect.height(),
        )
        source_rect = self._qt_core.QRectF(
            (visible_min_x - image_min_x)
            / image_width
            * pixmap.width(),
            (image_max_y - visible_max_y)
            / image_height
            * pixmap.height(),
            (visible_max_x - visible_min_x)
            / image_width
            * pixmap.width(),
            (visible_max_y - visible_min_y)
            / image_height
            * pixmap.height(),
        )
        painter.drawPixmap(target_rect, pixmap, source_rect)

    def _bev_qpixmap(self, bev):
        cached_pixmap = bev.get("qt_pixmap")
        if cached_pixmap is not None:
            return cached_pixmap

        rgba = bev["rgba_image"]
        image_formats = getattr(
            self._qt_gui.QImage,
            "Format",
            self._qt_gui.QImage,
        )
        image = self._qt_gui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            rgba.strides[0],
            getattr(image_formats, "Format_RGBA8888"),
        )
        pixmap = self._qt_gui.QPixmap.fromImage(image)
        bev["qt_pixmap"] = pixmap
        return pixmap

    def _interaction_axis_ticks(self, matplotlib_axis, vmin=None, vmax=None):
        if vmin is None or vmax is None:
            locations = np.asarray(
                matplotlib_axis.get_majorticklocs(),
                dtype=np.float64,
            )
        else:
            locator = matplotlib_axis.get_major_locator()
            try:
                locations = np.asarray(
                    locator.tick_values(float(vmin), float(vmax)),
                    dtype=np.float64,
                )
            except (AttributeError, TypeError, ValueError):
                locations = np.asarray(
                    matplotlib_axis.get_majorticklocs(),
                    dtype=np.float64,
                )
        formatter = matplotlib_axis.get_major_formatter()
        formatter.set_locs(locations)
        labels = [
            formatter(float(location), index)
            for index, location in enumerate(locations)
        ]
        offset = ""
        if callable(getattr(formatter, "get_offset", None)):
            offset = formatter.get_offset()
        return locations, labels, offset

    def _draw_interaction_grid(
        self,
        painter,
        data_rect,
        x_ticks,
        y_ticks,
    ):
        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )
        grid_color = self._qt_gui.QColor(self.GRID_COLOR)
        grid_color.setAlphaF(0.45)
        grid_pen = self._qt_gui.QPen(grid_color)
        grid_pen.setWidthF(0.45)
        pen_styles = getattr(
            self._qt_core.Qt,
            "PenStyle",
            self._qt_core.Qt,
        )
        grid_pen.setStyle(getattr(pen_styles, "DashLine"))
        painter.save()
        painter.setClipRect(data_rect)
        painter.setPen(grid_pen)
        for value in x_ticks:
            if view_min_x <= value <= view_max_x:
                pixel_x = data_rect.left() + (
                    (value - view_min_x)
                    / (view_max_x - view_min_x)
                    * data_rect.width()
                )
                painter.drawLine(
                    self._qt_core.QPointF(pixel_x, data_rect.top()),
                    self._qt_core.QPointF(pixel_x, data_rect.bottom()),
                )
        for value in y_ticks:
            if view_min_y <= value <= view_max_y:
                pixel_y = data_rect.bottom() - (
                    (value - view_min_y)
                    / (view_max_y - view_min_y)
                    * data_rect.height()
                )
                painter.drawLine(
                    self._qt_core.QPointF(data_rect.left(), pixel_y),
                    self._qt_core.QPointF(data_rect.right(), pixel_y),
                )
        painter.restore()

    def _positions_to_qpath(self, positions):
        path = self._qt_gui.QPainterPath()
        path_started = False
        for x_value, y_value in positions[:, :2]:
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                path_started = False
                continue
            if path_started:
                path.lineTo(float(x_value), float(y_value))
            else:
                path.moveTo(float(x_value), float(y_value))
                path_started = True
        return path

    def _build_trajectory_qitems(self):
        return [
            (self._positions_to_qpath(traj["positions"]), traj)
            for traj in self._draw_order_trajectories()
        ]

    def _draw_interaction_trajectories(self, painter, data_rect):
        if not self._trajectory_qitems:
            return

        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )
        scale_x = data_rect.width() / (view_max_x - view_min_x)
        scale_y = data_rect.height() / (view_max_y - view_min_y)
        transform = self._qt_gui.QTransform(
            scale_x,
            0.0,
            0.0,
            -scale_y,
            data_rect.left() - scale_x * view_min_x,
            data_rect.top() + scale_y * view_max_y,
        )
        pen_styles = getattr(
            self._qt_core.Qt,
            "PenStyle",
            self._qt_core.Qt,
        )
        interacting = bool(self._interaction_active or self._drag_state)
        painter.save()
        painter.setClipRect(data_rect)
        painter.setTransform(transform, True)
        for qpath, traj in self._trajectory_qitems:
            if not traj.get("visible", True):
                continue
            trajectory_pen = self._qt_gui.QPen(
                self._qt_gui.QColor(traj["color"])
            )
            trajectory_pen.setWidthF(traj["linewidth"])
            trajectory_pen.setCosmetic(True)
            if interacting or traj["linestyle"] != "--":
                trajectory_pen.setStyle(getattr(pen_styles, "SolidLine"))
            else:
                trajectory_pen.setStyle(getattr(pen_styles, "DashLine"))
            painter.setPen(trajectory_pen)
            painter.drawPath(qpath)
        painter.restore()

    def _triangle_path(self, center, radius):
        path = self._qt_gui.QPainterPath()
        half_width = radius * math.sqrt(3.0) / 2.0
        path.moveTo(
            self._qt_core.QPointF(center.x(), center.y() - radius)
        )
        path.lineTo(
            self._qt_core.QPointF(
                center.x() - half_width,
                center.y() + radius / 2.0,
            )
        )
        path.lineTo(
            self._qt_core.QPointF(
                center.x() + half_width,
                center.y() + radius / 2.0,
            )
        )
        path.closeSubpath()
        return path

    def _draw_interaction_start_marker(self, painter, data_rect):
        if self.start_xy is None:
            return
        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )
        start_x, start_y = self.start_xy
        if not (
            view_min_x <= start_x <= view_max_x
            and view_min_y <= start_y <= view_max_y
        ):
            return
        pixel_x = data_rect.left() + (
            (start_x - view_min_x)
            / (view_max_x - view_min_x)
            * data_rect.width()
        )
        pixel_y = data_rect.bottom() - (
            (start_y - view_min_y)
            / (view_max_y - view_min_y)
            * data_rect.height()
        )
        radius = 7.0 * self._ui_scale()
        center = self._qt_core.QPointF(pixel_x, pixel_y)
        fill = self._qt_gui.QColor(self.START_MARKER_COLOR)
        pen_styles = getattr(
            self._qt_core.Qt,
            "PenStyle",
            self._qt_core.Qt,
        )
        painter.save()
        painter.setClipRect(data_rect)
        painter.setRenderHint(
            getattr(
                getattr(
                    self._qt_gui.QPainter,
                    "RenderHint",
                    self._qt_gui.QPainter,
                ),
                "Antialiasing",
            ),
            True,
        )
        painter.setPen(getattr(pen_styles, "NoPen"))
        painter.setBrush(fill)
        painter.drawPath(self._triangle_path(center, radius))
        painter.restore()

    def _draw_interaction_legend(self, painter, data_rect):
        visible = self._visible_trajectories()
        if not visible and self.start_xy is None:
            return

        scale = self._ui_scale()
        font = self._qt_gui.QFont()
        font.setPointSizeF(self.LEGEND_ITEM_FONTSIZE * scale)
        metrics = self._qt_gui.QFontMetricsF(font)
        pad = 6.0 * scale
        margin = 8.0 * scale
        gap = 5.0 * scale
        line_len = 22.0 * scale
        row_height = max(float(metrics.height()) * 1.12, 14.0 * scale)
        text_width = 0.0
        legend_labels = [traj["label"] for traj in visible]
        if self.start_xy is not None:
            legend_labels.append(self.START_MARKER_LABEL)
        for label in legend_labels:
            text_width = max(
                text_width,
                float(metrics.horizontalAdvance(label)),
            )
        box_width = pad + line_len + gap + text_width + pad
        box_height = pad + row_height * len(legend_labels) + pad
        box = self._qt_core.QRectF(
            data_rect.right() - margin - box_width,
            data_rect.top() + margin,
            box_width,
            box_height,
        )
        box = box.intersected(data_rect)
        if box.width() <= 8.0 or box.height() <= 8.0:
            return

        fill = self._qt_gui.QColor("#ffffff")
        fill.setAlphaF(0.92)
        border_pen = self._qt_gui.QPen(self._qt_gui.QColor(self.UI_BORDER))
        border_pen.setWidthF(max(0.8 * scale, 1.0))
        brush_styles = getattr(
            self._qt_core.Qt,
            "BrushStyle",
            self._qt_core.Qt,
        )
        pen_styles = getattr(
            self._qt_core.Qt,
            "PenStyle",
            self._qt_core.Qt,
        )
        alignments = getattr(
            self._qt_core.Qt,
            "AlignmentFlag",
            self._qt_core.Qt,
        )
        align_left_center = (
            getattr(alignments, "AlignLeft")
            | getattr(alignments, "AlignVCenter")
        )

        painter.save()
        painter.setRenderHint(
            getattr(
                getattr(
                    self._qt_gui.QPainter,
                    "RenderHint",
                    self._qt_gui.QPainter,
                ),
                "Antialiasing",
            ),
            True,
        )
        painter.setPen(border_pen)
        painter.setBrush(fill)
        painter.drawRect(box)

        painter.setFont(font)
        y = box.top() + pad
        for traj in visible:
            center_y = y + row_height / 2.0
            line_left = box.left() + pad
            line_right = line_left + line_len
            line_pen = self._qt_gui.QPen(self._qt_gui.QColor(traj["color"]))
            line_pen.setWidthF(max(traj["linewidth"] * scale, 1.8))
            line_pen.setCosmetic(True)
            if traj["linestyle"] == "--":
                line_pen.setStyle(getattr(pen_styles, "DashLine"))
            else:
                line_pen.setStyle(getattr(pen_styles, "SolidLine"))
            painter.setPen(line_pen)
            painter.drawLine(
                self._qt_core.QPointF(line_left, center_y),
                self._qt_core.QPointF(line_right, center_y),
            )
            painter.setPen(self._qt_gui.QColor(self.UI_TEXT))
            painter.setBrush(getattr(brush_styles, "NoBrush"))
            text_rect = self._qt_core.QRectF(
                line_right + gap,
                y,
                text_width,
                row_height,
            )
            painter.drawText(text_rect, align_left_center, traj["label"])
            y += row_height
        if self.start_xy is not None:
            center_y = y + row_height / 2.0
            triangle_center = self._qt_core.QPointF(
                box.left() + pad + line_len / 2.0,
                center_y,
            )
            painter.setPen(getattr(pen_styles, "NoPen"))
            painter.setBrush(self._qt_gui.QColor(self.START_MARKER_COLOR))
            painter.drawPath(
                self._triangle_path(triangle_center, 6.0 * scale)
            )
            painter.setPen(self._qt_gui.QColor(self.UI_TEXT))
            painter.setBrush(getattr(brush_styles, "NoBrush"))
            text_rect = self._qt_core.QRectF(
                box.left() + pad + line_len + gap,
                y,
                text_width,
                row_height,
            )
            painter.drawText(
                text_rect,
                align_left_center,
                self.START_MARKER_LABEL,
            )
        painter.restore()

    def _draw_interaction_axes(
        self,
        painter,
        data_rect,
        x_ticks,
        x_labels,
        x_offset,
        y_ticks,
        y_labels,
        y_offset,
    ):
        painter.setRenderHint(
            getattr(
                getattr(
                    self._qt_gui.QPainter,
                    "RenderHint",
                    self._qt_gui.QPainter,
                ),
                "TextAntialiasing",
            ),
            True,
        )
        border_pen = self._qt_gui.QPen(
            self._qt_gui.QColor(self.UI_BORDER)
        )
        border_pen.setWidthF(0.8)
        painter.setPen(border_pen)
        brush_styles = getattr(
            self._qt_core.Qt,
            "BrushStyle",
            self._qt_core.Qt,
        )
        painter.setBrush(getattr(brush_styles, "NoBrush"))
        painter.drawRect(data_rect)

        tick_font = self._qt_gui.QFont()
        tick_font.setPointSizeF(9.0)
        painter.setFont(tick_font)
        painter.setPen(self._qt_gui.QColor(self.UI_MUTED_TEXT))
        tick_metrics = self._qt_gui.QFontMetricsF(tick_font)
        tick_length = 4.0
        (view_min_x, view_max_x), (view_min_y, view_max_y) = (
            self._current_view_limits()
        )

        for value, label in zip(x_ticks, x_labels):
            if not view_min_x <= value <= view_max_x:
                continue
            pixel_x = data_rect.left() + (
                (value - view_min_x)
                / (view_max_x - view_min_x)
                * data_rect.width()
            )
            painter.drawLine(
                self._qt_core.QPointF(pixel_x, data_rect.bottom()),
                self._qt_core.QPointF(
                    pixel_x,
                    data_rect.bottom() + tick_length,
                ),
            )
            label_width = max(tick_metrics.horizontalAdvance(label) + 8.0, 48.0)
            painter.drawText(
                self._qt_core.QRectF(
                    pixel_x - label_width / 2.0,
                    data_rect.bottom() + tick_length + 2.0,
                    label_width,
                    tick_metrics.height() + 3.0,
                ),
                self._qt_align_center,
                label,
            )

        y_label_right = data_rect.left() - tick_length - 4.0
        y_label_width = max(data_rect.left() - 22.0, 20.0)
        for value, label in zip(y_ticks, y_labels):
            if not view_min_y <= value <= view_max_y:
                continue
            pixel_y = data_rect.bottom() - (
                (value - view_min_y)
                / (view_max_y - view_min_y)
                * data_rect.height()
            )
            painter.drawLine(
                self._qt_core.QPointF(data_rect.left(), pixel_y),
                self._qt_core.QPointF(
                    data_rect.left() - tick_length,
                    pixel_y,
                ),
            )
            painter.drawText(
                self._qt_core.QRectF(
                    y_label_right - y_label_width,
                    pixel_y - tick_metrics.height() / 2.0,
                    y_label_width,
                    tick_metrics.height(),
                ),
                self._qt_align_right_center,
                label,
            )

        offset_font = self._qt_gui.QFont(tick_font)
        offset_font.setPointSizeF(8.5)
        painter.setFont(offset_font)
        if x_offset:
            painter.drawText(
                self._qt_core.QRectF(
                    data_rect.right() - 100.0,
                    data_rect.bottom() + tick_metrics.height() + 7.0,
                    100.0,
                    tick_metrics.height(),
                ),
                self._qt_align_right_center,
                x_offset,
            )
        if y_offset:
            painter.drawText(
                self._qt_core.QRectF(
                    data_rect.left(),
                    data_rect.top() - tick_metrics.height() - 2.0,
                    100.0,
                    tick_metrics.height(),
                ),
                self._qt_align_right_center,
                y_offset,
            )

        label_font = self._qt_gui.QFont()
        label_font.setPointSizeF(10.0)
        painter.setFont(label_font)
        painter.setPen(self._qt_gui.QColor("#475569"))
        painter.drawText(
            self._qt_core.QRectF(
                data_rect.left(),
                data_rect.bottom() + tick_metrics.height() + 10.0,
                data_rect.width(),
                24.0,
            ),
            self._qt_align_center,
            "X (m)",
        )
        painter.save()
        painter.translate(
            max(10.0, data_rect.left() - 42.0),
            data_rect.center().y(),
        )
        painter.rotate(-90.0)
        painter.drawText(
            self._qt_core.QRectF(
                -data_rect.height() / 2.0,
                -12.0,
                data_rect.height(),
                24.0,
            ),
            self._qt_align_center,
            "Y (m)",
        )
        painter.restore()

        title_font = self._qt_gui.QFont()
        title_font.setPointSizeF(14.0)
        title_font.setBold(False)
        painter.setFont(title_font)
        painter.setPen(self._qt_gui.QColor(self.UI_TEXT))
        title_gap = (
            self.TITLE_PAD_POINTS
            * float(self._interaction_overlay.logicalDpiY())
            / 72.0
        )
        painter.drawText(
            self._qt_core.QRectF(
                data_rect.left(),
                0.0,
                data_rect.width(),
                max(1.0, data_rect.top() - title_gap),
            ),
            self._qt_align_bottom_center,
            self.title,
        )

    def _install_canvas_draw_guards(self, canvas):
        self._raw_canvas_draw = canvas.draw
        self._raw_canvas_draw_idle = canvas.draw_idle

        def _guarded_draw(*args, **kwargs):
            if self._interaction_active:
                return None
            return self._raw_canvas_draw(*args, **kwargs)

        def _guarded_draw_idle(*args, **kwargs):
            if self._interaction_active:
                return None
            return self._raw_canvas_draw_idle(*args, **kwargs)

        canvas.draw = _guarded_draw
        canvas.draw_idle = _guarded_draw_idle

    def _begin_interaction(self):
        if self._interaction_active:
            return
        self._interaction_restore_timer.stop()
        self._bev_qpixmap(self.bev)
        self._interaction_active = True
        self._interaction_overlay.update()

    def _show_persistent_overlay(self):
        self._bev_qpixmap(self.bev)
        self._update_interaction_overlay_geometry()
        self._interaction_overlay.show()
        self._interaction_overlay.raise_()
        self._raise_toolbar_dropdowns()
        self._interaction_overlay.update()

    def _update_interaction_overlay(self):
        if not self._interaction_active:
            self._begin_interaction()
        self._interaction_overlay.update()

    def _schedule_interaction_finish(self):
        self._interaction_restore_timer.stop()
        self._interaction_restore_timer.start()

    def _finish_interaction(self, draw=True, force=False):
        if not self._interaction_active:
            return False
        if (
            not force
            and hasattr(self, "_qt_zoom_slider")
            and self._qt_zoom_slider.isSliderDown()
        ):
            return False

        self._interaction_restore_timer.stop()
        self._commit_overlay_view()
        self._interaction_active = False
        if draw:
            # Synchronize Matplotlib behind the persistent Qt plot layer.
            # The screen never switches renderers when interaction ends.
            self._raw_canvas_draw()
        self._interaction_overlay.raise_()
        self._raise_toolbar_dropdowns()
        self._interaction_overlay.update()
        return True

    def _cancel_interaction(self):
        if hasattr(self, "_interaction_restore_timer"):
            self._interaction_restore_timer.stop()
        self._interaction_active = False
        self._overlay_view_limits = None
        if hasattr(self, "_interaction_overlay"):
            self._interaction_overlay.raise_()
            self._raise_toolbar_dropdowns()
            self._interaction_overlay.update()

    def _select_bev_resolution(self, width, height):
        self.selected_bev_width = width
        self.selected_bev_height = height
        for resolution, action in self._resolution_actions.items():
            action.setChecked(resolution == (width, height))
        hide_dropdown(getattr(self, "_qt_resolution_menu", None))
        rospy.loginfo(
            "Selected BEV rebuild resolution: %d x %d; "
            "click Rebuild BEV to apply",
            width,
            height,
        )

    def _zoom_slider_released(self):
        self._finish_interaction(force=True)

    def _adjust_zoom_factor(self, delta):
        (x_min, x_max), (y_min, y_max) = self._current_view_limits()
        current_width = abs(x_max - x_min)
        current_zoom = self.full_view_width / current_width
        # Start from the same two-decimal value shown beside the slider so
        # every click visibly changes that value by exactly 0.01x.
        displayed_zoom = round(current_zoom * 100.0) / 100.0
        min_zoom = self.full_view_width / self.max_view_width
        max_zoom = self.full_view_width / self.min_view_width
        new_zoom = min(
            max_zoom,
            max(min_zoom, displayed_zoom + float(delta)),
        )
        new_width = self.full_view_width / new_zoom
        if math.isclose(new_width, current_width, rel_tol=1e-12):
            return

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        new_height = new_width / self.BEV_ASPECT_RATIO
        self._apply_overlay_view(
            (
                center_x - new_width / 2.0,
                center_x + new_width / 2.0,
            ),
            (
                center_y - new_height / 2.0,
                center_y + new_height / 2.0,
            ),
        )

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

    def _zoom_slider_changed(self, slider_position):
        if self._updating_zoom_slider:
            return

        slider_value = (
            float(slider_position) / float(self.ZOOM_SLIDER_STEPS)
        )

        (x_min, x_max), (y_min, y_max) = self._current_view_limits()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        new_width = self._slider_value_to_view_width(slider_value)
        new_height = new_width / self.BEV_ASPECT_RATIO
        self._apply_overlay_view(
            (
                center_x - new_width / 2.0,
                center_x + new_width / 2.0,
            ),
            (
                center_y - new_height / 2.0,
                center_y + new_height / 2.0,
            ),
            sync_slider=False,
        )
        self._set_zoom_slider_value_text(new_width)

    def _sync_zoom_slider_to_current_view(self):
        if not hasattr(self, "_qt_zoom_slider"):
            return

        (x_min, x_max), _y_limits = self._current_view_limits()
        view_width = abs(x_max - x_min)
        slider_value = self._view_width_to_slider_value(view_width)
        slider_position = int(
            round(slider_value * self.ZOOM_SLIDER_STEPS)
        )
        if self._qt_zoom_slider.value() != slider_position:
            self._updating_zoom_slider = True
            try:
                self._qt_zoom_slider.setValue(slider_position)
            finally:
                self._updating_zoom_slider = False
        self._set_zoom_slider_value_text(view_width)

    def _set_zoom_slider_value_text(self, view_width):
        zoom_factor = self.full_view_width / float(view_width)
        self._qt_zoom_value_label.setText(
            "{:.2f}×".format(zoom_factor)
        )

    def _on_key_press(self, event):
        if event.key and event.key.lower() == "escape":
            self._hide_toolbar_menus()
            return
        if event.key and event.key.lower() == "r":
            self._restore_full_bev()

    def _reset_clicked(self, _event):
        self._restore_full_bev()

    def _save_clicked(self, _event):
        self._export_bev_image("png", self._save_action, "Save PNG")

    def _save_tiff_clicked(self, _event):
        self._export_bev_image("tiff", self._save_tiff_action, "Save TIFF")

    def _export_bev_image(self, image_format, action, idle_label):
        self._finish_interaction(force=True)
        action.setText("Saving…")
        action.setEnabled(False)
        self.figure.canvas.flush_events()
        try:
            output_path = self._next_export_path(image_format)
            self._save_bev_image(output_path, image_format)
        except (OSError, ValueError) as error:
            rospy.logerr("Could not save PCD BEV image: %s", error)
        else:
            rospy.loginfo("Saved PCD BEV image: %s", output_path)
        finally:
            action.setText(idle_label)
            action.setEnabled(True)

    def _next_export_path(self, image_format="png"):
        filename_stem = self.pcd_path.stem
        extension = image_format.lower()
        candidate = self.export_dir / "{}.{}".format(filename_stem, extension)
        suffix = 1
        while candidate.exists():
            candidate = self.export_dir / "{}_{}.{}".format(
                filename_stem,
                suffix,
                extension,
            )
            suffix += 1
        return candidate

    def _save_bev_image(self, output_path, image_format="png"):
        export_figure = self._create_export_figure()
        save_kwargs = {
            "dpi": self.EXPORT_DPI,
            "format": image_format,
            "facecolor": "#ffffff",
        }
        if image_format in ("tif", "tiff"):
            save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        try:
            export_figure.savefig(str(output_path), **save_kwargs)
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
        export_axis.imshow(
            self.bev["rgba_image"],
            origin="upper",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
            aspect="equal",
        )
        self._draw_trajectories(export_axis)
        self._draw_start_marker(export_axis)
        export_axis.set_anchor("C")
        export_legend = self._draw_legend(export_axis, figure=export_figure)

        export_axis.set_title(self.title, pad=self.TITLE_PAD_POINTS)
        export_axis.set_xlabel("X (m)")
        export_axis.set_ylabel("Y (m)")
        export_axis.grid(
            True,
            color=self.GRID_COLOR,
            linestyle="--",
            linewidth=0.45,
            alpha=0.45,
        )
        self._style_plot_chrome(export_axis, export_legend)
        export_axis.set_xlim(self.axis.get_xlim())
        export_axis.set_ylim(self.axis.get_ylim())
        return export_figure

    def _parse_path_list(self, value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            items = value
        else:
            text = str(value).strip()
            if not text:
                return []
            items = text.replace(";", ",").split(",")
        paths = []
        for item in items:
            text = str(item).strip()
            if text:
                paths.append(text)
        return paths

    def _is_ours_loaded(self, traj):
        return bool(traj.get("is_ours"))

    def _visible_trajectories(self):
        return [
            traj for traj in self.trajectories if traj.get("visible", True)
        ]

    def _set_trajectory_visible(self, traj, visible):
        traj["visible"] = bool(visible)
        line = traj.get("mpl_line")
        if line is not None:
            line.set_visible(traj["visible"])
        self._request_plot_chrome_refresh()

    @staticmethod
    def _normalize_color_text(text):
        from matplotlib.backends.qt_compat import QtGui

        value = str(text).strip()
        if not value:
            return None
        if not value.startswith("#"):
            value = "#" + value
        color = QtGui.QColor(value)
        if not color.isValid():
            return None
        return color.name()

    def _style_color_swatch(self, button, color_text, selected=False):
        border = "#4338ca" if selected else "#94a3b8"
        button.setStyleSheet(
            "QToolButton#pcdColorSwatch {{"
            "background: {0};"
            "border: {1} solid {2};"
            "border-radius: 5px;"
            "padding: 0px;"
            "}}"
            "QToolButton#pcdColorSwatch:hover {{"
            "border: 2px solid #4338ca;"
            "}}".format(color_text, "2px" if selected else "1px", border)
        )

    def _set_trajectory_color(self, traj, color_text, editor=None, swatch=None):
        parsed = self._normalize_color_text(color_text)
        if parsed is None:
            if editor is not None:
                editor.setText(traj["color"])
            return
        if parsed == traj["color"]:
            if editor is not None:
                editor.setText(parsed)
            return
        traj["color"] = parsed
        if editor is not None:
            editor.setText(parsed)
        if swatch is not None:
            self._style_color_swatch(swatch, parsed)
        line = traj.get("mpl_line")
        if line is not None:
            line.set_color(parsed)
        self._request_plot_chrome_refresh()

    def _select_color_target(self, traj, editor, swatch):
        self._color_pick_target = (traj, editor, swatch)
        for _traj, _editor, button in getattr(self, "_color_editors", []):
            self._style_color_swatch(
                button,
                _traj["color"],
                selected=button is swatch,
            )

    def _build_inline_color_palette(self, parent):
        from matplotlib.backends.qt_compat import QtWidgets

        palette = QtWidgets.QWidget(parent)
        palette.setObjectName("pcdColorPalette")
        grid = QtWidgets.QGridLayout(palette)
        grid.setContentsMargins(0, 6, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        colors = (
            "#DC2626",
            "#EA580C",
            "#CA8A04",
            "#16A34A",
            "#0891B2",
            "#1D4ED8",
            "#6D28D9",
            "#9A3412",
            "#EF4444",
            "#F97316",
            "#EAB308",
            "#22C55E",
            "#06B6D4",
            "#3B82F6",
            "#8B5CF6",
            "#0F172A",
            "#FCA5A5",
            "#FDBA74",
            "#FDE047",
            "#86EFAC",
            "#67E8F9",
            "#93C5FD",
            "#C4B5FD",
            "#64748B",
        )
        columns = 8
        for index, color_text in enumerate(colors):
            cell = QtWidgets.QToolButton(palette)
            cell.setObjectName("pcdColorSwatch")
            cell.setFixedSize(20, 20)
            cell.setToolTip(color_text)
            self._style_color_swatch(cell, color_text)
            cell.clicked.connect(
                lambda _checked=False, value=color_text:
                self._apply_palette_color(value)
            )
            grid.addWidget(cell, index // columns, index % columns)
        return palette

    def _apply_palette_color(self, color_text):
        target = getattr(self, "_color_pick_target", None)
        if target is None:
            return
        traj, editor, swatch = target
        self._set_trajectory_color(traj, color_text, editor, swatch)
        self._style_color_swatch(swatch, color_text, selected=True)

    def _order_trajectories(self, trajectories):
        mains = [traj for traj in trajectories if traj["is_main"]]
        ours = [
            traj
            for traj in trajectories
            if not traj["is_main"] and self._is_ours_loaded(traj)
        ]
        others = [
            traj
            for traj in trajectories
            if not traj["is_main"] and not self._is_ours_loaded(traj)
        ]
        return mains + ours + others

    def _draw_order_trajectories(self):
        mains = [traj for traj in self.trajectories if traj["is_main"]]
        ours = [
            traj
            for traj in self.trajectories
            if not traj["is_main"] and self._is_ours_loaded(traj)
        ]
        others = [
            traj
            for traj in self.trajectories
            if not traj["is_main"] and not self._is_ours_loaded(traj)
        ]
        return others + ours + mains

    def _load_trajectories(self):
        trajectories = []
        benchmark_path = str(
            rospy.get_param("~benchmark_trajectory_path", "")
        ).strip()
        if benchmark_path:
            trajectory = self._read_trajectory(
                benchmark_path,
                label="benchmark",
                color=self.MAIN_TRAJECTORY_COLOR,
                linestyle="-",
                linewidth=self.MAIN_TRAJECTORY_LINEWIDTH,
                is_main=True,
                is_ours=False,
            )
            if trajectory is not None:
                trajectories.append(trajectory)

        ours_path = str(rospy.get_param("~ours_trajectory_path", "")).strip()
        if ours_path:
            trajectory = self._read_trajectory(
                ours_path,
                label="ours",
                color=self.OURS_TRAJECTORY_COLOR,
                linestyle="--",
                linewidth=self.OTHER_TRAJECTORY_LINEWIDTH,
                is_main=False,
                is_ours=True,
            )
            if trajectory is not None:
                trajectories.append(trajectory)

        other_paths = self._parse_path_list(
            rospy.get_param("~other_trajectory_paths", "")
        )
        other_names = self._parse_path_list(
            rospy.get_param("~other_trajectory_names", "")
        )
        for index, path_param in enumerate(other_paths):
            label = other_names[index] if index < len(other_names) else ""
            trajectory = self._read_trajectory(
                path_param,
                label=label,
                color=self.OTHER_TRAJECTORY_COLORS[
                    index % len(self.OTHER_TRAJECTORY_COLORS)
                ],
                linestyle="--",
                linewidth=self.OTHER_TRAJECTORY_LINEWIDTH,
                is_main=False,
                is_ours=False,
            )
            if trajectory is not None:
                trajectories.append(trajectory)
        trajectories = self._align_other_trajectories(trajectories)
        trajectories = self._order_trajectories(trajectories)
        self.start_xy = self._trajectory_start_xy(trajectories)
        return trajectories

    def _read_trajectory(
        self,
        path_param,
        label,
        color,
        linestyle,
        linewidth,
        is_main,
        is_ours=False,
    ):
        path = Path(path_param).expanduser()
        try:
            timestamps, positions, quaternions = read_tum_trajectory_poses(path)
        except PcdError as error:
            rospy.logwarn("Skipping TUM trajectory: %s", error)
            return None
        if not label:
            if is_main:
                label = "benchmark"
            elif is_ours:
                label = "ours"
            else:
                label = path.stem
        rospy.loginfo(
            "Loaded %s TUM trajectory: %s (%d poses, label=%s)",
            "main" if is_main else "ours" if is_ours else "other",
            path,
            positions.shape[0],
            label,
        )
        return {
            "path": path,
            "timestamps": timestamps,
            "positions": positions,
            "quaternions": quaternions,
            "label": label,
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
            "is_main": is_main,
            "is_ours": is_ours,
            "visible": True,
        }

    @staticmethod
    def _start_pose_index(timestamps):
        for index, timestamp in enumerate(timestamps):
            if timestamp > 0.0:
                return index
        return 0

    @staticmethod
    def _valid_pose_slice(timestamps, positions):
        start_index = TrajMonitor._start_pose_index(timestamps)
        return timestamps[start_index:], positions[start_index:]

    @staticmethod
    def _associate_xyz(t_ref, xyz_ref, t_est, xyz_est, max_diff):
        if t_ref.size == 0 or t_est.size == 0:
            empty = xyz_ref[:0]
            return empty, empty
        idx = np.searchsorted(t_ref, t_est)
        idx = np.clip(idx, 0, len(t_ref) - 1)
        idx_lo = np.clip(idx - 1, 0, len(t_ref) - 1)
        use_lo = np.abs(t_ref[idx_lo] - t_est) <= np.abs(t_ref[idx] - t_est)
        best = np.where(use_lo, idx_lo, idx)
        ok = np.abs(t_ref[best] - t_est) <= max_diff
        return xyz_ref[best[ok]], xyz_est[ok]

    @staticmethod
    def _motion_window(xyz, start_m, window_m, min_pts):
        if len(xyz) < 3:
            return 0, len(xyz)
        disp = np.linalg.norm(xyz - xyz[0], axis=1)
        moved = np.where(disp >= start_m)[0]
        i0 = int(moved[0]) if moved.size else 0
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        slen = np.concatenate([[0.0], np.cumsum(step)])
        i1 = int(np.searchsorted(slen, slen[i0] + window_m))
        i1 = min(max(i1, i0 + min_pts), len(xyz))
        if i1 - i0 < 3:
            return 0, min(len(xyz), max(min_pts, 3))
        return i0, i1

    @staticmethod
    def _umeyama_se3(src, dst):
        """Least-squares SE(3): dst ~= R @ src + t. src/dst are Nx3."""
        if src.shape[0] < 3 or src.shape != dst.shape:
            raise ValueError("need at least 3 corresponding 3D points")
        n = src.shape[0]
        mean_src = src.mean(axis=0)
        mean_dst = dst.mean(axis=0)
        cov = ((dst - mean_dst).T @ (src - mean_src)) / float(n)
        u, singular, vt = np.linalg.svd(cov)
        if np.count_nonzero(singular > np.finfo(singular.dtype).eps) < 2:
            raise ValueError("degenerate Umeyama covariance")
        correction = np.eye(3)
        if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
            correction[2, 2] = -1.0
        rotation = u @ correction @ vt
        translation = mean_dst - rotation @ mean_src
        return rotation, translation

    def _align_other_trajectories(self, trajectories):
        mains = [traj for traj in trajectories if traj["is_main"]]
        if not mains:
            return trajectories

        main = mains[0]
        t_ref, xyz_ref = self._valid_pose_slice(
            main["timestamps"], main["positions"]
        )
        rospy.loginfo(
            "Aligning other trajectories to %s with 3D Umeyama "
            "on the first %.1f m after moving %.1f m",
            main["label"],
            self.ALIGN_WINDOW_M,
            self.ALIGN_MOTION_START_M,
        )

        for traj in trajectories:
            if traj["is_main"]:
                continue
            t_est, xyz_est = self._valid_pose_slice(
                traj["timestamps"], traj["positions"]
            )
            ref_pts, est_pts = self._associate_xyz(
                t_ref,
                xyz_ref,
                t_est,
                xyz_est,
                self.ALIGN_MAX_TIME_DIFF,
            )
            if len(ref_pts) < self.ALIGN_MIN_PAIRS:
                rospy.logwarn(
                    "Skip Umeyama for %s: only %d time-matched poses",
                    traj["label"],
                    len(ref_pts),
                )
                continue
            i0, i1 = self._motion_window(
                ref_pts,
                self.ALIGN_MOTION_START_M,
                self.ALIGN_WINDOW_M,
                self.ALIGN_MIN_PAIRS,
            )
            try:
                rotation, translation = self._umeyama_se3(
                    est_pts[i0:i1], ref_pts[i0:i1]
                )
            except ValueError as error:
                rospy.logwarn("Skip Umeyama for %s: %s", traj["label"], error)
                continue
            traj["positions"] = np.ascontiguousarray(
                (traj["positions"] @ rotation.T) + translation
            )
            yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
            rospy.loginfo(
                "Aligned %s to %s: Umeyama yaw=%.3f deg, "
                "t=[%.3f, %.3f, %.3f] m, window=%d pairs",
                traj["label"],
                main["label"],
                yaw,
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                i1 - i0,
            )
        return trajectories

    def _trajectory_start_xy(self, trajectories):
        mains = [traj for traj in trajectories if traj["is_main"]]
        chosen = mains[0] if mains else (trajectories[0] if trajectories else None)
        if chosen is None:
            return None
        index = self._start_pose_index(chosen["timestamps"])
        return (
            float(chosen["positions"][index, 0]),
            float(chosen["positions"][index, 1]),
        )

    def _draw_start_marker(self, axis, figure=None):
        if self.start_xy is None:
            return None
        scale = self._ui_scale(axis.figure if figure is None else figure)
        (marker,) = axis.plot(
            [self.start_xy[0]],
            [self.start_xy[1]],
            linestyle="None",
            marker="^",
            markersize=self.START_MARKER_SIZE * scale,
            color=self.START_MARKER_COLOR,
            markeredgecolor="none",
            markeredgewidth=0.0,
            zorder=12,
        )
        return marker

    def _draw_trajectories(self, axis):
        lines = []
        live_axis = getattr(self, "axis", None)
        for index, traj in enumerate(self._draw_order_trajectories()):
            if axis is not live_axis and not traj.get("visible", True):
                continue
            (line,) = axis.plot(
                traj["positions"][:, 0],
                traj["positions"][:, 1],
                color=traj["color"],
                linestyle=traj["linestyle"],
                linewidth=traj["linewidth"],
                zorder=5 + index,
                label=traj["label"],
            )
            if axis is live_axis:
                traj["mpl_line"] = line
                line.set_visible(traj.get("visible", True))
            lines.append(line)
        return lines

    def _draw_legend(self, axis, figure=None):
        visible = self._visible_trajectories()
        if not visible and self.start_xy is None:
            return None
        scale = self._ui_scale(axis.figure if figure is None else figure)
        handles = [
            Line2D(
                [0],
                [0],
                color=traj["color"],
                linestyle=traj["linestyle"],
                linewidth=max(traj["linewidth"] * scale, 1.5),
                label=traj["label"],
            )
            for traj in visible
        ]
        if self.start_xy is not None:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker="^",
                    markersize=self.START_MARKER_SIZE * scale,
                    color=self.START_MARKER_COLOR,
                    markeredgecolor="none",
                    markeredgewidth=0.0,
                    label=self.START_MARKER_LABEL,
                )
            )
        return axis.legend(
            handles=handles,
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            fontsize=self.LEGEND_ITEM_FONTSIZE * scale,
            handlelength=self.LEGEND_HANDLELENGTH,
            handleheight=self.LEGEND_HANDLEHEIGHT,
            borderpad=self.LEGEND_BORDERPAD,
            labelspacing=self.LEGEND_LABELSPACING,
            handletextpad=self.LEGEND_HANDLETEXTPAD,
            borderaxespad=0.4,
        )

    def _rebuild_clicked(self, _event):
        self._finish_interaction(force=True)
        x_limits = self.axis.get_xlim()
        y_limits = self.axis.get_ylim()
        extent = extent_for_bounds(
            x_limits,
            y_limits,
            aspect_ratio=self.BEV_ASPECT_RATIO,
            padding_ratio=0.0,
        )

        self._rebuild_action.setText("Rebuilding…")
        self._rebuild_action.setEnabled(False)
        self.figure.canvas.flush_events()
        try:
            bev = build_bev(
                self.sampled_xyz,
                extent,
                grid_width=self.selected_bev_width,
                grid_height=self.selected_bev_height,
                dataset_stats=self.dataset_stats,
            )
            self._build_bev_image(bev)
        except PcdError as error:
            rospy.logerr("Could not rebuild PCD BEV: %s", error)
        else:
            self._apply_bev(bev)
            self._log_bev_info("Rebuilt BEV", bev)
        finally:
            self._rebuild_action.setText("Rebuild BEV")
            self._rebuild_action.setEnabled(True)
            self.figure.canvas.draw_idle()

    def _restore_full_bev(self):
        self._cancel_interaction()
        self._apply_bev(self.full_bev)
        self._log_bev_info("Restored full BEV", self.full_bev)
        self.figure.canvas.draw_idle()

    def _apply_bev(self, bev):
        self._cancel_interaction()
        self._drag_state = None
        self._build_bev_image(bev)
        self.bev = bev
        self._bev_qpixmap(bev)
        self.image.set_data(bev["rgba_image"])
        self.image.set_extent(bev["extent"])
        self._set_plot_title()
        self._set_axis_extent(bev["extent"])
        self._interaction_overlay.raise_()
        self._raise_toolbar_dropdowns()
        self._interaction_overlay.update()

    def _log_bev_info(self, label, bev):
        min_x, max_x, min_y, max_y = bev["extent"]
        min_z, max_z = bev["xyz_bounds"][2]
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
            "source Z [%.6g, %.6g] m",
            label,
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
        )

    @staticmethod
    def _hex_to_rgba(hex_color):
        value = hex_color.lstrip("#")
        return np.array(
            (
                int(value[0:2], 16),
                int(value[2:4], 16),
                int(value[4:6], 16),
                255,
            ),
            dtype=np.uint8,
        )

    def _build_bev_image(self, bev):
        if "rgba_image" in bev:
            return

        occupied = np.isfinite(bev["grid"])
        rgba_bottom_up = np.empty(occupied.shape + (4,), dtype=np.uint8)
        rgba_bottom_up[...] = self._hex_to_rgba(self.PLOT_EMPTY_COLOR)
        rgba_bottom_up[occupied] = self._hex_to_rgba(self.PLOT_POINT_COLOR)
        # Store rows in display order (top to bottom). Matplotlib uses
        # origin="upper" and Qt can consume the same contiguous RGBA buffer.
        bev["rgba_image"] = np.ascontiguousarray(rgba_bottom_up[::-1])

    def _set_axis_extent(self, extent):
        min_x, max_x, min_y, max_y = extent
        self.axis.set_xlim(min_x, max_x)
        self.axis.set_ylim(min_y, max_y)
        self._sync_zoom_slider_to_current_view()

    def _on_close(self, _event):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Trajectory monitor window closed")

    def _close_figure(self):
        if hasattr(self, "figure"):
            if hasattr(self, "_interaction_restore_timer"):
                self._interaction_restore_timer.stop()
            plt.close(self.figure)

    def show(self):
        plt.show()


def main():
    rospy.init_node("traj_monitor")
    try:
        monitor = TrajMonitor()
    except (PcdError, ValueError) as error:
        rospy.logfatal("Could not start traj_monitor: %s", error)
        return 1
    monitor.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
