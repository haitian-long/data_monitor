#!/usr/bin/env python3

import copy
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
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
    INTERACTION_RESTORE_DELAY_MS = 180
    MIN_VIEW_VOXELS = 8.0
    MAX_FULL_VIEW_SCALE = 4.0
    ZOOM_SLIDER_STEPS = 10000
    TITLE_PAD_POINTS = 15.0
    EXPORT_DPI = 200
    UI_BACKGROUND = "#f8fafc"
    UI_SURFACE = "#ffffff"
    UI_BORDER = "#cbd5e1"
    UI_TEXT = "#0f172a"
    UI_MUTED_TEXT = "#64748b"

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
        self._build_bev_image(self.bev)
        self.full_bev = self.bev
        self._log_bev_info("Initial BEV", self.bev)

        self._drag_state = None
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
        plot_position = self.axis.get_position()
        self.axis.set_position(
            (
                (1.0 - plot_position.width) / 2.0,
                plot_position.y0,
                plot_position.width,
                plot_position.height,
            )
        )

        colormap = copy.copy(plt.get_cmap(self.colormap))
        colormap.set_bad(color="#f3f4f6", alpha=1.0)
        self.image = self.axis.imshow(
            self.bev["rgba_image"],
            origin="upper",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
            aspect="equal",
        )
        self.trajectory_line = self._draw_trajectory(self.axis)
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
        self.color_mappable = ScalarMappable(
            norm=self._normalization_for_bev(self.bev),
            cmap=colormap,
        )
        self.color_mappable.set_array([])
        self.colorbar = self.figure.colorbar(
            self.color_mappable,
            cax=self.colorbar_axis,
        )
        self.colorbar.set_label("Maximum Z in pixel (m), fixed P1-P99")

        self._set_plot_title()
        self.axis.set_xlabel("X (m)")
        self.axis.set_ylabel("Y (m)")
        self.axis.grid(True, color="#ffffff", linestyle="--", linewidth=0.45, alpha=0.35)
        self._style_plot_chrome(self.axis, self.colorbar)
        self._set_axis_extent(self.bev["extent"])

        self._create_native_toolbar()
        self._layout_figure()
        self._create_interaction_overlay()
        self._bev_qpixmap(self.bev)

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

        self._begin_interaction()
        self.axis.set_xlim(
            event.xdata - relative_x * new_width,
            event.xdata + (1.0 - relative_x) * new_width,
        )
        self.axis.set_ylim(
            event.ydata - relative_y * new_height,
            event.ydata + (1.0 - relative_y) * new_height,
        )
        self._sync_zoom_slider_to_current_view()
        self._update_interaction_overlay()
        self._schedule_interaction_finish()

    def _on_resize(self, _event):
        self._cancel_interaction()
        self._layout_figure()
        self.figure.canvas.draw_idle()
        self._update_interaction_overlay_geometry()
        self._interaction_overlay.raise_()
        self._interaction_overlay.update()

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

        x_limits, y_limits = self._drag_limits_for_event(event)
        self._set_axis_limits(x_limits, y_limits)
        self._update_interaction_overlay()

    def _on_button_release(self, event):
        if self._drag_state is None:
            return

        if event.x is not None and event.y is not None:
            x_limits, y_limits = self._drag_limits_for_event(event)
            self._set_axis_limits(x_limits, y_limits)

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

    def _set_plot_title(self):
        self.axis.set_title(
            self.title,
            pad=self.TITLE_PAD_POINTS,
        )

    def _style_plot_chrome(self, axis, colorbar):
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

        colorbar.outline.set_edgecolor(self.UI_BORDER)
        colorbar.outline.set_linewidth(0.8)
        colorbar.ax.tick_params(
            colors=self.UI_MUTED_TEXT,
            labelsize=8.5,
            length=3.0,
            width=0.8,
        )
        colorbar.ax.yaxis.label.set_color("#475569")
        colorbar.ax.yaxis.label.set_fontsize(9.5)

    def _create_native_toolbar(self):
        try:
            from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
        except ImportError as error:
            raise PcdError(
                "The native pcd_monitor toolbar requires a Qt Matplotlib backend"
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
                "The native pcd_monitor toolbar requires FigureManagerQT"
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
        popup_mode = getattr(
            QtWidgets.QToolButton,
            "ToolButtonPopupMode",
            QtWidgets.QToolButton,
        )
        instant_popup = getattr(popup_mode, "InstantPopup")

        window.removeToolBar(toolbar)
        window.addToolBar(top_tool_bar_area, toolbar)
        toolbar.clear()
        toolbar.setObjectName("pcdMonitorToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(text_only_style)

        self._reset_action = toolbar.addAction("Reset view")
        self._reset_action.setToolTip("Restore the initial full BEV view")
        self._reset_action.triggered.connect(
            lambda _checked=False: self._reset_clicked(None)
        )

        self._zoom_tool_button = QtWidgets.QToolButton(toolbar)
        self._zoom_tool_button.setText("Zoom")
        self._zoom_tool_button.setPopupMode(instant_popup)
        self._zoom_tool_button.setToolTip(
            "Adjust the BEV zoom factor; the mouse wheel remains available"
        )
        zoom_menu = QtWidgets.QMenu(self._zoom_tool_button)
        zoom_menu.setObjectName("pcdMonitorMenu")
        zoom_content = QtWidgets.QWidget(zoom_menu)
        zoom_content.setObjectName("pcdZoomContent")
        zoom_layout = QtWidgets.QHBoxLayout(zoom_content)
        zoom_layout.setContentsMargins(16, 10, 16, 10)
        zoom_layout.setSpacing(10)

        zoom_out_label = QtWidgets.QLabel("−", zoom_content)
        zoom_out_label.setObjectName("pcdZoomBoundLabel")
        zoom_in_label = QtWidgets.QLabel("+", zoom_content)
        zoom_in_label.setObjectName("pcdZoomBoundLabel")
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

        zoom_layout.addWidget(zoom_out_label)
        zoom_layout.addWidget(self._qt_zoom_slider, 1)
        zoom_layout.addWidget(zoom_in_label)
        zoom_layout.addWidget(self._qt_zoom_value_label)
        zoom_content.setMinimumWidth(560)

        zoom_widget_action = QtWidgets.QWidgetAction(zoom_menu)
        zoom_widget_action.setDefaultWidget(zoom_content)
        zoom_menu.addAction(zoom_widget_action)
        self._zoom_tool_button.setMenu(zoom_menu)
        toolbar.addWidget(self._zoom_tool_button)
        self._qt_zoom_menu = zoom_menu
        self._qt_zoom_widget_action = zoom_widget_action

        self._resolution_tool_button = QtWidgets.QToolButton(toolbar)
        self._resolution_tool_button.setText("Resolution")
        self._resolution_tool_button.setPopupMode(instant_popup)
        self._resolution_tool_button.setToolTip(
            "Select the resolution used by the next BEV rebuild"
        )
        resolution_menu = QtWidgets.QMenu(self._resolution_tool_button)
        resolution_menu.setObjectName("pcdMonitorMenu")
        action_group_class = (
            getattr(QtGui, "QActionGroup", None)
            or getattr(QtWidgets, "QActionGroup")
        )
        self._resolution_action_group = action_group_class(resolution_menu)
        self._resolution_action_group.setExclusive(True)
        self._resolution_actions = {}
        for width, height in self.BEV_RESOLUTIONS:
            action = resolution_menu.addAction(
                "{} × {}".format(width, height)
            )
            action.setCheckable(True)
            action.setChecked(
                width == self.selected_bev_width
                and height == self.selected_bev_height
            )
            action.triggered.connect(
                lambda _checked=False,
                selected_width=width,
                selected_height=height:
                self._select_bev_resolution(
                    selected_width,
                    selected_height,
                )
            )
            self._resolution_action_group.addAction(action)
            self._resolution_actions[(width, height)] = action
        self._resolution_tool_button.setMenu(resolution_menu)
        toolbar.addWidget(self._resolution_tool_button)
        self._qt_resolution_menu = resolution_menu

        toolbar.addSeparator()
        self._save_action = toolbar.addAction("Save Image")
        self._save_action.setToolTip(
            "Save the current BEV, title and colorbar beside the PCD file"
        )
        self._save_action.triggered.connect(
            lambda _checked=False: self._save_clicked(None)
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
            QToolBar#pcdMonitorToolbar {
                background: #ffffff;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                spacing: 4px;
                padding: 6px 10px;
            }
            QToolBar#pcdMonitorToolbar QToolButton {
                color: #334155;
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 7px 11px;
                font-weight: 600;
            }
            QToolBar#pcdMonitorToolbar QToolButton:hover,
            QToolBar#pcdMonitorToolbar QToolButton:pressed {
                color: #4338ca;
                background: #eef2ff;
            }
            QToolBar#pcdMonitorToolbar QToolButton#pcdPrimaryToolButton {
                color: #ffffff;
                background: #4f46e5;
                padding-left: 15px;
                padding-right: 15px;
            }
            QToolBar#pcdMonitorToolbar
            QToolButton#pcdPrimaryToolButton:hover,
            QToolBar#pcdMonitorToolbar
            QToolButton#pcdPrimaryToolButton:pressed {
                color: #ffffff;
                background: #4338ca;
            }
            QToolBar#pcdMonitorToolbar QToolBarSeparator {
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
            QWidget#pcdZoomContent {
                background: #ffffff;
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
        zoom_menu.aboutToShow.connect(
            self._sync_zoom_slider_to_current_view
        )

        rospy.loginfo(
            "Installed native Qt toolbar: Reset | Zoom | Resolution | "
            "Save | Rebuild"
        )

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
        self._trajectory_qpath = self._build_trajectory_qpath()

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

        # Keep the independently rendered colorbar visible beside the
        # interaction layer.
        colorbar_left = float(self.colorbar_axis.bbox.x0) * scale_x
        if colorbar_left > axis_right:
            overlay_right = min(overlay_right, colorbar_left - 2.0)

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
            self._qt_gui.QColor("#ffffff"),
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

        x_ticks, x_labels, x_offset = self._interaction_axis_ticks(
            self.axis.xaxis
        )
        y_ticks, y_labels, y_offset = self._interaction_axis_ticks(
            self.axis.yaxis
        )
        self._draw_interaction_grid(
            painter,
            data_rect,
            x_ticks,
            y_ticks,
        )
        self._draw_interaction_trajectory(painter, data_rect)
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
        view_min_x, view_max_x = self.axis.get_xlim()
        view_min_y, view_max_y = self.axis.get_ylim()
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

    def _interaction_axis_ticks(self, matplotlib_axis):
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
        return locations, labels, formatter.get_offset()

    def _draw_interaction_grid(
        self,
        painter,
        data_rect,
        x_ticks,
        y_ticks,
    ):
        view_min_x, view_max_x = self.axis.get_xlim()
        view_min_y, view_max_y = self.axis.get_ylim()
        grid_color = self._qt_gui.QColor("#ffffff")
        grid_color.setAlphaF(0.35)
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

    def _build_trajectory_qpath(self):
        if self.trajectory_positions is None:
            return None

        path = self._qt_gui.QPainterPath()
        path_started = False
        for x_value, y_value in self.trajectory_positions[:, :2]:
            if not math.isfinite(x_value) or not math.isfinite(y_value):
                path_started = False
                continue
            if path_started:
                path.lineTo(float(x_value), float(y_value))
            else:
                path.moveTo(float(x_value), float(y_value))
                path_started = True
        return path

    def _draw_interaction_trajectory(self, painter, data_rect):
        if self._trajectory_qpath is None:
            return

        view_min_x, view_max_x = self.axis.get_xlim()
        view_min_y, view_max_y = self.axis.get_ylim()
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
        trajectory_pen = self._qt_gui.QPen(
            self._qt_gui.QColor("#FBBC05")
        )
        trajectory_pen.setWidthF(1.5)
        trajectory_pen.setCosmetic(True)
        painter.save()
        painter.setClipRect(data_rect)
        painter.setPen(trajectory_pen)
        painter.setTransform(transform, True)
        painter.drawPath(self._trajectory_qpath)
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
        view_min_x, view_max_x = self.axis.get_xlim()
        view_min_y, view_max_y = self.axis.get_ylim()

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
        if draw:
            # Synchronize Matplotlib behind the persistent Qt plot layer.
            # The screen never switches renderers when interaction ends.
            self.figure.canvas.draw()
        self._interaction_active = False
        self._interaction_overlay.raise_()
        self._interaction_overlay.update()
        return True

    def _cancel_interaction(self):
        if hasattr(self, "_interaction_restore_timer"):
            self._interaction_restore_timer.stop()
        self._interaction_active = False
        if hasattr(self, "_interaction_overlay"):
            self._interaction_overlay.raise_()
            self._interaction_overlay.update()

    def _select_bev_resolution(self, width, height):
        self.selected_bev_width = width
        self.selected_bev_height = height
        for resolution, action in self._resolution_actions.items():
            action.setChecked(resolution == (width, height))
        rospy.loginfo(
            "Selected BEV rebuild resolution: %d x %d; "
            "click Rebuild BEV to apply",
            width,
            height,
        )

    def _zoom_slider_released(self):
        self._finish_interaction(force=True)

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

        x_min, x_max = self.axis.get_xlim()
        y_min, y_max = self.axis.get_ylim()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        new_width = self._slider_value_to_view_width(slider_value)
        new_height = new_width / self.BEV_ASPECT_RATIO

        self._begin_interaction()
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
        self._update_interaction_overlay()
        self._schedule_interaction_finish()

    def _sync_zoom_slider_to_current_view(self):
        if not hasattr(self, "_qt_zoom_slider"):
            return

        x_min, x_max = self.axis.get_xlim()
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
        if event.key and event.key.lower() == "r":
            self._restore_full_bev()

    def _reset_clicked(self, _event):
        self._restore_full_bev()

    def _save_clicked(self, _event):
        self._finish_interaction(force=True)
        self._save_action.setText("Saving…")
        self._save_action.setEnabled(False)
        self.figure.canvas.flush_events()
        try:
            output_path = self._next_export_path()
            self._save_bev_image(output_path)
        except (OSError, ValueError) as error:
            rospy.logerr("Could not save PCD BEV image: %s", error)
        else:
            rospy.loginfo("Saved PCD BEV image: %s", output_path)
        finally:
            self._save_action.setText("Save Image")
            self._save_action.setEnabled(True)

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
        export_axis.imshow(
            self.bev["rgba_image"],
            origin="upper",
            extent=self.bev["extent"],
            interpolation="nearest",
            resample=False,
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
        export_mappable = ScalarMappable(
            norm=self._normalization_for_bev(self.bev),
            cmap=colormap,
        )
        export_mappable.set_array([])
        export_colorbar = export_figure.colorbar(
            export_mappable,
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
        self._style_plot_chrome(export_axis, export_colorbar)
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
        self.color_mappable.set_norm(self._normalization_for_bev(bev))
        self.colorbar.update_normal(self.color_mappable)
        self._set_plot_title()
        self._set_axis_extent(bev["extent"])
        self._interaction_overlay.raise_()
        self._interaction_overlay.update()

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

    def _build_bev_image(self, bev):
        if "rgba_image" in bev:
            return

        colormap = copy.copy(plt.get_cmap(self.colormap))
        colormap.set_bad(color="#f3f4f6", alpha=1.0)
        masked_grid = np.ma.masked_invalid(bev["grid"])
        rgba_bottom_up = colormap(
            self._normalization_for_bev(bev)(masked_grid),
            bytes=True,
        )
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
