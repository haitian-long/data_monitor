"""In-window toolbar dropdowns.

QMenu popups are separate windows. On WSLg their close is delayed and the
dismiss click can show them again. These panels are ordinary child widgets
of the main window, so hide() is immediate and there is no popup grab.
"""


def _qt_named_flag(namespace, *names):
    current = namespace
    for name in names:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _escape_key(QtCore):
    return _qt_named_flag(QtCore.Qt, "Key", "Key_Escape") or _qt_named_flag(
        QtCore.Qt,
        "Key_Escape",
    )


def dropdown_stylesheet():
    return """
        QFrame#pcdToolbarDropdown {
            color: #334155;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }
        QWidget#pcdZoomContent,
        QWidget#pcdFilterContent,
        QWidget#pcdColorContent,
        QWidget#pcdResolutionContent {
            background: #ffffff;
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
        QToolButton#pcdResolutionChoice {
            color: #334155;
            background: transparent;
            border: none;
            border-radius: 6px;
            padding: 8px 12px;
            font-weight: 600;
            text-align: left;
        }
        QToolButton#pcdResolutionChoice:checked,
        QToolButton#pcdResolutionChoice:hover {
            color: #4338ca;
            background: #eef2ff;
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


def create_dropdown_panel(window, QtCore, QtWidgets):
    escape = _escape_key(QtCore)

    class DropdownPanel(QtWidgets.QFrame):
        def keyPressEvent(panel_self, event):
            if escape is not None and event.key() == escape:
                hide_dropdown(panel_self)
                event.accept()
                return
            super(DropdownPanel, panel_self).keyPressEvent(event)

    panel = DropdownPanel(window)
    panel.setObjectName("pcdToolbarDropdown")
    panel.setStyleSheet(dropdown_stylesheet())
    layout = QtWidgets.QVBoxLayout(panel)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(0)
    panel.hide()
    return panel


def create_dropdown_content(panel, QtWidgets):
    return QtWidgets.QWidget(panel)


def hide_dropdown(panel):
    if panel is None or not panel.isVisible():
        return False
    panel.hide()
    button = getattr(panel, "_toolbar_button", None)
    if button is not None:
        button.setDown(False)
    on_hide = getattr(panel, "_on_hide", None)
    if on_hide is not None:
        on_hide()
    return True


def show_dropdown(button, panel, QtCore):
    parent = panel.parent()
    origin = button.mapTo(parent, QtCore.QPoint(0, button.height()))
    panel.adjustSize()
    left = origin.x()
    top = origin.y()
    if parent is not None:
        left = min(max(left, 8), max(8, parent.width() - panel.width() - 8))
        top = min(max(top, 8), max(8, parent.height() - panel.height() - 8))
    panel.move(left, top)
    panel.show()
    panel.raise_()
    panel.setFocus()
    button.setDown(True)


def attach_toolbar_dropdown(
    button,
    panel,
    on_show=None,
    on_hide=None,
    QtCore=None,
    sibling_panels=None,
):
    if QtCore is None:
        raise TypeError("attach_toolbar_dropdown requires QtCore")

    panel._toolbar_button = button
    panel._on_hide = on_hide

    def _toggle(_checked=False):
        if panel.isVisible():
            hide_dropdown(panel)
            return
        if sibling_panels is not None:
            for other in sibling_panels():
                if other is not None and other is not panel:
                    hide_dropdown(other)
        if on_show is not None:
            on_show()
        show_dropdown(button, panel, QtCore)

    button.clicked.connect(_toggle)
    button._toolbar_dropdown_toggle = _toggle


def raise_visible_dropdowns(panels):
    for panel in panels:
        if panel is not None and panel.isVisible():
            panel.raise_()
