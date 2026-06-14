import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from matplotlib.colors import ListedColormap
from PyQt6.QtCore import QCoreApplication, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QWidget,
)
from qtrangeslider import QRangeSlider

from core.project_naming import is_segmentation_channel
from ui.toolbar.Action import Action
from utils import resource_path


class TabIndices(IntEnum):
    EXTRACT = 0
    VIEW = 1
    ANALYZE = 2


class ToolBarUI(QToolBar):
    # Public signals
    tabChanged = pyqtSignal(int)
    channelChanged = pyqtSignal(int)
    cmapChanged = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.initialized = False
        self._pending_channel_index: int | None = None
        self._init_tab_buttons()
        self._init_actions(parent)
        # self._init_channel_selector(parent)
        self._init_status_line()
        self._init_cmap_selector(parent)
        self._init_contrast_slider()
        self._init_channel_selector(parent)

        self._populate_toolbar()
        self._retranslateUI()

    def _init_tab_buttons(self):
        self.tab_buttons = []
        tab_names = ["Extract", "View", "Analyze"]
        for name in tab_names:
            button = QToolButton()
            button.setText(name)
            button.setStyleSheet(self._tab_button_style())
            button.setCheckable(True)
            self.tab_buttons.append(button)

        # Group buttons for exclusive behavior
        self.tabButtonGroup = QButtonGroup(self)
        self.tabButtonGroup.setExclusive(True)
        for idx, button in enumerate(self.tab_buttons):
            self.tabButtonGroup.addButton(button, idx)

        self.tabButtonGroup.idClicked.connect(self.onTabButtonClicked)
        self.tab_buttons[0].setChecked(True)

    def _tab_button_style(self):
        from ui.theme import ThemeManager
        c = ThemeManager.instance().get_current()
        return f"""
            QToolButton {{
                padding: 8px 16px;
                border: none;
                background: transparent;
                font-size: 12px;
            }}
            QToolButton:hover {{
                background: {c["bg_hover"]};
            }}
            QToolButton:checked {{
                border-bottom: 2px solid {c["tab_checked"]};
                font-weight: Bold;
            }}
        """

    @pyqtSlot(int)
    def onTabButtonClicked(self, index):
        self.tabChanged.emit(index)
        if index == TabIndices.EXTRACT:
            self.actionReset.setVisible(True)
            self.channel_selector_action.setVisible(True)
            self.cmap_action.setVisible(True)
            self.auto_contrast_button_action.setVisible(True)
            self.contrast_slider_action.setVisible(True)
        else:
            # hide all actions
            self.actionReset.setVisible(False)
            self.channel_selector_action.setVisible(False)
            self.cmap_action.setVisible(False)
            self.auto_contrast_button_action.setVisible(False)
            self.contrast_slider_action.setVisible(False)

    def _init_actions(self, parent):
        self.actionRotate = Action(
            parent, "actionRotate", resource_path("assets/icons/rotate-right.png")
        )
        self.actionReset = Action(
            parent, "actionReset", resource_path("assets/icons/reset.png")
        )
        self.actionOpenBrightnessContrast = Action(
            parent, "actionBC", resource_path("assets/icons/brightness.png")
        )

    def _init_channel_selector(self, parent):
        self.channelSelector = QComboBox(parent)
        self.channelSelector.setMinimumWidth(100)
        self.channelSelector.currentIndexChanged.connect(
            self.on_channelSelector_currentIndexChanged
        )

    def updateChannelSelector(self, channels: dict, clear=False):
        self.initialized = False
        was_blocked = self.channelSelector.blockSignals(True)
        try:
            # Always rebuild from source-of-truth channels to avoid duplicate entries.
            self.clearChannelSelector()
            channel_keys = sorted(
                channels.keys(), key=lambda x: int(x.replace("Channel ", ""))
            )
            for channel_key in channel_keys:
                wrapper = channels[channel_key]
                display_name = getattr(wrapper, "name", "") or channel_key
                if is_segmentation_channel(wrapper):
                    text = f"{channel_key} ({display_name})"
                elif display_name != channel_key:
                    text = f"{channel_key} ({display_name})"
                else:
                    text = channel_key
                self.channelSelector.addItem(text)

            if self._pending_channel_index is not None and self.channelSelector.count() > 0:
                max_index = self.channelSelector.count() - 1
                safe_index = max(0, min(self._pending_channel_index, max_index))
                self.channelSelector.setCurrentIndex(safe_index)
                self._pending_channel_index = None
        finally:
            self.channelSelector.blockSignals(was_blocked)
        return

    def setChannelSelector(self, channel: int):
        if self.channelSelector.count() == 0:
            self._pending_channel_index = channel
            return

        max_index = self.channelSelector.count() - 1
        safe_index = max(0, min(channel, max_index))
        was_blocked = self.channelSelector.blockSignals(True)
        try:
            self.channelSelector.setCurrentIndex(safe_index)
        finally:
            self.channelSelector.blockSignals(was_blocked)

    def clearChannelSelector(self):
        self.channelSelector.clear()
        return

    @pyqtSlot(int)
    def on_channelSelector_currentIndexChanged(self, index):
        if self.initialized is False:
            self.initialized = True
        if self.channelSelector.count() > 0:
            self.channelChanged.emit(index)

    def _init_status_line(self):
        self.statusLine = QLabel("")
        self.statusLine.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.statusLine.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.statusLine.setStyleSheet("margin: 10px;")

    def _init_cmap_selector(self, parent):
        self.cmapSelector = QComboBox(parent)
        self.cmapSelector.setMinimumSize(QSize(200, 20))
        self.cmapSelector.setMaximumWidth(200)
        self.cmapSelector.currentTextChanged.connect(self.on_cmapTextChanged)

        thumbnails = self._generate_cmap_thumbnails()
        for index, thumbnail in enumerate(thumbnails):
            icon = self._numpy_to_QIcon(thumbnail)
            self.cmapSelector.setIconSize(QSize(80, 20))
            self.cmapSelector.addItem(icon, self.cmap_names[index])

    @pyqtSlot(str)
    def on_cmapTextChanged(self, cmap_str: str):
        self.cmapChanged.emit(cmap_str)

    def update_cmap_selector(self, cmap_value):
        self.cmapSelector.setCurrentText(cmap_value)

    def _generate_cmap_thumbnails(self):
        self.cmap_names = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "label_image",
        ]
        thumbnails = []
        for cmap_name in self.cmap_names:
            if cmap_name == "label_image":
                # Make a thumbnail with discrete colors (like skimage.label2rgb default)
                from skimage.color import color_dict

                # Get a list of default categorical colors
                colors = [
                    "red",  # (1, 0, 0)
                    "blue",  # (0, 0, 1)
                    "green",  # (0, 0.502, 0)
                    "gold",  # (1, 0.843, 0)
                    "purple",  # (0.502, 0, 0.502)
                    "orange",  # (1, 0.647, 0)
                    "cyan",  # (0, 1, 1)
                    "magenta",  # (1, 0, 1)
                    "brown",  # (0.647, 0.165, 0.165)
                    "lime",  # (0, 1, 0)
                ]

                colors = [
                    color_dict[i] for i in colors
                ]  # take first N colors (e.g. 10)
                n = len(colors)

                # Make an array of shape (1, n) with categorical colors
                gradient = np.arange(n).reshape(1, -1)

                fig, ax = plt.subplots(figsize=(4, 1))
                cmap = ListedColormap(colors)
                ax.imshow(gradient, aspect="auto", cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                thumbnails.append(np.array(fig.canvas.renderer.buffer_rgba()))
                plt.close()
            else:
                # Continuous cmap preview
                gradient = np.linspace(0, 1, 256)
                gradient = np.vstack((gradient, gradient))
                fig, ax = plt.subplots(figsize=(4, 1))
                ax.imshow(gradient, aspect="auto", cmap=cmap_name)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout(pad=0)
                fig.canvas.draw()
                thumbnails.append(np.array(fig.canvas.renderer.buffer_rgba()))
                plt.close()
        return thumbnails

    def _numpy_to_QIcon(self, array: np.ndarray):
        height, width, channels = array.shape
        image = QImage(array.tobytes(), width, height, QImage.Format.Format_RGBA8888)
        return QIcon(QPixmap.fromImage(image))

    def _init_contrast_slider(self):
        self.auto_contrast_button = QPushButton("Auto Contrast", self)
        self.contrast_slider = QRangeSlider(parent=self)
        self.contrast_slider.setOrientation(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 255)
        self.contrast_slider.setMaximumWidth(200)
        self.contrast_slider.setMinimumWidth(100)

    def update_contrast_slider(self, values):
        self.contrast_slider.blockSignals(True)
        self.contrast_slider.setValue(values)
        self.contrast_slider.blockSignals(False)

    def _create_tab_button_bar(self, fixed_width=400):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(0)

        for btn in self.tab_buttons:
            btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            layout.addWidget(btn)

        container.setFixedWidth(fixed_width)

        return container

    def _populate_toolbar(self):
        tab_bar = self._create_tab_button_bar(fixed_width=488)
        self.addWidget(tab_bar)
        self.addSeparator()
        self.addAction(self.actionReset)
        self.cmap_action = self.addWidget(self.cmapSelector)
        self.channel_selector_action = self.addWidget(self.channelSelector)
        self.auto_contrast_button_action = self.addWidget(self.auto_contrast_button)
        self.contrast_slider_action = self.addWidget(self.contrast_slider)
        self.addWidget(self.statusLine)

    def _retranslateUI(self):
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionReset.setText(_translate("MainWindow", "Reset"))
        self.actionReset.setToolTip(_translate("MainWindow", "Reset Image"))
