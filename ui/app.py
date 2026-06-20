"""
Main application window module.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ui.alignment.cell_intensity_ui import CellIntensityUI
    from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
    from ui.alignment.register_ui import RegisterUI
    from ui.analysis.AnalysisTab import AnalysisTab
    from ui.canvas import ImageGraphicsViewUI, ReferenceGraphicsViewUI
    from ui.image_manager import ImageManager
    from ui.processing.crop_ui import CropUI
    from ui.processing.gaussian_blur import GaussianBlur
    from ui.processing.rotation_ui import RotateUI
    from ui.stardist.stardist_ui import StarDistUI
    from ui.view_tab import ImageOverlay

# pylint: disable=no-name-in-module
from PyQt6.QtCore import (QCoreApplication, QMetaObject, QPointF, QSize, Qt)
from PyQt6.QtGui import (QIcon, QImageReader, QKeySequence, QMouseEvent,
                         QShortcut)
from PyQt6.QtWidgets import (QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSizePolicy, QSplitter,
                             QSplitterHandle, QStackedWidget, QStatusBar,
                             QTabWidget, QVBoxLayout, QWidget)

from core import MetaData
from core.project_manager import ProjectManager
from core.project_naming import is_segmentation_name
from ui.toolbar.menubar_ui import MenuBarUI
from ui.toolbar.toolbar_ui import ToolBarUI
from utils import resource_path

logger = logging.getLogger(__name__)


def should_use_custom_windows_chrome() -> bool:
    return False


class _ResizeDragButton(QPushButton):
    """Pill button that forwards mouse events to the SidebarHandle, reusing its drag-to-resize logic."""

    def __init__(self, handle: "SidebarHandle"):
        super().__init__(handle)
        self._handle = handle
        self._icon_path = resource_path("assets/icons/left_right_indicator.svg")
        self.setIconSize(QSize(12, 12))
        self.setFixedSize(14, 40)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Drag to resize sidebar")
        self._apply_theme()

    def _apply_theme(self):
        from ui.theme import ThemeManager, make_themed_icon
        c = ThemeManager.instance().get_current()
        self.setIcon(make_themed_icon(str(self._icon_path), 12))
        self.setStyleSheet(
            f"""
            QPushButton {{
                background: {c["handle_bg"]};
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {c["handle_hover"]}; }}
        """
        )

    def _forward(self, event, method_name):
        pos = QPointF(self.mapTo(self._handle, event.position().toPoint()))
        new_ev = QMouseEvent(
            event.type(),
            pos,
            event.globalPosition(),
            event.button(),
            event.buttons(),
            event.modifiers(),
        )
        getattr(self._handle, method_name)(new_ev)

    def mousePressEvent(self, event):
        mw = self._handle.window()
        if hasattr(mw, "side_panel") and not mw.side_panel.isVisible():
            mw.toggle_side_panel()
        self._forward(event, "mousePressEvent")

    def mouseMoveEvent(self, event):
        self._forward(event, "mouseMoveEvent")

    def mouseReleaseEvent(self, event):
        self._forward(event, "mouseReleaseEvent")


class SidebarHandle(QSplitterHandle):
    """Custom splitter handle: visible 16px strip with a collapse/expand pill button."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setFixedWidth(16)
        self.setStyleSheet("background: transparent;")

        self._btn = QPushButton(self)
        self._btn.setFixedSize(14, 40)
        self._btn.setIconSize(QSize(12, 12))
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setToolTip("Collapse/expand sidebar")
        self._collapsed = False
        self._apply_btn_theme()

        self._resize_btn = _ResizeDragButton(handle=self)

        self._container = QWidget(self)
        self._container.setStyleSheet("background: transparent;")
        self._container.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
        )
        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._btn)
        layout.addWidget(self._resize_btn)
        self._container.adjustSize()

    def _apply_btn_theme(self):
        from ui.theme import ThemeManager, make_themed_icon
        c = ThemeManager.instance().get_current()
        icon_name = "chevron_right.svg" if self._collapsed else "chevron_left.svg"
        icon_path = resource_path(f"assets/icons/{icon_name}")
        self._btn.setIcon(make_themed_icon(str(icon_path), 12))
        self._btn.setStyleSheet(
            f"""
            QPushButton {{
                background: {c["handle_bg"]};
                border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: {c["handle_hover"]}; }}
        """
        )

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._apply_btn_theme()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._container.adjustSize()
        x = (self.width() - self._container.width()) // 2
        y = (self.height() - self._container.height()) // 2
        self._container.move(x, y)


class CollapsibleSplitter(QSplitter):
    """QSplitter that uses SidebarHandle for its divider."""

    def createHandle(self):
        handle = SidebarHandle(self.orientation(), self)
        handle._btn.clicked.connect(lambda: self.window().toggle_side_panel())
        return handle


class MainWindow(QMainWindow):
    """
    Main Application Window.
    """

    # pylint: disable=too-many-instance-attributes, attribute-defined-outside-init
    def __init__(self, parent=None, project_path: Optional[Path] = None, cli_args=None):
        QImageReader.setAllocationLimit(0)
        super().__init__(parent)

        self._init_attributes()

        self.current_project_path = project_path

        self.args = argparse.Namespace(
            image=getattr(cli_args, "image", None),
            reference=getattr(cli_args, "reference", None),
        )

        self._setup_main_window()
        self._add_shortcuts()

        self._setup_central_widget()
        self._setup_menubar_and_toolbar()
        self._setup_side_panel()
        self._setup_canvas()

        # Tab setup!
        self._setup_images_tab()
        self._setup_view_tab()
        self._setup_analysis_tab()
        self._setup_metadata_tab()

        self._setup_status_bar()
        self._setup_layout()

        self._connect_signals()
        self._retranslate_ui()
        QMetaObject.connectSlotsByName(self)

        if self.current_project_path:
            self._load_project(self.current_project_path)

        if cli_args is not None:
            self._apply_cli_args(cli_args)

    def _init_attributes(self):
        """Initialize instance attributes."""
        self.custom_windows_chrome_enabled: bool = False
        self.menu_bar: Optional[MenuBarUI] = None
        self.tool_bar: Optional[ToolBarUI] = None
        self.central_widget_layout: Optional[QHBoxLayout] = None
        self.main_layout: Optional[QHBoxLayout] = None
        self.side_panel_container: Optional[QWidget] = None
        self.side_panel: Optional[QWidget] = None
        self.side_panel_layout: Optional[QVBoxLayout] = None
        self.stacked_widget: Optional[QStackedWidget] = None
        self.canvas: Optional[ImageGraphicsViewUI] = None
        self.small_view: Optional[ReferenceGraphicsViewUI] = None
        self.images_tab_container: Optional[QWidget] = None
        self.images_tab: Optional[ImageManager] = None
        self.processing_tabs: Optional[QTabWidget] = None
        self.crop_groupbox: Optional[CropUI] = None
        self.rotate_groupbox: Optional[RotateUI] = None
        self.flip_groupbox: Optional[QGroupBox] = None
        self.flip_horizontal_btn: Optional[QPushButton] = None
        self.flip_vertical_btn: Optional[QPushButton] = None
        self.cell_layer_alignment: Optional[CellLayerAlignmentUI] = None
        self.register_groupbox: Optional[RegisterUI] = None
        self.gaussian_blur: Optional[GaussianBlur] = None
        self.stardist_groupbox: Optional[StarDistUI] = None
        self.cell_intensity_groupbox: Optional[CellIntensityUI] = None
        self.view_tab: Optional[ImageOverlay] = None
        self.analysis_tab: Optional[AnalysisTab] = None
        self.metadata_tab: Optional[MetaData] = None
        self.statusbar: Optional[QStatusBar] = None
        self.progress_bar_label: Optional[QLabel] = None
        self.progress_bar: Optional[QProgressBar] = None
        self.metadata: Optional[dict] = None
        self.splitter: Optional[QSplitter] = None
        self.last_sidebar_width: int = 400
        self.current_project_path: Optional[Path] = None
        self.log_dialog = None

    def _apply_cli_args(self, cli_args) -> None:
        seg_path = getattr(cli_args, "segmentation", None)
        cell_data_path = getattr(cli_args, "cell_data", None)
        if seg_path and cell_data_path:
            self._apply_view_tab_args(seg_path, cell_data_path)

    def _apply_view_tab_args(self, seg_path: str, cell_data_path: str) -> None:
        from pathlib import Path

        import pandas as pd

        if not Path(seg_path).is_file():
            logger.warning("CLI -s path not found: %s", seg_path)
            return
        if not Path(cell_data_path).is_file():
            logger.warning("CLI -c path not found: %s", cell_data_path)
            return

        view = self.view_tab

        view.im_path = seg_path
        view.open_image_label.setText(
            f"File: {view.less_than_15_chars(os.path.basename(seg_path))}"
        )
        view.open_image_label.setVisible(True)

        view.df_path = cell_data_path
        view.open_df_label.setText(
            f"File: {view.less_than_15_chars(os.path.basename(cell_data_path))}"
        )
        view.open_df_label.setVisible(True)

        try:
            if cell_data_path.endswith("csv"):
                df = pd.read_csv(cell_data_path)
            elif cell_data_path.endswith("xlsx"):
                df = pd.read_excel(cell_data_path)
            else:
                logger.warning("CLI -c: unsupported format %s", cell_data_path)
                return
            df = df[df.columns.drop(list(df.filter(regex="N/A")))]
            view.loaded_df = df
        except Exception as exc:
            logger.error("CLI -c load failed: %s", exc)
            return

        self.stacked_widget.setCurrentIndex(1)
        self.tool_bar.tab_buttons[1].setChecked(True)
        self.tool_bar.onTabButtonClicked(1)

    def _setup_main_window(self):
        """Setup main window properties"""
        self.setWindowIcon(QIcon(resource_path("assets/final_icon.png")))
        self.resize(1440, 1000)
        self.setMinimumSize(1200, 800)

    def _load_project(self, project_path: Path):
        """Load an existing project."""
        from core.canvas import ImageWrapper

        self.current_project_path = project_path

        metadata = ProjectManager.load_project(project_path)
        if metadata is None:
            return

        self.setWindowTitle(f"MIST-Explorer - {metadata.name}")

        if self.images_tab:
            self.images_tab.set_project_path(project_path)

        for image_meta in metadata.images:
            image_data = {}
            channel_names = ProjectManager.list_saved_channels(
                project_path, image_meta.uuid
            )
            if not channel_names:
                channel_names = [
                    f"Channel {channel_num}"
                    for channel_num in range(1, image_meta.channel_count + 1)
                ]

            for channel_name in channel_names:
                channel_array = ProjectManager.load_image(
                    project_path, image_meta.uuid, channel_name
                )
                if channel_array is None:
                    # Fallback: load from original source file (reference-only save)
                    channel_array = ProjectManager.load_channel_from_source(
                        image_meta.original_filename, channel_name
                    )
                if channel_array is not None:
                    contrast = image_meta.contrast_settings.get(channel_name, (0, 255))
                    display_name = image_meta.channel_display_names.get(
                        channel_name, channel_name
                    )
                    stored_cmap = image_meta.channel_cmaps.get(channel_name)
                    if is_segmentation_name(display_name):
                        channel_cmap = "label_image"
                    else:
                        channel_cmap = stored_cmap or "gray"
                    wrapper = ImageWrapper(
                        channel_array,
                        name=display_name,
                        cmap=channel_cmap,
                    )
                    wrapper.contrast_min = contrast[0]
                    wrapper.contrast_max = contrast[1]
                    if channel_cmap == "label_image":
                        wrapper.is_virtual_segmentation = True
                    image_data[channel_name] = wrapper

            if image_data:
                image_uuid = str(image_meta.uuid)
                self.images_tab.storage.add_data(
                    image_uuid,
                    {
                        "name": image_meta.name,
                        "data": image_data,
                        "original_filename": image_meta.original_filename,
                    },
                )
                self.images_tab.add_item(image_uuid)

    def open_project_folder(self):
        """Open the current project folder in file explorer."""
        if self.current_project_path:
            ProjectManager.open_project_folder(self.current_project_path)

    def _prompt_save_unsaved_created(self) -> bool:
        """Prompt the user to save unsaved created images if any exist.

        Returns True to proceed (save or discard), False if the user cancelled.
        """
        if self.images_tab is None or not self.images_tab.has_unsaved_created_images():
            return True
        if not self.images_tab._can_save_project():
            # Temp project or no project path — skip silently
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Work",
            "You have unsaved images created in this session (e.g. crops).\n\n"
            "Save them to the project before exiting?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self.images_tab.save_created_images_to_project()
        return True

    def closeEvent(self, event):  # type: ignore
        if self.images_tab is not None:
            self.images_tab.save_all_images(copy_data=False)
        if self._prompt_save_unsaved_created():
            event.accept()
        else:
            event.ignore()

    def save_all_images_to_folder(self):
        """Open a folder picker and export all workspace images as TIFFs there."""
        if self.images_tab is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save All Images"
        )
        if folder:
            self.images_tab.save_all_images_to(folder)

    def switch_project(self):
        """Cleanly restart the application to return to the project launcher."""
        logger.info("Switching project. Restarting application...")
        if not self._prompt_save_unsaved_created():
            return
        os.execl(sys.executable, sys.executable, *sys.argv)

    # pylint: disable=invalid-name
    def eventFilter(self, obj, event):  # type: ignore
        return super().eventFilter(obj, event)

    def _add_shortcuts(self):
        """Add keyboard shortcuts"""

        # Can add more shortcuts by adding tuple in form: (key_press_string, function)
        shortcuts = [
            ("R", self.select),
            ("C", self.circle_select),
            ("L", self.poly_select),
        ]

        for key_sequence, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            shortcut.activated.connect(slot)

    def _setup_central_widget(self):
        widget = QWidget(self)
        self.central_widget_layout = QHBoxLayout(widget)
        self.main_layout = QHBoxLayout()
        self.setCentralWidget(widget)

    def _setup_menubar_and_toolbar(self):
        self.menu_bar = MenuBarUI(self)

        self.tool_bar = ToolBarUI(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.tool_bar)
        self.setMenuBar(self.menu_bar)

    def _setup_side_panel(self):
        """Setup the collapsible side panel"""

        self.side_panel_container = QWidget(self.centralWidget())
        container_layout = QHBoxLayout(self.side_panel_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self.side_panel = QWidget(self.side_panel_container)
        self.side_panel_layout = QVBoxLayout(self.side_panel)
        # self.sidePanelLayout.setSpacing(10)
        self.side_panel.setMinimumWidth(400)

        self.side_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        # Create stacked widget for tabs
        self.stacked_widget = QStackedWidget(self.side_panel)
        self.side_panel_layout.addWidget(self.stacked_widget)

        # Add to container
        container_layout.addWidget(self.side_panel)

    def _setup_canvas(self):
        """Setup the main canvas and reference view"""
        from ui.canvas import ImageGraphicsViewUI, ReferenceGraphicsViewUI

        self.canvas = ImageGraphicsViewUI(self.centralWidget(), enc=self)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.small_view = ReferenceGraphicsViewUI(self.centralWidget())
        self.small_view.setParent(self.canvas)
        self.small_view.hide()
        self.canvas.reference_view = self.small_view

    def _setup_images_tab(self):
        """Setup the images workspace tab with an image manager and processing tabs."""
        from ui.image_manager import ImageManager

        images_scroll = self._create_scroll_area()

        # The main widget for the "Images" tab, containing the manager and the processing tabs
        self.images_tab_container = QWidget(self.side_panel)
        images_tab_layout = QVBoxLayout(self.images_tab_container)

        # Image Manager (the file tree)
        self.images_tab = ImageManager(self.canvas)

        if self.current_project_path:
            self.images_tab.set_project_path(self.current_project_path)

        # "Extract" page setup
        self.processing_tabs = QTabWidget(self.side_panel)
        splitter = QSplitter(Qt.Orientation.Vertical, self.side_panel)
        splitter.addWidget(self.images_tab)
        splitter.addWidget(self.processing_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setCollapsible(0, False)

        self.processing_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,  # height = sizeHint, won't grab extra space
        )
        self.processing_tabs.setMinimumHeight(200)

        # Add splitter to layout
        images_tab_layout.addWidget(splitter)

        # Add individual processing tabs
        self._setup_transform_tab()
        self._setup_alignment_tab()
        self._setup_segmentation_tab()
        self._setup_quantification_tab()

        images_scroll.setWidget(self.images_tab_container)
        self.stacked_widget.addWidget(images_scroll)

    def _setup_transform_tab(self):
        """Sets up the 'Transform' tab with Crop, Rotate, and Flip tools."""
        from ui.alignment.cell_layer_alignment_ui import CellLayerAlignmentUI
        from ui.processing.crop_ui import CropUI
        from ui.processing.rotation_ui import RotateUI

        transform_tab = QWidget()
        transform_layout = QVBoxLayout(transform_tab)
        transform_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        transform_layout.setContentsMargins(4, 4, 4, 4)

        # Crop and Rotate
        self.crop_groupbox = CropUI(transform_tab)
        self.rotate_groupbox = RotateUI(transform_tab)
        rotate_crop_hlayout = QHBoxLayout()
        rotate_crop_hlayout.addWidget(self.crop_groupbox.crop_groupbox)
        rotate_crop_hlayout.addWidget(self.rotate_groupbox.rotate_groupbox)
        transform_layout.addLayout(rotate_crop_hlayout)

        # Flip
        self.flip_groupbox = QGroupBox("Flip Image")
        flip_layout = QHBoxLayout(self.flip_groupbox)
        self.flip_horizontal_btn = QPushButton("Flip Horizontal")
        self.flip_vertical_btn = QPushButton("Flip Vertical")
        self.flip_horizontal_btn.clicked.connect(self.canvas.flip_horizontal)
        self.flip_vertical_btn.clicked.connect(self.canvas.flip_vertical)
        flip_layout.addWidget(self.flip_horizontal_btn)
        flip_layout.addWidget(self.flip_vertical_btn)
        transform_layout.addWidget(self.flip_groupbox)

        # Cell Layer Alignment
        self.cell_layer_alignment = CellLayerAlignmentUI(
            transform_layout,
            self.images_tab.storage,
            transform_tab,
        )
        self.cell_layer_alignment.aligner.progress.connect(self.update_progress_bar)

        transform_scroll = QScrollArea()
        transform_scroll.setWidget(transform_tab)
        transform_scroll.setWidgetResizable(True)
        transform_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.processing_tabs.addTab(transform_scroll, "Transform")

    def _setup_alignment_tab(self):
        """Sets up the 'Alignment' tab with Register tools."""
        from ui.alignment.register_ui import RegisterUI

        alignment_tab = QWidget()
        alignment_layout = QVBoxLayout(alignment_tab)
        alignment_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        alignment_layout.setContentsMargins(4, 4, 4, 4)

        # Register UI
        self.register_groupbox = RegisterUI(alignment_tab, alignment_layout)

        alignment_scroll = QScrollArea()
        alignment_scroll.setWidget(alignment_tab)
        alignment_scroll.setWidgetResizable(True)
        alignment_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.processing_tabs.addTab(alignment_scroll, "Alignment")

    def _setup_segmentation_tab(self):
        """Sets up the 'Segmentation' tab with Gaussian Blur and StarDist."""
        from ui.processing.gaussian_blur import GaussianBlur
        from ui.processing.segmentation_image_selector import \
            SegmentationImageSelector
        from ui.stardist.stardist_ui import StarDistUI

        segmentation_tab = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_tab)
        segmentation_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        segmentation_layout.setContentsMargins(4, 4, 4, 4)

        # Shared image + channel selector
        self.segmentation_image_selector = SegmentationImageSelector(segmentation_tab)
        segmentation_layout.addWidget(self.segmentation_image_selector)

        # Gaussian Blur
        self.gaussian_blur = GaussianBlur(segmentation_tab, segmentation_layout)

        # StarDist
        self.stardist_groupbox = StarDistUI(segmentation_tab, segmentation_layout)
        self.stardist_groupbox.hide_channel_selector_row()

        segmentation_scroll = QScrollArea()
        segmentation_scroll.setWidget(segmentation_tab)
        segmentation_scroll.setWidgetResizable(True)
        segmentation_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.processing_tabs.addTab(segmentation_scroll, "Segmentation")

    def _setup_quantification_tab(self):
        """Sets up the 'Quantification' tab with Cell Intensity."""
        from ui.alignment.cell_intensity_ui import CellIntensityUI

        quantification_tab = QWidget()
        quantification_layout = QVBoxLayout(quantification_tab)
        quantification_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        quantification_layout.setContentsMargins(4, 4, 4, 4)

        self.cell_intensity_groupbox = CellIntensityUI(
            quantification_tab, quantification_layout
        )

        quantification_scroll = QScrollArea()
        quantification_scroll.setWidget(quantification_tab)
        quantification_scroll.setWidgetResizable(True)
        quantification_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.processing_tabs.addTab(quantification_scroll, "Generation")

    def _setup_view_tab(self):
        """Setup the view tab"""
        from ui.view_tab import ImageOverlay

        view_scroll = self._create_scroll_area()

        self.view_tab = ImageOverlay(self.canvas, enc=self)
        self.view_tab.setObjectName("view_tab")
        self.view_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        view_scroll.setWidget(self.view_tab)
        self.stacked_widget.addWidget(view_scroll)

    def _setup_analysis_tab(self):
        """Setup the analysis tab"""
        from ui.analysis.AnalysisTab import AnalysisTab

        analysis_scroll = self._create_scroll_area()

        self.analysis_tab = AnalysisTab(self.canvas, self)
        self.analysis_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.analysis_tab.setObjectName("analysis_tab")

        analysis_scroll.setWidget(self.analysis_tab)
        self.stacked_widget.addWidget(analysis_scroll)

    def _setup_metadata_tab(self):
        """Setup the metadata tab"""
        metadata_scroll = self._create_scroll_area()

        self.metadata_tab = MetaData(self.canvas)
        self.metadata_tab.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        metadata_scroll.setWidget(self.metadata_tab)
        self.stacked_widget.addWidget(metadata_scroll)

    def _create_scroll_area(self):
        """Create a standardized scroll area for tabs"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll_area

    def _setup_status_bar(self):
        """Setup status bar with progress bar"""
        container = QWidget()
        self.statusbar = QStatusBar(self)
        self.setStatusBar(self.statusbar)

        self.progress_bar_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)

        progress_bar_layout = QHBoxLayout()
        progress_bar_layout.addWidget(self.progress_bar_label)
        progress_bar_layout.addWidget(self.progress_bar)
        container.setLayout(progress_bar_layout)

        self.statusbar.addPermanentWidget(container)

        # Progress bar height/margin — colors come from the global theme QSS
        self.progress_bar.setStyleSheet(
            "QProgressBar { height: 5px; margin-right: 30px; }"
        )

    def _setup_layout(self):
        """Setup the final layout structure"""
        # Create a horizontal splitter
        self.splitter = CollapsibleSplitter(Qt.Orientation.Horizontal)

        # Add sidebar container and canvas to splitter
        self.splitter.addWidget(self.side_panel_container)
        self.splitter.addWidget(self.canvas)

        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        # Set stretch factors so canvas takes available space
        self.splitter.setStretchFactor(1, 1)

        # Add splitter to main layout
        self.main_layout.addWidget(self.splitter)

        # Add main layout to central widget
        self.central_widget_layout.addLayout(self.main_layout)

    def _connect_signals(self):
        """Connect remaining signals"""
        # Connect cell intensity error signal to statusbar
        if self.cell_intensity_groupbox:
            self.cell_intensity_groupbox.errorSignal.connect(
                lambda msg: self.statusbar.showMessage(msg, 5000)
            )

        # Connect toolbar tab change signal
        # Start with Images tab
        self.stacked_widget.setCurrentIndex(0)

        # Modular save coordinator (kept for internal use)
        self._save_handlers = [self.images_tab.save_all_images]

        # Theme change: refresh non-QSS elements
        from ui.theme import ThemeManager
        ThemeManager.instance().theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, mode: str):
        """Refresh elements that the global QSS cannot reach."""
        # Sidebar handle buttons
        handle = self.splitter.handle(1) if self.splitter.count() > 1 else None
        if handle and hasattr(handle, "_apply_btn_theme"):
            handle._apply_btn_theme()
            handle._resize_btn._apply_theme()

    def _retranslate_ui(self):
        """Set UI text and translations"""
        _translate = QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MIST-Explorer"))

    # Event handlers and utility methods
    def update_mouse_position_label(self, text):
        """Update mouse position in toolbar"""
        self.tool_bar.statusLine.setText(text)

    def update_progress_bar(self, value, text):
        """Update progress bar with value and text"""
        self.progress_bar.setValue(value)
        self.progress_bar_label.setText(text + " ...")
        if value >= 100:
            self.progress_bar.hide()
            self.progress_bar_label.hide()
        else:
            self.progress_bar.show()
            self.progress_bar_label.show()

    def select(self):
        """Set rectangle selection mode"""
        logger.debug("selecting")
        self.canvas.select = "rect"

    def circle_select(self):
        """Set circle selection mode"""
        logger.debug("selecting")
        self.canvas.select = "circle"

    def poly_select(self):
        """Toggle polygon selection mode"""
        if self.canvas.select == "poly":
            self.canvas.select = False
            return
        self.canvas.select = "poly"

    def _sync_handle_button(self):
        """Update the handle pill button direction after a toggle."""
        handle = self.splitter.handle(1)
        if isinstance(handle, SidebarHandle):
            handle.set_collapsed(not self.side_panel.isVisible())

    def toggle_side_panel(self):
        """Toggle side panel visibility"""
        if self.side_panel.isVisible():
            # Save width before hiding, if it's substantial
            current_sizes = self.splitter.sizes()
            if current_sizes[0] > 100:
                self.last_sidebar_width = current_sizes[0]

            self.side_panel.hide()

            # Collapse splitter to just the handle width
            hw = self.splitter.handleWidth()
            total = sum(current_sizes)
            self.splitter.setSizes([hw, total - hw])
        else:
            self.side_panel.show()

            # Restore previous width
            current_sizes = self.splitter.sizes()
            total = sum(current_sizes)
            target = getattr(self, "last_sidebar_width", 400)
            self.splitter.setSizes([target, total - target])

        self._sync_handle_button()

    def show_log_dialog(self) -> None:
        """Show the application log dialog."""
        from ui.log_dialog import LogDialog

        if self.log_dialog is None:
            self.log_dialog = LogDialog(self)
        self.log_dialog.show()
        self.log_dialog.raise_()

    def get_metadata(self, metadata: dict):
        """Update metadata tab with new metadata"""
        self.metadata = metadata
        self.metadata_tab.populate_table(self.metadata)
