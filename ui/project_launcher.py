from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.project_manager import ProjectManager


class ProjectCard(QWidget):
    def __init__(self, metadata, launcher, parent=None):
        super().__init__(parent)
        self.metadata = metadata
        self.launcher = launcher
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(10)

        from ui.theme import ThemeManager

        tc = ThemeManager.instance().get_current()

        icon_label = QLabel()
        icon_label.setFixedSize(32, 32)
        icon = QPixmap(32, 32)
        icon.fill(QColor(tc["bg_tertiary"]))
        icon_label.setPixmap(icon)
        layout.addWidget(icon_label)

        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        muted = f"color: {tc['text_secondary']};"

        name_label = QLabel(self.metadata.name)
        font = name_label.font()
        font.setBold(True)
        name_label.setFont(font)

        date_label = QLabel(
            f"Modified: {self.metadata.modified_at.strftime('%Y-%m-%d %H:%M')}"
        )
        font = date_label.font()
        font.setPointSize(10)
        date_label.setFont(font)
        date_label.setStyleSheet(muted)

        image_count = len(self.metadata.images)
        count_label = QLabel(f"{image_count} image{'s' if image_count != 1 else ''}")
        font = count_label.font()
        font.setPointSize(10)
        count_label.setFont(font)
        count_label.setStyleSheet(muted)

        size_bytes = ProjectManager.get_folder_size(self.metadata.path)
        size_str = ProjectManager.format_size(size_bytes)
        size_label = QLabel(size_str)
        font = size_label.font()
        font.setPointSize(10)
        size_label.setFont(font)
        size_label.setStyleSheet(muted)

        info_layout.addWidget(name_label)
        info_layout.addWidget(date_label)
        info_layout.addWidget(count_label)
        info_layout.addWidget(size_label)
        layout.addLayout(info_layout)

        layout.addStretch()

        open_btn = QPushButton("Open")
        open_btn.setFixedWidth(70)
        open_btn.setFixedHeight(28)
        open_btn.clicked.connect(self._on_open)
        layout.addWidget(open_btn)

        delete_btn = QPushButton("Delete")
        delete_btn.setFixedWidth(70)
        delete_btn.setFixedHeight(28)
        delete_btn.clicked.connect(self._on_delete)
        layout.addWidget(delete_btn)

        self.setFixedHeight(80)

    def _on_open(self):
        self.launcher.set_selected_project(self.metadata.path)
        self.launcher.accept()

    def _on_delete(self):
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f'Are you sure you want to delete "{self.metadata.name}"?\n\nThis will permanently delete the project folder.',
            QMessageBox.StandardButton.No | QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            ProjectManager.delete_project(self.metadata.path)
            self.launcher.refresh_recent_projects()


class ProjectLauncher(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_project_path: Optional[Path] = None
        self.setup_ui()
        self.refresh_recent_projects()

    def setup_ui(self):
        self.setWindowTitle("MIST-Explorer")
        self.setMinimumSize(450, 350)
        self.setWindowFlags(Qt.WindowType.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title_label = QLabel("MIST-Explorer")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(18)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        layout.addSpacing(5)

        new_project_btn = QPushButton("New Project")
        new_project_btn.setFixedHeight(36)
        new_project_btn.clicked.connect(self._on_new_project)
        layout.addWidget(new_project_btn)

        temp_project_btn = QPushButton("Temp Project")
        temp_project_btn.setFixedHeight(32)
        temp_project_btn.clicked.connect(self._on_temp_project)
        layout.addWidget(temp_project_btn)

        layout.addSpacing(8)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        recent_label = QLabel("Recent Projects")
        font = recent_label.font()
        font.setPointSize(12)
        font.setBold(True)
        recent_label.setFont(font)
        layout.addWidget(recent_label)

        self.recent_list = QListWidget()
        self.recent_list.setFrameShape(QListWidget.Shape.Box)
        self.recent_list.setLineWidth(1)
        self.recent_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self.recent_list)

        open_folder_btn = QPushButton("Open Projects Folder")
        open_folder_btn.setFixedHeight(28)
        open_folder_btn.clicked.connect(self._on_open_folder)
        layout.addWidget(open_folder_btn)

    def refresh_recent_projects(self):
        self.recent_list.clear()
        projects = ProjectManager.get_recent_projects()

        if not projects:
            empty_label = QLabel("No recent projects")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_item = QListWidgetItem()
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            empty_item.setSizeHint(QSize(0, 50))
            self.recent_list.addItem(empty_item)
            self.recent_list.setItemWidget(empty_item, empty_label)
            return

        for project in projects:
            card = ProjectCard(project, self, self.recent_list)
            item = QListWidgetItem()
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setSizeHint(QSize(0, 80))
            self.recent_list.addItem(item)
            self.recent_list.setItemWidget(item, card)

    def _on_new_project(self):
        name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Enter project name:",
            text="Untitled Project",
        )
        if ok and name.strip():
            self.selected_project_path = ProjectManager.create_project(name.strip())
            self.accept()

    def _on_temp_project(self):
        self.selected_project_path = ProjectManager.create_temp_project()
        self.accept()

    def _on_open_folder(self):
        ProjectManager.ensure_storage_exists()
        ProjectManager.open_project_folder(ProjectManager.get_projects_path())

    def set_selected_project(self, path: Path):
        self.selected_project_path = path

    def get_selected_project(self) -> Optional[Path]:
        return self.selected_project_path
