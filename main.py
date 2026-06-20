"""
Main script for starting the MIST-Explorer application.
"""
import argparse
import io
import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Log directory ---
_log_dir = Path.home() / ".mist-explorer" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
_log_file = _log_dir / "mist-explorer.log"
_crash_file = _log_dir / "crash.log"


def setup_logging(console_logs=False):
    """Configure logging for the application."""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    if console_logs:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.WARNING)
    handlers.append(console_handler)

    # File handler (if not console-only)
    if not console_logs:
        file_handler = RotatingFileHandler(
            _log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)

    # Use force=True to reconfigure if basicConfig was already called
    logging.basicConfig(level=logging.WARNING, handlers=handlers, force=True)
    for name in ("core", "ui", "controller", "UMAP_App", "MIST_UMAP", "MIST_MAIN"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    logger = logging.getLogger("MIST_MAIN")
    logger.info("=== MIST-Explorer started ===")
    if not console_logs:
        logger.info(f"Log file: {_log_file}")


# --- Global crash handler ---
def _crash_handler(exc_type, exc_value, exc_tb):
    """Write unhandled exceptions to crash.log and the main log, then exit."""
    logger = logging.getLogger("MIST_MAIN")
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.critical(f"Unhandled exception:\n{tb_text}")
    try:
        with open(_crash_file, "w", encoding="utf-8") as f:
            f.write(f"MIST-Explorer Crash Report\n{'=' * 40}\n\n{tb_text}")
    except Exception:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _crash_handler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()
try:
    import pyi_splash
except ImportError:
    class PyiSplashMock:
        @staticmethod
        def close():
            pass
    pyi_splash = PyiSplashMock()


class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setMinimumSize(300, 100)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.label = QLabel("Loading MIST-Explorer...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def update_progress(self, value, text):
        self.progress_bar.setValue(value)
        self.label.setText(text)
        QCoreApplication.processEvents()


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temp",
        action="store_true",
        help="Start immediately with a temporary project.",
    )
    parser.add_argument("-i", "--image", default=None,
        help="Image for Extract tab.")
    parser.add_argument("-s", "--segmentation", default=None,
        help="Segmentation mask (PNG/TIFF) for View tab.")
    parser.add_argument("-c", "--cell-data", dest="cell_data", default=None,
        help="Cell data CSV/XLSX for View tab.")
    parser.add_argument("-r", "--reference", default=None,
        help="Reference image for Extract tab.")
    parser.add_argument(
        "--console-logs",
        action="store_true",
        help="Redirect logs to console/stdout instead of a file.",
    )
    return parser.parse_known_args(argv)


def launch_main_window(app: QApplication, project_path: Path, cli_args=None):
    loading_dialog = LoadingDialog()
    loading_dialog.show()
    app.processEvents()

    steps = [
        ("Loading numpy...", 20),
        ("Initializing core modules...", 40),
        ("Loading image processing...", 50),
        ("Loading analysis tools...", 60),
        ("Loading UI components...", 80),
        ("Starting application...", 90),
    ]

    for message, value in steps:
        loading_dialog.update_progress(value, message)
        app.processEvents()

    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = np.bool_

    from controller import Controller
    from ui import app as ui_app
    from ui.theme import ThemeManager

    # Initialize app-wide theme (no-op if already done)
    ThemeManager.init(app)

    loading_dialog.update_progress(100, "Done")
    app.processEvents()

    window = ui_app.MainWindow(project_path=project_path, cli_args=cli_args)
    Controller.init(window)
    window.show()
    loading_dialog.hide()

    app.exec()


if __name__ == "__main__":
    cli_args, qt_args = parse_cli_args(sys.argv[1:])
    setup_logging(cli_args.console_logs)
    __app = QApplication([sys.argv[0], *qt_args])

    loading_dialog = LoadingDialog()
    loading_dialog.show()
    try:
        pyi_splash.close()
    except Exception:
        pass
    __app.processEvents()

    loading_dialog.update_progress(5, "Initializing...")
    __app.processEvents()

    loading_dialog.update_progress(8, "Applying theme...")
    __app.processEvents()

    from ui.theme import ThemeManager
    ThemeManager.init(__app)

    loading_dialog.update_progress(10, "Loading project manager...")
    __app.processEvents()

    from core.project_manager import ProjectManager

    loading_dialog.update_progress(15, "Loading recent projects...")
    __app.processEvents()

    projects = ProjectManager.get_recent_projects()

    loading_dialog.accept()
    __app.processEvents()

    if (cli_args.image or cli_args.segmentation) and not cli_args.temp:
        cli_args.temp = True

    if cli_args.temp:
        launch_main_window(__app, ProjectManager.create_temp_project(), cli_args)
    else:
        from ui.project_launcher import ProjectLauncher

        launcher = ProjectLauncher()
        if launcher.exec() == ProjectLauncher.DialogCode.Accepted:
            project_path = launcher.get_selected_project()
            if project_path:
                launch_main_window(__app, project_path, cli_args)
        else:
            sys.exit(0)
