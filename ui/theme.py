"""
Centralized dark/light mode theme system for MIST-Explorer.

Usage:
    from ui.theme import ThemeManager, make_themed_icon

    # At app startup (main.py):
    ThemeManager.init(app)

    # Access colors:
    ThemeManager.instance().get("accent")
    ThemeManager.instance().get_current()  # full dict

    # Toggle:
    ThemeManager.instance().toggle()

    # React to changes:
    ThemeManager.instance().theme_changed.connect(my_handler)
"""

import json
import logging
from pathlib import Path

from PyQt6.QtCore import QByteArray, QObject, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPalette, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QApplication


logger = logging.getLogger(__name__)

_SETTINGS_DIR = Path.home() / ".mist-explorer"
_SETTINGS_FILE = _SETTINGS_DIR / "settings.json"


THEMES = {
    "DARK": {
        # Backgrounds
        "bg_primary": "#0b0c10",
        "bg_secondary": "#1f2833",
        "bg_tertiary": "#2d3436",
        "bg_hover": "rgba(255, 255, 255, 0.06)",
        "bg_pressed": "rgba(255, 255, 255, 0.12)",
        # Text
        "text_primary": "#c5c6c7",
        "text_secondary": "#888888",
        "text_on_accent": "#0b0c10",
        # Accent
        "accent": "#66fcf1",
        "accent_dim": "#45a29e",
        "accent_hover": "#7ffdfa",
        # Semantic
        "success": "#4CAF50",
        "success_hover": "#45a049",
        "danger": "#e57373",
        "danger_hover": "#ef5350",
        "warning": "#f5c518",
        "info": "#3b82f6",
        # Borders
        "border": "#2d3436",
        "border_focus": "#66fcf1",
        # Canvas
        "canvas_bg": "#000000",
        # Plot
        "plot_cmap": "plasma",
        "plot_palette": "tab20",
        "pg_bg": "#0b0c10",
        # Icons
        "icon_color": "#ffffff",
        # Badge
        "badge_view_bg": "#3b82f6",
        "badge_ref_bg": "#22c55e",
        "badge_label_bg": "#a855f7",
        "badge_fg": "#ffffff",
        # Syntax highlighting
        "syntax_keyword": "#7878fa",
        "syntax_function": "#28aa28",
        "syntax_string": "#dc7846",
        "syntax_comment": "#787878",
        # Sidebar handle
        "handle_bg": "#555555",
        "handle_hover": "#777777",
        "handle_text": "#ddd",
        # Tab checked underline
        "tab_checked": "#66fcf1",
        # Size grip
        "size_grip_color": "#2196F3",
        # --- Legacy UMAP keys (for backward compat) ---
        "bg_dark": "#0b0c10",
        "bg_panel": "#1f2833",
        "text_main": "#c5c6c7",
        "alert": "#ffaa00",
        "grid": "#2d3436",
    },
    "LIGHT": {
        # Backgrounds
        "bg_primary": "#ffffff",
        "bg_secondary": "#f0f2f5",
        "bg_tertiary": "#e0e3e8",
        "bg_hover": "rgba(0, 0, 0, 0.06)",
        "bg_pressed": "rgba(0, 0, 0, 0.12)",
        # Text
        "text_primary": "#2d3436",
        "text_secondary": "#636e72",
        "text_on_accent": "#ffffff",
        # Accent
        "accent": "#0984e3",
        "accent_dim": "#74b9ff",
        "accent_hover": "#0773c5",
        # Semantic
        "success": "#27ae60",
        "success_hover": "#219a52",
        "danger": "#e74c3c",
        "danger_hover": "#c0392b",
        "warning": "#e67e22",
        "info": "#2980b9",
        # Borders
        "border": "#dfe6e9",
        "border_focus": "#0984e3",
        # Canvas
        "canvas_bg": "#d0d0d0",
        # Plot
        "plot_cmap": "viridis",
        "plot_palette": "tab20",
        "pg_bg": "#ffffff",
        # Icons
        "icon_color": "#2d3436",
        # Badge
        "badge_view_bg": "#3b82f6",
        "badge_ref_bg": "#22c55e",
        "badge_label_bg": "#a855f7",
        "badge_fg": "#ffffff",
        # Syntax highlighting
        "syntax_keyword": "#0000ff",
        "syntax_function": "#008000",
        "syntax_string": "#a31515",
        "syntax_comment": "#6a9955",
        # Sidebar handle
        "handle_bg": "#c0c0c0",
        "handle_hover": "#a0a0a0",
        "handle_text": "#333",
        # Tab checked underline
        "tab_checked": "#0984e3",
        # Size grip
        "size_grip_color": "#2196F3",
        # --- Legacy UMAP keys (for backward compat) ---
        "bg_dark": "#ffffff",
        "bg_panel": "#f0f2f5",
        "text_main": "#2d3436",
        "alert": "#d63031",
        "grid": "#dfe6e9",
    },
}


class ThemeManager(QObject):
    """Singleton that manages app-wide theming."""

    theme_changed = pyqtSignal(str)

    _instance: "ThemeManager | None" = None

    def __init__(self, app: QApplication):
        super().__init__(app)
        self._app = app
        self._mode = "DARK"

    @classmethod
    def init(cls, app: QApplication) -> "ThemeManager":
        """Create the singleton and apply the initial theme."""
        if cls._instance is not None:
            return cls._instance
        inst = cls(app)
        cls._instance = inst

        inst._mode = inst.detect_system_theme()

        inst._apply()
        logger.info("ThemeManager initialized: %s", inst._mode)
        return inst

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            raise RuntimeError("ThemeManager.init(app) must be called first")
        return cls._instance

    # --- Public API ---

    @property
    def current_mode(self) -> str:
        return self._mode

    def get_current(self) -> dict:
        return THEMES[self._mode]

    def get(self, key: str) -> str:
        return THEMES[self._mode][key]

    def toggle(self):
        self.set_mode("LIGHT" if self._mode == "DARK" else "DARK")

    def set_mode(self, mode: str):
        if mode not in ("DARK", "LIGHT"):
            return
        if mode == self._mode:
            return
        self._mode = mode
        self._apply()
        self._save_preference(mode)
        self.theme_changed.emit(mode)
        logger.info("Theme switched to: %s", mode)

    def is_dark(self) -> bool:
        return self._mode == "DARK"

    # --- Internal ---

    def _apply(self):
        _apply_palette(self._app, THEMES[self._mode])
        qss = get_app_stylesheet(THEMES[self._mode], self._mode)
        self._app.setStyleSheet(qss)

    @staticmethod
    def detect_system_theme() -> str:
        palette = QApplication.palette()
        return "DARK" if palette.color(QPalette.ColorRole.Window).lightness() < 128 else "LIGHT"

    def _load_preference(self) -> str | None:
        try:
            if _SETTINGS_FILE.exists():
                data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                return data.get("theme")
        except Exception:
            pass
        return None

    def _save_preference(self, mode: str):
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            data = {}
            if _SETTINGS_FILE.exists():
                try:
                    data = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
                except Exception:
                    pass
            data["theme"] = mode
            _SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("Could not save theme preference", exc_info=True)


# ---------------------------------------------------------------------------
# QPalette — gives native-looking widgets proper dark/light colors
# ---------------------------------------------------------------------------

def _apply_palette(app: QApplication, c: dict):
    """Set QPalette so native widgets (buttons, inputs, menus) get correct
    colors without losing their platform-native look."""
    p = QPalette()

    _set = p.setColor
    CR = QPalette.ColorRole

    _set(CR.Window,          QColor(c["bg_primary"]))
    _set(CR.WindowText,      QColor(c["text_primary"]))
    _set(CR.Base,            QColor(c["bg_secondary"]))
    _set(CR.AlternateBase,   QColor(c["bg_tertiary"]))
    _set(CR.Text,            QColor(c["text_primary"]))
    _set(CR.Button,          QColor(c["bg_secondary"]))
    _set(CR.ButtonText,      QColor(c["text_primary"]))
    _set(CR.BrightText,      QColor(c["accent"]))
    _set(CR.Highlight,       QColor(c["accent"]))
    _set(CR.HighlightedText, QColor(c["text_on_accent"]))
    _set(CR.Link,            QColor(c["accent"]))
    _set(CR.PlaceholderText, QColor(c["text_secondary"]))
    _set(CR.ToolTipBase,     QColor(c["bg_secondary"]))
    _set(CR.ToolTipText,     QColor(c["text_primary"]))
    _set(CR.Light,           QColor(c["bg_tertiary"]))
    _set(CR.Mid,             QColor(c["border"]))
    _set(CR.Dark,            QColor(c["text_secondary"]))
    _set(CR.Midlight,        QColor(c["bg_secondary"]))
    _set(CR.Shadow,          QColor(c["bg_primary"]))

    # Disabled group
    CG = QPalette.ColorGroup
    _set(CG.Disabled, CR.WindowText,  QColor(c["text_secondary"]))
    _set(CG.Disabled, CR.Text,        QColor(c["text_secondary"]))
    _set(CG.Disabled, CR.ButtonText,  QColor(c["text_secondary"]))

    app.setPalette(p)


# ---------------------------------------------------------------------------
# App-wide QSS — *minimal*: only rules that QPalette cannot express
# ---------------------------------------------------------------------------

def get_app_stylesheet(c: dict, mode: str) -> str:
    """Minimal QSS that supplements QPalette.

    QPalette already handles: window/widget backgrounds, text colors,
    button colors, highlight colors, input base colors, disabled states,
    tooltip colors.  We only add rules here for things QPalette can't do
    (accent borders, custom sub-controls, specific hover overrides).
    """
    return f"""
    /* ── Toolbar & MenuBar (macOS ignores palette for native chrome) ── */
    QToolBar {{
        background-color: {c["bg_secondary"]};
        border-bottom: 1px solid {c["border"]};
        spacing: 4px;
    }}
    QMenuBar {{
        background-color: {c["bg_secondary"]};
        color: {c["text_primary"]};
    }}
    QMenuBar::item:selected {{
        background-color: {c["accent"]};
        color: {c["text_on_accent"]};
    }}
    QMenu {{
        background-color: {c["bg_secondary"]};
        color: {c["text_primary"]};
        border: 1px solid {c["border"]};
    }}
    QMenu::item:selected {{
        background-color: {c["accent"]};
        color: {c["text_on_accent"]};
    }}

    /* ── Tooltip border (palette can't set borders) ── */
    QToolTip {{
        border: 1px solid {c["border"]};
        padding: 4px;
    }}

    /* ── Splitter handle transparent ── */
    QSplitter::handle {{
        background: transparent;
    }}

    /* ── Progress bar chunk color ── */
    QProgressBar::chunk {{
        background-color: {c["accent"]};
    }}

    /* ── Scroll bar styling (palette doesn't style sub-controls) ── */
    QScrollBar:vertical {{
        border: none;
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background: {c["accent_dim"]};
        min-height: 20px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        border: none;
        height: 10px;
    }}
    QScrollBar::handle:horizontal {{
        background: {c["accent_dim"]};
        min-width: 20px;
        border-radius: 4px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* ── Scroll area: no extra border ── */
    QScrollArea {{
        border: none;
    }}

    /* ── Status bar subtle top border ── */
    QStatusBar {{
        border-top: 1px solid {c["border"]};
    }}
    """


# ---------------------------------------------------------------------------
# Matplotlib theme helper
# ---------------------------------------------------------------------------

def apply_matplotlib_theme(figure, ax=None):
    """Apply the current theme to a matplotlib figure/axes."""
    c = ThemeManager.instance().get_current()

    figure.patch.set_facecolor(c["bg_primary"])

    if ax:
        ax.set_facecolor(c["bg_primary"])
        ax.tick_params(colors=c["text_primary"], which="both")

        for spine in ax.spines.values():
            spine.set_edgecolor(c["accent_dim"])
            spine.set_linewidth(1.5)

        ax.xaxis.label.set_color(c["accent"])
        ax.yaxis.label.set_color(c["accent"])
        ax.title.set_color(c["accent"])

        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor(c["bg_secondary"])
            frame.set_edgecolor(c["border"])
            for text in legend.get_texts():
                text.set_color(c["text_primary"])


# ---------------------------------------------------------------------------
# SVG icon helper
# ---------------------------------------------------------------------------

def make_themed_icon(svg_path: str, size: int = 22):
    """Load an SVG and replace `currentColor` with the theme icon color."""
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import Qt

    color = ThemeManager.instance().get("icon_color")
    with open(svg_path, "r") as f:
        svg_text = f.read()
    svg_text = svg_text.replace("currentColor", color)
    renderer = QSvgRenderer(QByteArray(svg_text.encode()))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
