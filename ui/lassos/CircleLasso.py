import pyqtgraph as pg
from PyQt6.QtGui import QColor, QBrush, QPainter
from PyQt6.QtCore import QRectF, Qt

from ui.lassos.Lasso import pick_distinct_color


class CircleLasso(pg.CircleROI):
    def _addHandles(self):
        pass

    def __init__(self, pos, size, parent=None, existing_colors=None):
        col = pick_distinct_color(existing_colors or [])[:3]
        self.color = QColor(*col)
        self.fill_alpha = 75
        pen = pg.mkPen(color=col, width=2)
        super().__init__(pos, size, pen=pen)
        self._filled = False

    def set_filled(self, filled: bool):
        self._filled = filled
        self.update()

    def set_fill_opacity(self, opacity: float):
        self.fill_alpha = int(opacity * 255)
        self.update()

    def paint(self, p, opt, widget):
        r = self.boundingRect()
        if r.width() <= 0 or r.height() <= 0:
            return
        if self._filled:
            p.save()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            c = QColor(self.color)
            c.setAlpha(self.fill_alpha)
            p.setBrush(QBrush(c))
            p.scale(r.width(), r.height())
            er = QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)
            p.drawEllipse(er)
            p.restore()
        super().paint(p, opt, widget)

    def remove_from_scene(self):
        scene = self.scene()
        if scene:
            scene.removeItem(self)
