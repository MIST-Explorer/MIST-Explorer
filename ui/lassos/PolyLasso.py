from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPolygonItem,
    QStyleOptionGraphicsItem,
)

from ui.lassos.Lasso import pick_distinct_color


class PolyLasso(QGraphicsPolygonItem):
    def __init__(self, parent=None, existing_colors=None):
        super().__init__(parent)
        self.points = []  # Store scene coordinates
        col = pick_distinct_color(existing_colors or [])[:3]
        self.color = QColor(*col, 100)
        self.fill_alpha = 100
        self.line_color = QColor(*col)  # Line color (solid)
        self.point_color = QColor(255, 0, 0)  # Point marker color (red)
        self.completed = False
        self.temp_line_point = None  # To draw a temporary line following the cursor
        self.point_size = 2  # Size of the point markers
        self.im_points = []  # Store image coordinates
        self.snap_point = None  # Point currently being highlighted as snap target

        # Only make it selectable, not movable since it should move with the image
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        # Set the Z value to ensure it's drawn above the image
        self.setZValue(1)

        # Set up the appearance
        self.setPen(QPen(self.line_color, 2))
        self.setBrush(QBrush(self.color))

    def add_point(self, scene_point, image_point=None):
        """Add a point to the polygon in scene coordinates"""
        if isinstance(scene_point, QPoint):
            scene_point = QPointF(scene_point)
        self.points.append(scene_point)

        if image_point:
            self.im_points.append(image_point)

        # Update the polygon
        self.update_polygon()

    def update_polygon(self):
        """Update the polygon with current points"""
        polygon = QPolygonF(self.points)
        self.setPolygon(polygon)

    def set_snap_point(self, point):
        """Highlight a point as the current snap target (None to clear)."""
        self.prepareGeometryChange()
        self.snap_point = point
        self.update()

    def set_temp_point(self, scene_point):
        """Set temporary point for line preview in scene coordinates"""
        self.prepareGeometryChange()
        if scene_point:
            if isinstance(scene_point, QPoint):
                self.temp_line_point = QPointF(scene_point)
            else:
                self.temp_line_point = scene_point
        else:
            self.temp_line_point = None
        self.update()

    def paint(self, painter, option, widget=None):
        """Custom paint implementation"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Compute scene-space radius that stays constant in screen pixels at any zoom
        lod = QStyleOptionGraphicsItem.levelOfDetailFromTransform(
            painter.worldTransform()
        )
        r = self.point_size / lod

        # Draw lines between points
        if len(self.points) > 1:
            pen = QPen(self.line_color, 2 / lod)
            painter.setPen(pen)

            # Draw connected lines
            path = QPainterPath()
            path.moveTo(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)

            # If completed, close the polygon and fill it
            if self.completed:
                path.lineTo(self.points[0])
                painter.fillPath(path, QBrush(self.color))

            painter.drawPath(path)

            # Draw preview line if we have a temp point
            if not self.completed and self.temp_line_point and self.points:
                dash_pen = QPen(self.line_color, 2 / lod, Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(self.points[-1], self.temp_line_point)

        # Draw points (constant screen size)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.point_color))

        for point in self.points:
            painter.drawEllipse(point, r, r)

        # Draw snap indicator ring (constant screen size, 2× the point radius)
        if self.snap_point is not None and not self.completed:
            snap_pen = QPen(QColor(255, 255, 255), 2)
            painter.setPen(snap_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(self.snap_point, r * 2, r * 2)

    def complete(self):
        """Complete the polygon"""
        if len(self.points) >= 3:
            self.completed = True
            self.update_polygon()
            self.temp_line_point = None
            self.update()
            return True
        return False

    def contains_scene_point(self, scene_point):
        """Check if the polygon contains the given scene point"""
        if len(self.points) < 3 or not self.completed:
            return False
        return self.polygon().containsPoint(scene_point, Qt.FillRule.OddEvenFill)

    def boundingRect(self):
        """Get the bounding rectangle of the polygon"""
        if not self.points:
            return QRectF()

        # Always include temp_line_point so Qt invalidates the full drawn area
        points = self.points.copy()
        if self.temp_line_point:
            points.append(self.temp_line_point)

        xs = [p.x() for p in points]
        ys = [p.y() for p in points]

        min_x = min(xs) - self.point_size * 2
        min_y = min(ys) - self.point_size * 2
        max_x = max(xs) + self.point_size * 2
        max_y = max(ys) + self.point_size * 2

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def set_filled(self, filled):
        """Toggle the fill state of the polygon"""
        if self.completed:
            if filled:
                self.color.setAlpha(self.fill_alpha)  # Fill with semi-transparency
            else:
                self.color.setAlpha(0)  # Make completely transparent
            self.setBrush(QBrush(self.color))  # Update the brush with new color
            self.update()  # Force a redraw

    def set_fill_opacity(self, opacity: float):
        self.fill_alpha = int(opacity * 255)
        if self.completed and self.brush().color().alpha() > 0:
            self.color.setAlpha(self.fill_alpha)
            self.setBrush(QBrush(self.color))
        self.update()
