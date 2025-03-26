from PyQt6.QtGui import QPainter, QPen, QColor, QPolygon, QBrush, QPainterPath, QPolygonF
from PyQt6.QtCore import Qt, QPoint, QRect, QPointF, QRectF
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPolygonItem
import random

class PolyLasso(QGraphicsPolygonItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []  # Store scene coordinates
        col = self._get_random_color()[:3]
        self.color = QColor(*col, 100)
        self.line_color = QColor(*col)  # Line color (solid)
        self.point_color = QColor(255, 0, 0)  # Point marker color (red)
        self.completed = False
        self.temp_line_point = None  # To draw a temporary line following the cursor
        self.point_size = 10  # Size of the point markers
        self.im_points = []  # Store image coordinates
        
        # Only make it selectable, not movable since it should move with the image
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        
        # Set the Z value to ensure it's drawn above the image
        self.setZValue(1)
        
        # Set up the appearance
        self.setPen(QPen(self.line_color, 2))
        self.setBrush(QBrush(self.color))

       
        
    def _get_random_color(self):
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 50)
    

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

    def set_temp_point(self, scene_point):
        """Set temporary point for line preview in scene coordinates"""
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
        
        # Draw lines between points
        if len(self.points) > 1:
            pen = QPen(self.line_color, 2)
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
                dash_pen = QPen(self.line_color, 2, Qt.PenStyle.DashLine)
                painter.setPen(dash_pen)
                painter.drawLine(self.points[-1], self.temp_line_point)
        
        # Draw points
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.point_color))
        
        for point in self.points:
            painter.drawEllipse(point, self.point_size/2, self.point_size/2)
        
        # Draw temp point
        if self.temp_line_point and not self.completed:
            painter.setBrush(QBrush(QColor(255, 165, 0)))  # Orange
            painter.drawEllipse(self.temp_line_point, self.point_size/2, self.point_size/2)

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
        
        if len(self.points) == 1:
            p = self.points[0]
            return QRectF(p.x() - self.point_size, p.y() - self.point_size,
                        self.point_size * 2, self.point_size * 2)
            
        # Calculate the bounding rect of all points including the temp point
        points = self.points.copy()
        if self.temp_line_point:
            points.append(self.temp_line_point)
            
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        
        min_x = min(xs) - self.point_size
        min_y = min(ys) - self.point_size
        max_x = max(xs) + self.point_size
        max_y = max(ys) + self.point_size
        
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def set_filled(self, filled):
        """Toggle the fill state of the polygon"""
        if self.completed:
            if filled:
                self.color.setAlpha(100)  # Fill with semi-transparency
            else:
                self.color.setAlpha(0)  # Make completely transparent
            self.setBrush(QBrush(self.color))  # Update the brush with new color
            self.update()  # Force a redraw