from PyQt6.QtGui import QPainter, QPen, QColor, QPolygon, QBrush
from PyQt6.QtCore import Qt, QPoint, QRect

class PolyLasso:
    def __init__(self):
        self.points = []  # Will store widget coordinates
        self.color = QColor(0, 0, 255, 100)  # Fill color (semi-transparent)
        self.line_color = QColor(0, 0, 255)  # Line color (solid)
        self.point_color = QColor(255, 0, 0)  # Point marker color (red)
        self.completed = False
        self.temp_line_point = None  # To draw a temporary line following the cursor
        self.point_size = 6  # Size of the point markers

    def add_point(self, point):
        """Add a point to the polygon"""
        if isinstance(point, QPoint):
            self.points.append(point)
        else:
            # Convert to QPoint if it's not already
            self.points.append(QPoint(int(point.x()), int(point.y())))
        
    def set_temp_point(self, point):
        """Set temporary point for line preview"""
        if point:
            if isinstance(point, QPoint):
                self.temp_line_point = point
            else:
                self.temp_line_point = QPoint(int(point.x()), int(point.y()))
        else:
            self.temp_line_point = None

    def get_polygon(self):
        """Get polygon from points"""
        return QPolygon(self.points)

    def draw(self, painter):
        """Draw the polygon with proper styling"""
        if not painter or len(self.points) == 0:
            return
        
        # Save the current painter state
        painter.save()
        
        # Draw lines between points
        if len(self.points) > 1:
            # Set up pen for lines
            pen = QPen(self.line_color)
            pen.setWidth(2)
            painter.setPen(pen)
            
            # Draw connected lines
            for i in range(len(self.points) - 1):
                painter.drawLine(self.points[i], self.points[i + 1])
            
            # If completed, close the polygon and fill it
            if self.completed:
                painter.drawLine(self.points[-1], self.points[0])
                
                # Fill the polygon
                painter.setBrush(QBrush(self.color))
                painter.drawPolygon(self.get_polygon())
            # Otherwise, draw preview line if we have a temp point
            elif self.temp_line_point:
                # Draw dashed line for preview
                dashPen = QPen(self.line_color)
                dashPen.setStyle(Qt.PenStyle.DashLine)
                dashPen.setWidth(2)
                painter.setPen(dashPen)
                painter.drawLine(self.points[-1], self.temp_line_point)
        
        # Draw points (on top of lines)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.point_color))
        
        for point in self.points:
            rect = QRect(point.x() - self.point_size//2, 
                         point.y() - self.point_size//2,
                         self.point_size, self.point_size)
            painter.drawEllipse(rect)
        
        # Draw temp point if it exists
        if self.temp_line_point and not self.completed:
            rect = QRect(self.temp_line_point.x() - self.point_size//2, 
                     self.temp_line_point.y() - self.point_size//2,
                     self.point_size, self.point_size)
            # Use a different color for the temp point
            painter.setBrush(QBrush(QColor(255, 165, 0)))  # Orange
            painter.drawEllipse(rect)
        
        # Restore the painter state
        painter.restore()
    
    def contains(self, point):
        """Check if the polygon contains the given point"""
        if len(self.points) < 3 or not self.completed:
            return False
            
        return self.get_polygon().containsPoint(point, Qt.FillRule.OddEvenFill)
        
    def bounding_rect(self):
        """Get the bounding rectangle of the polygon"""
        if len(self.points) < 1:
            return QRect()
            
        # If we only have one point, return a small rectangle around it
        if len(self.points) == 1:
            p = self.points[0]
            return QRect(p.x() - 5, p.y() - 5, 10, 10)
            
        return self.get_polygon().boundingRect()