import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ZoomableImageView(QGraphicsView):
    """Custom QGraphicsView with zoom and pan functionality"""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene())

        # Enable drag mode for panning
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        # Set view properties
        self.setRenderHint(self.renderHints() | self.renderHints().Antialiasing)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Zoom settings
        self.zoom_factor = 1.15
        self.min_zoom = 0.1
        self.max_zoom = 10000.0
        self.current_zoom = 1.0

        self.pixmap_item = None

    def set_image(self, pixmap):
        """Set the image to display"""
        scene = self.scene()
        assert scene is not None, "Scene should be initialized"
        scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(self.pixmap_item)
        scene.setSceneRect(
            pixmap.rect().x(), pixmap.rect().y(), pixmap.width(), pixmap.height()
        )

        # Fit image in view initially
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_zoom = 1.0

    def wheelEvent(self, event: QWheelEvent | None):
        """Handle mouse wheel for zooming"""
        if event is None:
            return
        # Get the position of the mouse cursor
        cursor_pos = event.position()
        scene_pos = self.mapToScene(cursor_pos.toPoint())

        # Determine zoom direction
        if event.angleDelta().y() > 0:
            zoom_factor = self.zoom_factor
        else:
            zoom_factor = 1.0 / self.zoom_factor

        # Check zoom limits
        new_zoom = self.current_zoom * zoom_factor
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return

        # Apply zoom
        self.scale(zoom_factor, zoom_factor)
        self.current_zoom = new_zoom

        # Keep the cursor position centered during zoom
        new_cursor_pos = self.mapFromScene(scene_pos)
        delta = cursor_pos.toPoint() - new_cursor_pos

        h_scrollbar = self.horizontalScrollBar()
        if h_scrollbar:
            h_scrollbar.setValue(h_scrollbar.value() - int(delta.x()))

        v_scrollbar = self.verticalScrollBar()
        if v_scrollbar:
            v_scrollbar.setValue(v_scrollbar.value() - int(delta.y()))

    def reset_zoom(self):
        """Reset zoom to fit the image in view"""
        if self.pixmap_item:
            self.resetTransform()
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.current_zoom = 1.0


class AlignmentPreviewDialog(QDialog):
    """Dialog to preview the alignment with zoom and pan functionality"""

    def __init__(self, target_image, aligned_image, parent=None):
        super().__init__(parent)
        self.target_image = target_image
        self.aligned_image = aligned_image
        self.result_accepted = False

        self.setWindowTitle("Alignment Preview - Use mouse wheel to zoom, drag to pan")
        self.resize(1000, 700)

        # Create layout
        self.setLayout(QVBoxLayout())

        # Create preview label
        self.preview_label = QLabel(
            "Red = Target, Green = Aligned | Mouse wheel: zoom, Drag: pan, Double-click: reset zoom"
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "font-weight: bold; font-size: 12px; margin: 5px;"
        )

        # Create NCC metric label
        self.ncc_label = QLabel("Calculating NCC...")
        self.ncc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ncc_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; margin: 5px; color: #2196F3; "
            "padding: 5px; border: 2px solid #2196F3; border-radius: 5px;"
        )

        # Create zoomable image view instead of QLabel
        self.image_view = ZoomableImageView()
        self.image_view.setMinimumSize(800, 500)

        # Add double-click to reset zoom
        self.image_view.mouseDoubleClickEvent = self.reset_zoom

        # Create control buttons
        self.control_layout = QHBoxLayout()

        # Create main action buttons
        self.button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Confirm Alignment")
        self.confirm_button.setStyleSheet(
            """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                min-width: 120px;
                min-height: 35px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        )
        self.confirm_button.clicked.connect(self.accept_alignment)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(
            """
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                min-width: 120px;
                min-height: 35px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """
        )
        self.cancel_button.clicked.connect(self.reject)

        # Arrange control buttons
        self.control_layout.addStretch()
        self.control_layout.addStretch()

        # Arrange main buttons
        self.button_layout.addWidget(self.confirm_button)
        self.button_layout.addWidget(self.cancel_button)

        # Add widgets to layout
        layout = self.layout()
        assert layout is not None, "Layout should be initialized"
        layout.addWidget(self.preview_label)
        layout.addWidget(self.ncc_label)
        layout.addWidget(self.image_view)
        control_widget = QWidget()
        control_widget.setLayout(self.control_layout)
        layout.addWidget(control_widget)
        buttons_widget = QWidget()
        buttons_widget.setLayout(self.button_layout)
        layout.addWidget(buttons_widget)

        # Create the overlay image and calculate NCC
        self.create_direct_overlay()

    def calculate_ncc(self, img1, img2):
        """
        Calculate NCC (Normalized Cross-Correlation) between two images.

        Args:
            img1: First image (reference/target)
            img2: Second image (aligned)

        Returns:
            NCC value between -1 and 1 (1 = perfect correlation)
        """
        try:
            # Ensure images have the same shape
            if img1.shape != img2.shape:
                min_h = min(img1.shape[0], img2.shape[0])
                min_w = min(img1.shape[1], img2.shape[1])
                img1 = img1[:min_h, :min_w]
                img2 = img2[:min_h, :min_w]

            # Convert to float to avoid overflow
            img1_float = img1.astype(np.float32)
            img2_float = img2.astype(np.float32)

            # Flatten images
            img1_flat = img1_float.flatten()
            img2_flat = img2_float.flatten()

            # Calculate means
            mean1 = np.mean(img1_flat)
            mean2 = np.mean(img2_flat)

            # Center the data
            img1_centered = img1_flat - mean1
            img2_centered = img2_flat - mean2

            # Calculate NCC
            numerator = np.sum(img1_centered * img2_centered)
            denominator = np.sqrt(np.sum(img1_centered**2) * np.sum(img2_centered**2))

            if denominator == 0:
                return 0.0  # No correlation if one image is constant

            ncc = numerator / denominator
            return ncc

        except Exception as e:
            print(f"Error calculating NCC: {str(e)}")
            return None

    def format_ncc_display(self, ncc_value):
        """Format NCC value for display"""
        if ncc_value is None:
            return "NCC: Unable to calculate"
        else:
            return f"NCC: {ncc_value:.4f}"

    def reset_zoom(self, event=None):
        """Reset zoom to fit image in view"""
        self.image_view.reset_zoom()

    def create_direct_overlay(self):
        """Create and display the overlay directly without using external libraries"""
        try:
            print("Creating direct overlay visualization...")
            # Ensure the images are available
            if self.target_image is None or self.aligned_image is None:
                raise ValueError("Missing one or both images for alignment preview")

            # Crop to same dimensions
            target_img, aligned_img = self._ensure_same_size(
                self.target_image, self.aligned_image
            )
            print(
                f"After cropping: target={target_img.shape}, aligned={aligned_img.shape}"
            )

            # Convert to grayscale if color
            if len(target_img.shape) > 2 and target_img.shape[2] > 1:
                target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
            else:
                target_gray = target_img

            if len(aligned_img.shape) > 2 and aligned_img.shape[2] > 1:
                aligned_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
            else:
                aligned_gray = aligned_img

            # Convert to uint8 if needed
            if target_gray.dtype != np.uint8:
                target_gray = self.to_uint8(target_gray)
            if aligned_gray.dtype != np.uint8:
                aligned_gray = self.to_uint8(aligned_gray)

            # Calculate NCC before contrast adjustment for more accurate measurement
            ncc_value = self.calculate_ncc(target_gray, aligned_gray)
            ncc_display = self.format_ncc_display(ncc_value)

            self.ncc_label.setText(ncc_display)

            # Adjust contrast for better visualization
            target_gray = self.adjust_contrast(target_gray)
            aligned_gray = self.adjust_contrast(aligned_gray)

            # Create RGB overlay
            h, w = target_gray.shape
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[:, :, 0] = target_gray  # Red channel = target
            overlay[:, :, 1] = aligned_gray  # Green channel = aligned

            # Convert to QImage
            print("Converting to QImage...")
            height, width, channels = overlay.shape
            bytes_per_line = channels * width
            q_image = QImage(
                overlay.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )

            # Convert to QPixmap and set in the zoomable view
            print("Setting pixmap...")
            pixmap = QPixmap.fromImage(q_image)
            self.image_view.set_image(pixmap)
            print(
                f"Overlay displayed successfully! NCC: {ncc_value:.4f}"
                if ncc_value is not None
                else "Overlay displayed successfully! NCC: Unable to calculate"
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error creating overlay: {str(e)}")
            self.ncc_label.setText("NCC: Error in calculation")

    def _ensure_same_size(self, img1, img2):
        """Ensure both images have the same dimensions by cropping to the smallest common size"""
        print(f"Original shapes: img1={img1.shape}, img2={img2.shape}")

        # Get the minimum width and height
        min_height = min(img1.shape[0], img2.shape[0])
        min_width = min(img1.shape[1], img2.shape[1])

        # Crop both images to these dimensions
        img1_cropped = img1[:min_height, :min_width]
        img2_cropped = img2[:min_height, :min_width]

        # If images are RGB (3 channels), make sure to keep the channel dimension
        if len(img1.shape) > 2:
            img1_cropped = img1_cropped[:min_height, :min_width, : img1.shape[2]]
        if len(img2.shape) > 2:
            img2_cropped = img2_cropped[:min_height, :min_width, : img2.shape[2]]

        print(f"Cropped shapes: img1={img1_cropped.shape}, img2={img2_cropped.shape}")

        return img1_cropped, img2_cropped

    def to_uint8(self, image):
        """Convert image to uint8 with proper scaling"""
        # Check if image is already uint8
        if image.dtype == np.uint8:
            return image

        # Convert to float and scale to 0-255
        img_float = image.astype(np.float32)
        if img_float.max() > img_float.min():  # Check to avoid division by zero
            img_norm = (img_float - img_float.min()) * (
                255.0 / (img_float.max() - img_float.min())
            )
            return img_norm.astype(np.uint8)
        else:
            return np.zeros_like(image, dtype=np.uint8)

    def adjust_contrast(self, img, min_percentile=2, max_percentile=98):
        """Adjust image contrast using percentile-based clipping"""
        # Calculate percentiles
        minval = np.percentile(img, min_percentile)
        maxval = np.percentile(img, max_percentile)

        # Clip and rescale
        img_adjusted = np.clip(img, minval, maxval)
        if maxval > minval:
            img_adjusted = ((img_adjusted - minval) / (maxval - minval)) * 255
        return img_adjusted.astype(np.uint8)

    def accept_alignment(self):
        """Set result as accepted and close dialog"""
        self.result_accepted = True
        self.accept()
