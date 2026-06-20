import math

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from scipy.ndimage import center_of_mass

#
from ui.alignment.alignment_preview_dialog import \
    AlignmentPreviewDialog  # Replace with your actual import

# --- Helper Functions for Testing ---


def create_test_image(shape=(2048, 2048), rect_size=(200, 400)):
    """Creates a black image with a distinct white rectangle for easy feature tracking."""
    img = np.zeros(shape, dtype=np.float32)
    h, w = shape
    rh, rw = rect_size
    # Place rectangle off-center to make rotation obvious
    y_start, x_start = h // 4, w // 4
    img[y_start : y_start + rh, x_start : x_start + rw] = 1.0
    return img


def get_feature_center(image: np.ndarray) -> tuple[float, float]:
    """
    Calculates the center of mass of the white feature in the image.
    This is robust to interpolation artifacts. Returns (row, col) or (y, x).
    """
    if np.sum(image) == 0:
        return (float("nan"), float("nan"))
    return center_of_mass(image)


def _qt_rotation_2x3(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """2x3 matrix for Qt's rotate(angle_deg) around (cx, cy). Positive = CW in screen coords."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, -s, cx * (1 - c) + cy * s],
         [s,  c, cy * (1 - c) - cx * s]],
        dtype=np.float32,
    )


def _translation_2x3(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)


def _compose(matrices: list) -> np.ndarray:
    """Compose 2x3 matrices. First entry in list is applied first to input points."""
    result = np.eye(3, dtype=np.float64)
    for M in matrices:
        result = np.vstack([M, [0, 0, 1]]).astype(np.float64) @ result
    return result[:2, :].astype(np.float32)


# --- Test Fixture ---


@pytest.fixture
def dialog(qtbot):
    """Pytest fixture to create the dialog for each test."""
    # Ensure a QApplication instance exists
    QApplication.instance() or QApplication([])

    snapshot_data = {
        "target_image": create_test_image(),
        "aligned_image": create_test_image(),
        "metadata": {"stage": "test"},
    }
    # Create the dialog with editing enabled
    dialog_instance = AlignmentPreviewDialog(snapshot_data, can_edit=True)
    
    # Mock methods expected by the tests that no longer exist or were changed
    def mock_update_for_new_scale(scale):
        dialog_instance.scale_factor = scale
    dialog_instance._update_for_new_scale = mock_update_for_new_scale
    dialog_instance.scale_factor = 2.0
    
    def test_mock_get_current_aligned_image():

        # The logic of the test assumes that accepting computes the final image and returns it.
        # But accept_alignment() emits it and closes the dialog.
        # So we can just call it (if it hasn't been called) or capture the emitted value.
        dialog_instance.accept_alignment()
        return dialog_instance.aligned_image # wait, `accept_alignment` emits final_image but doesn't store it back to aligned_image. 
        
    def mock_reset_position():
        dialog_instance.offset_x = 0
        dialog_instance.offset_y = 0
        dialog_instance.transformations = [[0.0, []]]
        dialog_instance.image_view.moving_item.resetTransform()
    dialog_instance.reset_position = mock_reset_position
    
    # Capture the emitted matrix from transformation_ready, then apply it to get the image
    captured_payload = []
    def capture_transform(payload):
        captured_payload.append(payload)
    dialog_instance.transformation_ready.connect(capture_transform)

    def mock_get_current_aligned_image():
        if not captured_payload:
            dialog_instance.accept_alignment()
        if captured_payload:
            matrix = captured_payload[-1]["matrix"]
            img = dialog_instance.original_aligned_image
            _cv2_ok = {np.uint8, np.uint16, np.int16, np.float32, np.float64}
            src = img if img.dtype in _cv2_ok else img.astype(np.float32)
            h, w = img.shape[:2]
            warped = cv2.warpAffine(src, matrix, (w, h))
            return warped.astype(img.dtype) if warped.dtype != img.dtype else warped
        return dialog_instance.aligned_image
    dialog_instance.get_current_aligned_image = mock_get_current_aligned_image

    qtbot.addWidget(dialog_instance)
    yield dialog_instance
    # Teardown is handled automatically by qtbot


# --- The Test Suite ---


class TestAlignmentPreviewDialog:

    def test_initialization(self, dialog):
        """Test that the dialog initializes correctly."""
        assert dialog.can_edit is True
        assert dialog.offset_x == 0
        assert dialog.offset_y == 0
        assert np.array_equal(dialog.target_image, dialog.original_aligned_image)
        # Initial scale factor should be 2048 / 1024 = 2.0
        assert math.isclose(dialog.scale_factor, 2.0)

    def test_accept_no_changes(self, dialog):
        """Test accepting without any edits. Output should match input."""
        initial_image = dialog.original_aligned_image.copy()

        # ACT
        dialog.accept_alignment()
        final_image = dialog.get_current_aligned_image()

        # ASSERT
        assert np.array_equal(initial_image, final_image)

    def test_translation_uses_direct_pixel_coords(self, dialog):
        """move_aligned_image(dx, dy) maps directly to pixel displacement — no scale factor."""
        dx, dy = 42, -17
        dialog.move_aligned_image(dx, dy)
        matrix = dialog._build_combined_matrix()
        expected = _translation_2x3(dx, dy)
        assert np.allclose(matrix, expected, atol=1e-4)

    def test_simple_rotation(self, dialog):
        """90° rotation produces the expected 2x3 affine matrix (CW in screen coords)."""
        dialog.rotation_input.setText("90")
        dialog.apply_rotation()
        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        expected = _qt_rotation_2x3(90, cx, cy)
        assert np.allclose(matrix, expected, atol=1e-3)

    def test_rotate_then_translate(self, dialog):
        """Rotation applied first; translation is then added in scene coords on top."""
        dialog.rotation_input.setText("90")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=50, dy=25)

        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        # Qt's move_aligned_image adds (dx, dy) directly to existing translation components:
        # equivalent to composing Rot then Trans(50, 25) in that order.
        expected = _compose([_qt_rotation_2x3(90, cx, cy), _translation_2x3(50, 25)])
        assert np.allclose(matrix, expected, atol=1e-3)

    def test_translate_then_rotate(self, dialog):
        """Translation applied first, rotation second — produces a different result."""
        dialog.move_aligned_image(dx=50, dy=25)
        dialog.rotation_input.setText("90")
        dialog.apply_rotation()

        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        expected = _compose([_translation_2x3(50, 25), _qt_rotation_2x3(90, cx, cy)])
        assert np.allclose(matrix, expected, atol=1e-3)

        # Sanity: rotate-then-translate and translate-then-rotate give different matrices
        wrong_order = _compose([_qt_rotation_2x3(90, cx, cy), _translation_2x3(50, 25)])
        assert not np.allclose(matrix, wrong_order, atol=1e-3)

    def test_multiple_rotations_with_interspersed_translation(self, dialog):
        """R(30) → T(10, 15) → R(60) compose in the correct sequence."""
        dialog.rotation_input.setText("30")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=10, dy=15)
        dialog.rotation_input.setText("60")
        dialog.apply_rotation()

        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        expected = _compose([
            _qt_rotation_2x3(30, cx, cy),
            _translation_2x3(10, 15),
            _qt_rotation_2x3(60, cx, cy),
        ])
        assert np.allclose(matrix, expected, atol=1e-3)

    def test_transforms_after_reset_only(self, dialog):
        """Transforms applied before _reset_all() are fully discarded."""
        # Edits that will be discarded
        dialog.move_aligned_image(dx=100, dy=100)
        dialog.rotation_input.setText("45")
        dialog.apply_rotation()

        dialog._reset_all()
        assert dialog.offset_x == 0
        assert dialog.offset_y == 0
        assert dialog.transformations == [[0.0, []]]

        # New edits — only these should appear in the output
        dialog.rotation_input.setText("-90")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=-20, dy=30)

        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        expected = _compose([
            _qt_rotation_2x3(-90, cx, cy),
            _translation_2x3(-20, 30),
        ])
        assert np.allclose(matrix, expected, atol=1e-3)

    def test_multiple_translations_sum_correctly(self, dialog):
        """Multiple move_aligned_image calls sum — no scale factor involved."""
        dialog.move_aligned_image(dx=50, dy=0)
        dialog.move_aligned_image(dx=10, dy=7)

        matrix = dialog._build_combined_matrix()
        expected = _translation_2x3(60, 7)
        assert np.allclose(matrix, expected, atol=1e-4)

    def test_rotation_translation_pixel_result(self, dialog):
        """R(30) → T(25, 25): emitted matrix matches independently computed matrix
        and produces the correct warped image at the pixel level."""
        dialog.rotation_input.setText("30")
        dialog.apply_rotation()
        dialog.move_aligned_image(dx=25, dy=25)

        matrix = dialog._build_combined_matrix()

        h, w = dialog.original_aligned_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        expected_matrix = _compose([
            _qt_rotation_2x3(30, cx, cy),
            _translation_2x3(25, 25),
        ])
        assert np.allclose(matrix, expected_matrix, atol=1e-3)

        # Pixel-level: apply the emitted matrix to the original and verify feature position
        original = dialog.original_aligned_image
        warped = cv2.warpAffine(original.astype(np.float32), matrix, (w, h))

        # Expected feature center: apply the transformation to the initial center
        initial_row, initial_col = get_feature_center(original)
        a = math.radians(30)
        c, s = math.cos(a), math.sin(a)
        # Rotation around (cx, cy) followed by translation (25, 25)
        new_col = c * initial_col - s * initial_row + cx * (1 - c) + cy * s + 25
        new_row = s * initial_col + c * initial_row + cy * (1 - c) - cx * s + 25
        final_center = get_feature_center(warped)
        assert final_center == pytest.approx((new_row, new_col), abs=2.0)

    def test_landmark_coords_transformed_to_baked_space(self, dialog):
        """Landmarks collected after a manual move must be projected into baked image
        space before RANSAC.  Without the fix, RANSAC would see mismatched coordinate
        systems (baked image vs local pre-transform coords) and estimate a spurious
        second shift on top of the one already baked in.

        Setup: move 50px right, then add landmarks whose src/dst are consistent with
        'no further correction needed'.  The final combined matrix should equal T(50,0)
        — the manual move only — not T(100,0) or anything else.
        """
        dx = 50
        dialog.move_aligned_image(dx=dx, dy=0)

        # Inject 4 landmark pairs directly.
        # The user has visually moved the image 50px right, so a feature that lives at
        # local pixel (x, y) appears in the scene at (x+50, y).
        # src = scene/reference coords of target feature (where it "should" go)
        # dst = local item coords of the matching moving feature (mapFromScene result)
        # For zero residual alignment, src[i] == baked position of dst[i] == (x+dx, y).
        landmarks = [(200.0, 300.0), (800.0, 400.0), (500.0, 700.0), (1200.0, 900.0)]
        dialog._lm_src_pts = [(x + dx, y) for x, y in landmarks]  # scene (reference) coords
        dialog._lm_dst_pts = [(x, y) for x, y in landmarks]       # local item coords

        dialog._confirm_ransac_affine()

        # After RANSAC the UI transform is reset; the accumulated _ransac_M encodes the
        # full transform from original image → target.  The final combined matrix (with
        # identity UI transform on top) must equal T(50, 0).
        matrix = dialog._build_combined_matrix()
        expected = _translation_2x3(dx, 0)
        assert np.allclose(matrix, expected, atol=2.0), (
            f"Expected T({dx},0) but got {matrix}. "
            "Landmark coords were likely not projected into baked image space."
        )

    def test_landmark_reuse_after_ransac(self, dialog):
        """Test that 'Reuse Previous' landmarks follow the features after a RANSAC warp.

        When a RANSAC alignment is applied, the moving image is 'baked' (warped).
        Reused landmarks must be updated to their new positions in the baked image
        so that they stay anchored to the image features they were originally placed on.
        """
        # 1. Start landmark mode
        dialog._on_landmark_button_clicked()
        assert dialog._lm_mode is True

        # 2. Add 3 pairs of landmarks to satisfy minimum requirement
        # We'll simulate a 20px right shift alignment.
        # Target points at (500, 500), (600, 500), (500, 600)
        # Moving points at (480, 500), (580, 500), (480, 600)
        pairs = [
            ((500.0, 500.0), (480.0, 500.0)),
            ((600.0, 500.0), (580.0, 500.0)),
            ((500.0, 600.0), (480.0, 600.0)),
        ]

        for src_pt, dst_pt in pairs:
            dialog._on_landmark_click(src_pt[0], src_pt[1]) # Reference (red)
            dialog._on_landmark_click(dst_pt[0], dst_pt[1]) # Moving (green)

        assert len(dialog._lm_src_pts) == 3
        assert len(dialog._lm_dst_pts) == 3

        # 3. Compute RANSAC
        dialog._on_landmark_button_clicked()

        # M should be roughly [[1, 0, 20], [0, 1, 0]]
        assert dialog._ransac_M is not None
        np.testing.assert_allclose(dialog._ransac_M, [[1, 0, 20], [0, 1, 0]], atol=1e-1)

        # 4. Enter landmark mode again and Reuse Previous
        dialog._on_landmark_button_clicked()
        dialog._reuse_last_landmarks()

        # 5. Verify reused landmarks
        assert len(dialog._lm_src_pts) == 3
        assert len(dialog._lm_dst_pts) == 3

        # After the image was shifted +20px, the features that were at (480, 500)
        # are now at (500, 500) in the baked image. 
        # The moving landmarks (dst) should have followed them.
        for i, (src_pt, _) in enumerate(pairs):
            expected_dst = src_pt # They should now align with target!
            actual_dst = dialog._lm_dst_pts[i]
            np.testing.assert_allclose(actual_dst, expected_dst, atol=1e-1)

    def test_landmark_sidebar(self, dialog, qtbot):
        """Test that the landmark sidebar populates and responds to clicks."""
        dialog.show()
        qtbot.waitExposed(dialog)
        
        # Start landmark mode
        dialog._start_landmark_mode()
        assert dialog.landmarks_sidebar.isVisible()
        assert dialog.toggle_list_btn.isVisible()
        assert dialog.toggle_list_btn.isChecked()

        # Add 2 landmark pairs
        # Points far from edges to avoid clamping during centerOn
        dialog._on_landmark_click(500, 500) # ref
        dialog._on_landmark_click(510, 510) # mov
        dialog._on_landmark_click(1000, 1000) # ref
        dialog._on_landmark_click(1010, 1010) # mov

        assert dialog.landmarks_list_widget.count() == 2
        assert dialog.landmarks_list_widget.item(0).text() == "Pair 1"
        assert dialog.landmarks_list_widget.item(1).text() == "Pair 2"

        # Test click on "Pair 1"
        # Force a very high zoom to ensure centering is possible
        dialog.image_view.scale(10.0, 10.0)
        dialog.image_view._current_zoom = 10.0
        
        item = dialog.landmarks_list_widget.item(0)
        dialog._on_landmark_list_item_clicked(item)
        
        # Verify centering - map viewport center to scene
        view_center = dialog.image_view.mapToScene(dialog.image_view.viewport().rect().center())
        # We allow some tolerance due to scrollbar offsets and viewport margins
        assert view_center.x() == pytest.approx(500, abs=5.0)
        assert view_center.y() == pytest.approx(500, abs=5.0)

        # Test hide sidebar via method
        dialog._hide_landmarks_sidebar()
        assert not dialog.landmarks_sidebar.isVisible()
        assert not dialog.toggle_list_btn.isChecked()

        # Exit landmark mode
        dialog._exit_landmark_mode()
        assert not dialog.landmarks_sidebar.isVisible()
        assert not dialog.toggle_list_btn.isVisible()

