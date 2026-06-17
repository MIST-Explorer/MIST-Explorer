import heapq
import logging
import cv2
import numpy as np

from utils import to_uint8

logger = logging.getLogger(__name__)

def adjust_contrast(img, contrast_min=2, contrast_max=98):
    """Rescales the image values to 0-255 using percentiles."""
    minval = np.percentile(img, contrast_min)
    maxval = np.percentile(img, contrast_max)
    img = np.clip(img, minval, maxval)
    diff = maxval - minval
    if diff > 0:
        img = ((img - minval) / diff) * 255
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)

def find_points(image, min_circularity=0.5, top_k=500):
    """Finds contour centers/centroids filtered by minimum circularity, sorted by area."""
    image = to_uint8(image.copy())
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduce noise
    thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Use a min-heap to keep top_k largest area centers
    heap = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)
        if circularity >= min_circularity:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Push to heap as (area, (cx, cy))
                if len(heap) < top_k:
                    heapq.heappush(heap, (area, (cx, cy)))
                else:
                    heapq.heappushpop(heap, (area, (cx, cy)))
    # Extract centers from heap, sorted by area descending
    top_centers = [center for _, center in sorted(heap, reverse=True)]
    return np.array(top_centers)

def find_points_robust(img, top_k=500, min_circularity=0.5):
    """Try multiple feature detectors in order of preference (ORB and Custom)."""
    detectors = [
        ("ORB", cv2.ORB_create(nfeatures=top_k or 500)),
        (
            "CUSTOM",
            None
        ),
    ]
    result = {}
    for name, detector in detectors:
        if name == "CUSTOM":
            points = find_points(img, min_circularity=min_circularity, top_k=top_k)
            result[name] = points
            continue
        kp = detector.detect(img, None)
        if len(kp) >= 10:  # Minimum threshold
            points = np.array([p.pt for p in kp])
            result[name] = points
    return result

def try_optical_flow_alignment(
        source,
        target,
        moving_points,
        fixed_points):
    """Try optical flow with multiple pyramid levels."""
    if len(moving_points) < 4:
        return None, False

    # Parameters for different scales
    lk_params = [
        dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        ),
        dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        ),
        dict(
            winSize=(31, 31),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
        ),
    ]
    best_inliers = 0
    best_M = None
    moving_points_cv = moving_points.astype(np.float32).reshape(-1, 1, 2)
    fixed_points_cv = fixed_points.astype(np.float32).reshape(-1, 1, 2)
    if len(moving_points_cv) > len(fixed_points_cv):
        moving_points_cv = moving_points_cv[: len(fixed_points_cv)]
    else:
        fixed_points_cv = fixed_points_cv[: len(moving_points_cv)]
    for params in lk_params:
        try:
            nextPts, status, err = cv2.calcOpticalFlowPyrLK(
                source, target, moving_points_cv, fixed_points_cv, **params
            )
        except cv2.error:
            logger.warning(
                "Could not compute optical flow for this level: %s", params)
            continue
        good_indices = (status.flatten() == 1) & (
            err.flatten() < 50
        )  # Error threshold

        if np.sum(good_indices) >= 4:
            good_moving = moving_points_cv[good_indices][:, 0, :]
            good_next = nextPts[good_indices][:, 0, :]

            M, inliers = cv2.estimateAffine2D(
                good_moving,
                good_next,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=100000,
                confidence=0.99,
            )
            if M is not None and np.sum(inliers) > best_inliers:
                best_inliers = np.sum(inliers)
                best_M = M

    return best_M, best_inliers
