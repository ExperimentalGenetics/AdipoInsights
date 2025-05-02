import cv2
import numpy as np
from typing import Dict
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from scipy.ndimage import binary_dilation
from skimage.feature import canny
from PIL import Image, ImageFile
from pathlib import Path
from skimage import io, color
from Analyzer.optimize import (get_bounding_box_from_mask, load_mask, resize_mask,preprocess_image)
from Analyzer.segmentation.aggregate_features import aggregate_features
from Analyzer.segmentation.outlier_detection import (detect_outliers,filter_outliers)

def remove_small_blobs(img, interval=[10,30], debug=True):
    mask, number_of_blobs = ndimage.label(img)
    if debug:
        print('Number of blobs before: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  
    if len(counts) <= 1:
        return img
    if debug:
        print(counts)
    remove = np.where((counts <= interval[0]) | (counts > interval[1]), True, False)
    remove_idx = np.nonzero(remove)[0]
    if debug:
        print("remove_idx:")
        print(remove_idx)
        print("len remove_idx: {}".format(len(remove_idx)))
    mask[np.isin(mask, remove_idx)] = 0
    mask[mask > 0] = 1  # set everything else to 1
    if debug:
        mask_after, number_of_blobs_after = ndimage.label(mask)
        print('Number of blobs after: ' + str(number_of_blobs_after))
    return mask

def segment_cells_canny(img_gray:np.ndarray,config: dict) -> np.ndarray:
    binary = canny(img_gray)
    binary = binary.astype(np.uint8)  # 0: black, 1: white
    binary = binary_dilation(binary, iterations=2).astype(np.uint8)
    binary = remove_small_blobs(binary, interval=config['cell_segmentation_interval_binary'], debug=False)
    binary_inv = 1 - binary
    binary_inv_clean = remove_small_blobs(
        binary_inv,
        interval=config['cell_segmentation_interval_binary_inv'],
        debug=False)
    instances, _ = ndimage.label(binary_inv_clean, np.ones([3] * 2))
    return instances 

def segment_cells_otsu(img_gray:np.ndarray,config: dict) -> np.ndarray:
    binary = img_gray >= threshold_multiotsu(img_gray, classes=2,)
    binary = 1 - binary
    binary = binary.astype(np.uint8)
    binary_dia = binary_dilation(binary, iterations=1).astype(np.uint8)
    binary_dia = remove_small_blobs(binary_dia, interval=config['cell_segmentation_interval_binary'], debug=False)
    binary_inv = 1 - binary_dia
    binary_inv_clean = remove_small_blobs(
        binary_inv,
        interval=config['cell_segmentation_interval_binary_inv'],
        debug=False)
    binary_inv_clean = ndimage.binary_fill_holes(binary_inv_clean)
    instances, _ = ndimage.label(binary_inv_clean, np.ones([3] * 2))
    return instances

def segment_cells(img_gray: np.ndarray, config: dict) -> np.ndarray:
    """Segment cells using the specified edge detection method."""
    if config['edge_detector'] == "canny":
        return segment_cells_canny(img_gray, config)
    elif config['edge_detector'] == "otsu":
        return segment_cells_otsu(img_gray, config)
    else:
        raise ValueError(f"Unknown edge detector: {config['edge_detector']}")
    
def process_cell(label: int, segments: np.ndarray) -> Dict:
    cell_mask = (segments == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return {"contours": contours, "centroid": (cX, cY), "label": label}

def adipocyte_detection(
    cropped_file_x20: Path,
    model_path: Path,
    wat_mask_file: Path,
    shape_data_path: Path,
    aggregated_features_path: Path,
    all_features_path: Path,
    wat_region_x20:Path,
    cells_path: Path,
    config: dict
    ) -> None:

    from Analyzer.segmentation.shape_features import shape_features
    img_orig = io.imread(str(cropped_file_x20))
    mask = load_mask(wat_mask_file)
    if mask is not None and img_orig[:, :, 0].shape != mask.shape:
        mask = resize_mask(mask, img_orig.shape[:2], wat_mask_file)
    img_resized = preprocess_image(img_orig, mask, config["max_20xsize"])
    top, bottom, left, right = get_bounding_box_from_mask(mask)
    img = img_resized[top:bottom, left:right]

    Image.fromarray(img).save(str(wat_region_x20), quality=85)
    img_gray = color.rgb2gray(img)
    segments = segment_cells(img_gray, config)

    # Check for valid segments
    unique_segments = np.unique(segments)
    if len(unique_segments) == 1 and unique_segments[0] == 0:
        raise ValueError(f"The file {cropped_file_x20} contains no valid WAT region. No shape features calculated.")

    features = shape_features(segments, img, shape_data_path)
    outliers_score = detect_outliers(shape_data_path, model_path)
    outliers = outliers_score < config.get('outlier_threshold', float('inf'))

    filtered_segments = filter_outliers(segments, outliers, unique_segments)
    
    if 'min_adipocyte_intensity' in config:
        threshold_intensity = config['min_adipocyte_intensity']
        flat_labels = filtered_segments.ravel()
        flat_intensities = img_gray.ravel()
        mask_nonzero = flat_labels > 0
        # Sum intensities and counts per label
        sums = np.bincount(flat_labels[mask_nonzero], weights=flat_intensities[mask_nonzero])
        counts = np.bincount(flat_labels[mask_nonzero])
        # Compute means safely, avoiding division by zero
        mean_intensities = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0
        )
        # Identify labels below threshold and zero them out
        low_labels = np.where(mean_intensities < threshold_intensity)[0]
        for lbl in low_labels:
            filtered_segments[filtered_segments == lbl] = 0

    img_overlay = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Create binary mask from filtered segments
    binary = (filtered_segments > 0).astype(np.uint8) * 255

    # Extract all contours in one pass
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Draw all contours at once
    cv2.drawContours(img_overlay, contours, -1, (0, 255, 255), thickness=2)

    # Compute connected components to get centroids
    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Draw centroid labels (skip background label 0)
    for i in range(1, num_labels):
        cX, cY = int(centroids[i][0]), int(centroids[i][1])
        cv2.putText(img_overlay, str(i), (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Convert back to RGB and save
    img_overlay_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB) 
    Image.fromarray(img_overlay_rgb).save(str(cells_path), quality=30)
    
    aggregate_features(features, outliers_score, mask,
                                            aggregated_features_path, all_features_path, config)
