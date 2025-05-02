from pathlib import Path
import torch
import numpy as np
from PIL import Image
from skimage import io, measure
from typing import Tuple
from Analyzer.optimize.prep import get_device


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Loads a mask image and converts it to a binary mask.
    """
    mask = io.imread(str(mask_path))
    if mask.ndim > 2:
        mask = np.array(mask)[:, :, 0]
    mask[mask > 0] = 1
    return mask

def get_bounding_box_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Bounding box from mask

    Args:
        mask: mask for box

    Returns:
        Tuple[int, int, int, int]: extract bounding box (top, bottom, left, right)
    """
    # print("get bb mask - ",mask.shape)
    xx, yy = np.meshgrid(
        range(0, mask.shape[0]), range(0, mask.shape[1]), indexing='ij')
    top = min(xx[mask == 1])
    bottom = max(xx[mask == 1])
    left = min(yy[mask == 1])
    right = max(yy[mask == 1])

    return top, bottom, left, right

def get_connected_comp(segmentation: np.ndarray) -> tuple:
    """
    Returns connected components using skimage.measure.label.
    """
    labels = measure.label(segmentation.astype(np.uint8), connectivity=2)
    unique_labels = np.unique(labels)
    cc_sizes = np.array([(labels == i).sum() for i in unique_labels])
    sort_idx = np.argsort(cc_sizes)[::-1]
    return labels, cc_sizes, sort_idx

def blobs_removal(segmentation: np.ndarray, min_blob_size: int) -> np.ndarray:
    """
    Removes connected components smaller than min_blob_size.
    """
    labels, cc_sizes, _ = get_connected_comp(segmentation)
    small_blob_labels = np.where(cc_sizes < min_blob_size)[0]
    for label in small_blob_labels:
        segmentation[labels == label] = 0
    return segmentation

def mean_intensity_of_cell(img_gray: np.ndarray, label_map: np.ndarray, label: int) -> float:
    device = get_device()
    img_gray = img_gray.astype(np.float32)
    label_map = label_map.astype(np.int32)
    img_tensor = torch.tensor(img_gray, dtype=torch.float32, device=device)
    label_tensor = torch.tensor(label_map, device=device)
    mask = (label_tensor == label)
    if mask.sum() == 0:
        return 0.0
    return float(img_tensor[mask].mean().cpu().item())

def resize_mask(mask: np.ndarray, target_shape: Tuple[int, int], save_path: Path) -> np.ndarray:
    """Resize the mask to match the target shape and save it."""
    mask_resized = Image.fromarray(mask)
    mask_resized = mask_resized.resize(target_shape[::-1])  # Reverse for (width, height)
    mask_resized.save(save_path)
    return np.array(mask_resized)

def preprocess_image(img: np.ndarray, mask: np.ndarray, max_size: int) -> np.ndarray:
    from Analyzer.optimize.crop import center_crop
    """Apply mask to the image and crop if necessary."""
    if mask is not None:
        img[mask == 0] = 0  # Apply the mask to the image
    if img.shape[0] > max_size or img.shape[1] > max_size:
        img = center_crop(img, new_size=max_size)
    return img