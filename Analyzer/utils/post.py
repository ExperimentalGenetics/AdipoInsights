import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.morphology import erosion, disk
from skimage import measure, morphology, transform, io
from skimage.measure import label, regionprops

from Analyzer.optimize.img_ops import load_mask

def mask_processing(
        skin_mask_file: Path,
        skin_mask_post_file: Path,
        x20_width: int,
        x20_height: int, 
        config: dict
        ) -> None:
    """
    Applies postprocessing to segmentation from stage 1.

    Args:
        skin_mask_file (Path): Path to s1 segmentation file.
        skin_mask_post_file (Path): Path to the processed segmentation file.
        x20_width (int): Width of the original x20 file.
        x20_height (int): Height of the original x20 file.
        config (dict): Configuration dictionary containing processing parameters.
            Required keys:
            - tissue_type (str): tissue type (sWAT or pWAT)
            - num_tissues (int, optional): Number of tissues to process

    Returns:
        None: Saves the processed mask to skin_mask_post_file
    """
    seg = load_mask(skin_mask_file)
    closing_size = 5
    if config["tissue_type"] not in ["sWAT", "pWAT"]:
        raise ValueError(f"Unsupported tissue_type: {config['tissue_type']}")
    num_tissues = int(config.get("num_tissues"))
    if config["tissue_type"] == "sWAT":
        processed_seg = (
            postprocess_segmentation(
                    seg,
                    num_blobs=num_tissues,    
                    closing_size=closing_size)
                if num_tissues
                else only_oneblob_segmentation(seg, closing_size=closing_size)
        )
    else:   
        processed_seg = (
            postprocess_segmentation(seg, num_blobs=num_tissues, 
                                    closing_size=closing_size)
            if num_tissues
            else only_oneblob_segmentation(seg, closing_size=closing_size)
        )
        
    resized_seg = to_resolution(processed_seg, (x20_height / 4, x20_width / 4))
    Image.fromarray((resized_seg * 255).astype(np.uint8)).save(skin_mask_post_file, optimize=True)

def blobs_removal(segmentation, min_blob_size):
    """
    Removes connected components (blobs) smaller than min_blob_size from the segmentation.

    Args:
        segmentation: Binary segmentation mask.
        min_blob_size: Minimum size of connected components to keep.

    Returns:
        np.ndarray: Segmentation with small blobs removed.
    """
    labels, cc_sizes,_ = get_connected_comp(segmentation)
    small_blob_labels = np.where(cc_sizes < min_blob_size)[0]
    for label in small_blob_labels:
        segmentation[labels == label] = 0
    return segmentation

def only_oneblob_segmentation(seg: np.ndarray, closing_size: int = 2) -> np.ndarray:
    """
    Postprocess segmentation (closing, select largest connected component)

    Args:
        seg: segmentation
        closing_size: size of selem to use.

    Returns:
        np.ndarray: new segmentation
    """
    if len(np.unique(seg)) < 2:
        return seg
    seg = seg.astype(bool)
    if seg.ndim != 2:
        raise ValueError(f"Expected a 2D segmentation array, but got shape {seg.shape}")
    closing_size = closing_size if closing_size % 2 == 1 else closing_size + 1
    footprint = np.ones((closing_size,) * seg.ndim, dtype=bool)  
    seg = morphology.binary_closing(seg, footprint=footprint)
    all_components, counts, _ = get_connected_comp(seg)
    if len(counts) < 2:
        return np.zeros_like(seg, dtype=np.uint8)
    foreground = (all_components == np.argsort(counts)[-2]).astype(np.uint8)
    return foreground

def postprocess_segmentation(seg: np.ndarray, num_blobs: int = 4, closing_size: int = 2) -> np.ndarray:
    """
    Postprocess segmentation (closing, select largest connected component)

    Args:
        seg: segmentation
        closing_size: size of selem to use.

    Returns:
        np.ndarray: new segmentation
    """
    if len(np.unique(seg)) < 2:
        return seg
    seg = morphology.binary_closing(seg, footprint=np.ones((closing_size, closing_size))) 
    all_components, counts, sort_idx = get_connected_comp(seg)
    blob_indices = sort_idx[1:num_blobs+1] 
    foreground = np.isin(all_components, blob_indices)
    return foreground

def get_connected_comp(segmentation):
    """
    Args:
        segmentation: (dx, dy)-array where each pixel has a value in {1,..., #classes}

    Returns:
        all_labels: (dx, dy)-array where each pixel is labeled according to the connected component it belongs to
        cc_sizes: 1d-array with counts for each connected component. Background has idx 0.
    """
    labels = measure.label(segmentation)
    cc_sizes = np.array([(labels == i).sum() for i in range(len(np.unique(labels)))])
    sorted_idx = np.argsort(cc_sizes)[::-1]
    return labels, cc_sizes, sorted_idx

def to_resolution(segmentation: np.ndarray, size: tuple):
    """
    Rescale downsampled image to specified size.

    Args:
        segmentation: segmentation to resample
        size: target size

    Returns:
        np.ndarray: resized segmentation
    """
    width = size[1]
    height = size[0]
    resized_seg = transform.resize(segmentation, (height, width), order=0)
    assert len(np.unique(resized_seg)) <= 3, "Segmentation has more values than expected"
    resized_seg[~np.isclose(resized_seg, 0)] = 1
    return resized_seg.astype(np.uint8)