
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from skimage import io
from typing import Dict, Any
from PIL import Image

from Analyzer.training.seg import HistoSegx5
from Analyzer.optimize.img_ops import load_mask, get_bounding_box_from_mask, blobs_removal, get_connected_comp
from Analyzer.training.augment import get_transforms_test_s1, get_transforms_test_s2
from Analyzer.data_extraction.dataset import LazyDataset, LazyDatasetS2


def collate_fn_factory(transform_fn):
    """
    Create a DataLoader collate function that can handle:
    - zero-arg factory (transform_fn returns actual fn)
    - keyword-based transforms (transform_fn(**sample))
    - positional transforms (transform_fn(sample))
    """
    import inspect
    sig = inspect.signature(transform_fn)
    params = list(sig.parameters.values())
    if not params:
        mode = 'factory'
    elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        mode = 'kw'
    else:
        mode = 'pos'

    def collate_fn(batch):
        transformed = []
        for sample in batch:
            if mode == 'factory':
                tf = transform_fn()
                ts = tf(sample)
            elif mode == 'kw':
                ts = transform_fn(**sample)
            else:
                ts = transform_fn(sample)
            transformed.append(ts)
        data = torch.stack([s["data"] for s in transformed])
        names = [s.get("data_name", None) for s in transformed]
        return {"data": data, "data_name": names}

    return collate_fn


def calculate_wat_mask(model_path: Path, resized_file: Path, skin_mask_file: Path, config: Dict[str, Any]):
    """Stage 1: Generate binary WAT mask from resized image."""
    # Load model weights manually
    ckpt = torch.load(str(model_path), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model = HistoSegx5(config['s1']['model_kwargs'])
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.freeze()
    device = torch.device(config.get('device', 'cpu'))
    model.to(device)

    # Prepare DataLoader
    dataset = LazyDataset(data_path=resized_file.parent, load_seg=False, filestem=resized_file.stem)
    assert len(dataset) == 1, "Dataset must contain exactly one image."
    transform_fn = get_transforms_test_s1()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_factory(transform_fn), num_workers=0)
    batch = next(iter(loader))

    # Inference
    with torch.no_grad():
        logits = model(batch['data'].to(device))
    seg = logits.argmax(dim=1)[0].cpu().numpy()
    mask = (seg != 0).astype(np.uint8)

    # Save
    plt.imsave(str(skin_mask_file), mask, cmap='gray')


def calculate_wat_segmentation_sWAT(model_path: Path, skin_mask_post_file: Path,
                                    cropped_file_x5: Path, wat_mask_file: Path,
                                    config: Dict[str, Any]):
    """Stage 2: Refine WAT mask for sWAT region."""
    # Load model weights manually
    ckpt = torch.load(str(model_path), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    model = HistoSegx5(config['s2']['model_kwargs'])
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.freeze()
    device = torch.device(config.get('device', 'cpu'))
    model.to(device)

    # Prepare DataLoader
    dataset = LazyDatasetS2(data_path=cropped_file_x5.parent, load_seg=False, filestem=cropped_file_x5.stem)
    assert len(dataset) == 1, "Dataset must contain exactly one image."
    transform_fn = get_transforms_test_s2()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_factory(transform_fn), num_workers=0)
    batch = next(iter(loader))

    # Inference
    with torch.no_grad():
        logits = model(batch['data'].to(device))
    seg = logits.argmax(dim=1)[0].cpu().numpy()

    # Post-process: XOR skin region and epidermis
    skin = io.imread(str(skin_mask_post_file))
    if skin.ndim == 3:
        skin = skin[:, :, 0]
    skin = (skin > 0).astype(bool)
    
    skinseg = load_mask(skin_mask_post_file)
    top, bottom, left, right = get_bounding_box_from_mask(skinseg)
    cropped_skinwat = skinseg[top:bottom, left:right].astype(bool)
    
    epidermis = seg == 2
    epidermis = blobs_removal(epidermis, min_blob_size=3000)
    epidermis = cv2.dilate(epidermis.astype(np.uint8), np.ones((1, 1), np.uint8), iterations=10)
    epidermis = epidermis.astype(bool)
    
    if cropped_skinwat.shape != epidermis.shape:
        cropped_skinwat = Image.fromarray(cropped_skinwat)
        cropped_skinwat = cropped_skinwat.resize((epidermis.shape[1], epidermis.shape[0]))
        cropped_skinwat = np.array(cropped_skinwat)
        
    cropped_skinwat_result = np.logical_xor(cropped_skinwat, epidermis)
    cropped_skinwat_result = cv2.erode(cropped_skinwat_result.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=4)
    cropped_skinwat_result = blobs_removal(cropped_skinwat_result, min_blob_size=15000)
    all_components, counts, sort_idx = get_connected_comp(cropped_skinwat_result)
    num_blobs = 1
    blob_indices = sort_idx[1:num_blobs+1] 
    foreground = np.isin(all_components, blob_indices)
    plt.imsave(wat_mask_file, foreground, cmap="gray")


def calculate_pwat_mask(model_path: Path, resized_file: Path, skin_mask_file: Path, config: Dict[str, Any]):
    """Generate mask for pWAT (uses same routine as Stage 1)."""
    calculate_wat_mask(model_path, resized_file, skin_mask_file, config)


def calculate_wat_segmentation_pWAT(model_path: Path, skin_mask_post_file: Path,
                                    cropped_file_x5: Path, wat_mask_file: Path,
                                    config: Dict[str, Any]):
    """Finalize pWAT segmentation by cropping the mask region."""
    skin = io.imread(str(skin_mask_post_file))
    if skin.ndim == 3:
        skin = skin[:, :, 0]
    bbox = get_bounding_box_from_mask(load_mask(skin_mask_post_file))
    region = load_mask(skin_mask_post_file)[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(bool)
    plt.imsave(str(wat_mask_file), region.astype(np.uint8), cmap='gray')

