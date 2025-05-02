import openslide
import numpy as np
from skimage import io
from pathlib import Path
from Analyzer.optimize.img_ops import load_mask, get_bounding_box_from_mask
from Analyzer.optimize.ndpi import convert_ndpi_to_tiff
from typing import Union

def center_crop(img: np.ndarray, new_size: Union[int, None] = None) -> np.ndarray:
    """
    Center-crop an image to a square of size new_size (or the minimum dimension if new_size is None).
    """
    h, w = img.shape[:2]
    new_size = min(new_size or min(h, w), h, w)
    start_x = (w - new_size) // 2
    start_y = (h - new_size) // 2
    if img.ndim == 2:
        return img[start_y:start_y+new_size, start_x:start_x+new_size]
    else:
        return img[start_y:start_y+new_size, start_x:start_x+new_size, ...]

def crop_skin_test(file: str, seg_path: str, output_dir: str) -> None:
    """
    Crop skin on x5 test image.

    Args:
        file: Path to x5 image file.
        seg_path: Path to the segmentation data from the s1 network.
        output_dir: Path to the directory where the cropped image will be saved.
    """
    img = io.imread(file)
    top, bottom, left, right = np.load(seg_path)['bbox']

    img = img.crop((left, top, right, bottom))
    img.save(output_dir / f'{file.stem}_cropped{file.suffix}')

def get_fields(filepath):
    _os = openslide.OpenSlide(filepath)
    return _os.properties['hamamatsu.SourceLens']

def crop_wat_test(
        ndpi_file: Path,
        skin_mask_post_file: Path,
        cropped_file_x5: Path,
        cropped_file_x20: Path,
        image_side: str) -> None:
    """
    Crop test image to specified bounding box

    Args:
        ndpi_file: path to the original ndpi file
        skin_mask_post_file: path to postprocessed s1 segmentation
        cropped_file_x5: target path to the cropped x5 file
        cropped_file_x20: target path to the cropped x20 file
        image_side: side in the original image file from which the mask was extracted.
    """
    seg = load_mask(skin_mask_post_file) 
    top, bottom, left, right = get_bounding_box_from_mask(seg)

    # Normalize coordinates to [0,1]
    x1 = left / seg.shape[1]
    y1 = top / seg.shape[0]
    width = (right / seg.shape[1]) - x1
    height = (bottom / seg.shape[0]) - y1
    
    if image_side == 'left':
        crop_region = f'{x1 / 2},{y1},{width / 2},{height}'
    elif image_side == 'none':
        crop_region = f'{x1},{y1},{width},{height}'
    else:
        raise ValueError(f'Invalid image side: {image_side}')

    try:
        for scale, path in zip(['x5', 'x20'], [cropped_file_x5, cropped_file_x20]):
            convert_ndpi_to_tiff(ndpi_file, path, crop_region, scale)
    
    except Exception as e:
        source_lens = int(get_fields(str(ndpi_file)))
        source_mag1 = "x" + str(int(source_lens/4)) ## 10x
        source_mag2 = "x" + str(source_lens) ##40x
        for scale, path in zip([source_mag1, source_mag2], [cropped_file_x5, cropped_file_x20]):
            convert_ndpi_to_tiff(ndpi_file, path, crop_region, scale)


