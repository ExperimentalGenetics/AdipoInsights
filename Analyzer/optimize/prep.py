import imageio
import re,json, os
import numpy as np
from tqdm import tqdm 
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2, openslide
import tifffile as tiff
from PIL import Image, UnidentifiedImageError
from skimage import io, transform
from pathlib import Path
from Analyzer.optimize.ndpi import get_meta_info_ndpi, convert_ndpi_to_tiff
import torch 
Image.MAX_IMAGE_PIXELS = 20000000000

def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_fields(filepath):
    _os = openslide.OpenSlide(filepath)
    return _os.properties['hamamatsu.SourceLens']

def resize_with_imageio(image_path, output_path, img_size):
    try:
        img = io.imread(image_path)
        resized_img = cv2.resize(img, (img_size, img_size))
        io.imsave(output_path, resized_img)
        return True
    except Exception as e:
        print(f"Error resizing image with imageio: {e}")
        return False

def resize_with_opencv(image_path, output_path, img_size):
    try:
        img = cv2.imread(str(image_path))
        resized_img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(output_path), resized_img)
        return True
    except Exception as e:
        print(f"Error resizing image with OpenCV: {e}")
        return False

def generate_tiffs_test(ndpi_file: Path, fixed_file: Path, resized_file: Path, inference_dir:Path, config: dict) -> dict:
    """
    Reads the original ndpi file, applies the double-sided fix and converts the image to tif files.

    Args:
        ndpi_file: Path to the original ndpi file.
        fixed_file: Path used to store the fixed tif file.
        resized_file: Path used to store the resized version of the file.
        config: Configuration parameters (using the target image size for the first segmentation model).

    Returns:
        Dictionary with the used image side (left or none) and the shape of the x20 file.
    """

    file_basename = os.path.basename(str(resized_file))
    path_components = inference_dir.parts
    img_size = config['s1']['preprocessing']['img_size']
    source_lens = int(get_fields(str(ndpi_file)))
    ndpi_info = get_meta_info_ndpi(ndpi_file)

    if ndpi_info['shape_ratio'] > config['shape_ratio_thresh']:
        # Just take the left image
        crop_region = '0,0,0.5,1'
        image_side = 'left'
    else:
        # Single-sided image
        crop_region = '0,0,1,1'
        image_side = 'none'

    try:
        convert_ndpi_to_tiff(ndpi_file, fixed_file, crop_region, 'x5')
    except Exception as e:
        source_mag = "x" + str(int(source_lens/4))
        convert_ndpi_to_tiff(ndpi_file, fixed_file, crop_region, source_mag)

    try:
        img = Image.open(fixed_file)
        img.load() 
        img = img.resize((img_size, img_size))
        img.save(str(resized_file))

    except (Exception, UnidentifiedImageError) as e:
        if not resize_with_imageio(fixed_file, resized_file, img_size):
            if not resize_with_opencv(fixed_file, resized_file, img_size):
                print(f"Failed to resize image {fixed_file} with all methods")
                return {}

    slidedata = {
        'image_name': str(ndpi_file),
        'filename':file_basename,
        'image_side': image_side,
        'x20_width': ndpi_info['width'] * 4,
        'x20_height': ndpi_info['height'] * 4,
        'SourceLens': source_lens
    }


    json_path = os.path.join("/".join(path_components[:-2]), "ndpi_fileinfo.json")

    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            existing_data = json.load(json_file)
        if not isinstance(existing_data, list):
            existing_data = [existing_data]

        # Check for duplicates
        if not any(entry['filename'] == file_basename for entry in existing_data):
            existing_data.append(slidedata)
            # Save updated data back to the JSON file
            with open(json_path, 'w') as json_file:
                json.dump(existing_data, json_file)
    else:
        # If file doesn't exist, create a new one with the new data
        with open(json_path, 'w') as json_file:
            json.dump([slidedata], json_file)

    return {
            'image_side': image_side,
            'x20_width': ndpi_info['width'] * 4,
            'x20_height': ndpi_info['height'] * 4
        }

def generate_20x(ndpi_file: Path, fixed_file: Path, config: dict) -> dict:
    """
    Reads the original ndpi file, applies the double-sided fix and converts the image to tif files.

    Args:
        ndpi_file: Path to the original ndpi file.
        fixed_file: Path used to store the fixed tif file.
        resized_file: Path used to store the resized version of the file.
        config: Configuration parameters (using the target image size for the first segmentation model).

    Returns:
        Dictionary with the used image side (left or none) and the shape of the x20 file.
    """
    ndpi_info = get_meta_info_ndpi(ndpi_file)

    if ndpi_info['shape_ratio'] > config['shape_ratio_thresh']:
        # Just take the left image
        crop_region = '0,0,0.5,1'
        image_side = 'left'
    else:
        # Single-sided image
        crop_region = '0,0,1,1'
        image_side = 'none'

    convert_ndpi_to_tiff(ndpi_file, fixed_file, crop_region, 'x20')
    

    return {
        'image_side': image_side,
        'x20_width': ndpi_info['width'] * 4,
        'x20_height': ndpi_info['height'] * 4
    }