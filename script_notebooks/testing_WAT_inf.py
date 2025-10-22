import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
import re
import sys
import glob
import warnings
import argparse

import pandas as pd

from pathlib import Path
from PIL import Image, ImageFile

from Analyzer.utils import (Pipeline, unify_path, calculate_wat_mask, mask_processing,
                            calculate_wat_segmentation_sWAT, calculate_pwat_mask,
                            calculate_wat_segmentation_pWAT)
from Analyzer.optimize import (generate_tiffs_test, crop_wat_test)
from Analyzer.segmentation import adipocyte_detection

warnings.filterwarnings("ignore", category=DeprecationWarning)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main(ndpi_input_arg: str, inf_dir: str, tissue_type: str, parent_dir_arg: str) -> None:
    """
    Main entry point of the pipeline.

    Args:
        ndpi_input_arg (str): Path to a .ndpi file, a directory of .ndpi files, or a CSV listing .ndpi file paths.
        inf_dir (str): Inference directory to store output.
        tissue_type (str): Tissue type ("sWAT" or "pWAT") that determines the configuration.
        parent_dir_arg (str): Parent directory that contains model files and other resources.
    """
    PARENT_DIR = os.path.abspath(parent_dir_arg)
    parent_dir_path = unify_path(PARENT_DIR)
    model_path_outlier = parent_dir_path / 'models' / 'adipocytes_IsolationForest.pkl'
    model_seg_s1 = parent_dir_path / 'models' / 'skin_tissue_segmentation.ckpt'
    model_seg_s2 = parent_dir_path / 'models' / 'wat_tissue_segmentation.ckpt'

    inf_dir_path = Path(inf_dir)

    if ',' in ndpi_input_arg:
        ndpi_files = [p.strip() for p in ndpi_input_arg.split(',') if p.strip()]
    elif os.path.isfile(ndpi_input_arg) and ndpi_input_arg.lower().endswith('.csv'):
        df = pd.read_csv(ndpi_input_arg)
        ndpi_files = df.iloc[:, 0].astype(str).tolist()
    elif os.path.isdir(ndpi_input_arg):
        ndpi_files = glob.glob(os.path.join(ndpi_input_arg, '*.ndpi'))
    else:
        ndpi_files = [ndpi_input_arg]
        
    print(f"Found {len(ndpi_files)} .ndpi file(s) to process.")

    if tissue_type == "sWAT":
        config_path = Path(os.path.abspath("Analyzer/configs/subcutaneous_wat.json"))
    elif tissue_type == "pWAT":
        config_path = Path(os.path.abspath("Analyzer/configs/perigonadal_wat.json"))
    else:
        raise ValueError("Unsupported tissue type.")

    # Process each ndpi file
    exit_code = 0
    for ndpi_file in ndpi_files:
        if ndpi_file.lower().endswith(".ndpi"):
            print("ndpi_file - ", ndpi_file)
            img_path = unify_path(ndpi_file)
            
            stem = Path(ndpi_file).stem
            unique_name = re.sub(r'[^A-Za-z0-9]', '_', stem)
            unique_name = re.sub(r'_+', '_', unique_name)
            unique_name = unique_name.strip('_')
            folder_name = unique_name
            
            inference_dir = inf_dir_path / folder_name
            inference_dir.mkdir(parents=True, exist_ok=True)

            pipeline = Pipeline(config_path, clear_existing=False)

            pipeline.add_step(generate_tiffs_test,
                inputs={
                    'ndpi_file': Path(ndpi_file),
                    'inference_dir': inference_dir
                },
                outputs={
                    'fixed_file': inference_dir / f'{folder_name}_fixed.tif',
                    'resized_file': inference_dir / f'{folder_name}_resized.tif'
                },
                always_compute=False
            )

            if tissue_type == "sWAT":
                pipeline.add_step(calculate_wat_mask,
                    inputs={
                        'model_path': model_seg_s1,  # For subcutaneous WAT
                        'resized_file': inference_dir / f'{folder_name}_resized.tif'
                    },
                    outputs={
                        'skin_mask_file': inference_dir / f'{folder_name}_wat.png'
                    },
                    always_compute=False
                )
                
                pipeline.add_step(mask_processing,
                    inputs={'skin_mask_file': inference_dir / f'{folder_name}_wat.png'},
                    outputs={'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png'},
                    always_compute=False
                )
                
                pipeline.add_step(crop_wat_test,
                    inputs={
                        'ndpi_file': Path(ndpi_file),
                        'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png'
                    },
                    outputs={
                        'cropped_file_x5': inference_dir / f'{folder_name}_x5_cropped.tif',
                        'cropped_file_x20': inference_dir / f'{folder_name}_x20_cropped.tif'
                    },
                    always_compute=False
                )
                
                pipeline.add_step(calculate_wat_segmentation_sWAT,
                    inputs={
                        'model_path': model_seg_s2,
                        'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png',
                        'cropped_file_x5': inference_dir / f'{folder_name}_x5_cropped.tif'
                    },
                    outputs={
                        'wat_mask_file': inference_dir / f'{folder_name}_wat_cropped.png'
                    },
                    always_compute=False
                )
            
            else:
                # Steps for tissue_type "pWAT"
                pipeline.add_step(calculate_pwat_mask,
                    inputs={
                        'model_path': model_seg_s2,  # For perigonadal WAT
                        'resized_file': inference_dir / f'{folder_name}_resized.tif'
                    },
                    outputs={
                        'skin_mask_file': inference_dir / f'{folder_name}_wat.png'
                    },
                    always_compute=False
                )
                
                pipeline.add_step(mask_processing,
                    inputs={'skin_mask_file': inference_dir / f'{folder_name}_wat.png'},
                    outputs={'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png'},
                    always_compute=False
                )
                
                pipeline.add_step(crop_wat_test,
                    inputs={
                        'ndpi_file': Path(ndpi_file),
                        'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png'
                    },
                    outputs={
                        'cropped_file_x5': inference_dir / f'{folder_name}_x5_cropped.tif',
                        'cropped_file_x20': inference_dir / f'{folder_name}_x20_cropped.tif'
                    },
                    always_compute=False
                )
                
                pipeline.add_step(calculate_wat_segmentation_pWAT,
                    inputs={
                        'model_path': model_seg_s2,
                        'skin_mask_post_file': inference_dir / f'{folder_name}_wat_post.png',
                        'cropped_file_x5': inference_dir / f'{folder_name}_x5_cropped.tif'
                    },
                    outputs={
                        'wat_mask_file': inference_dir / f'{folder_name}_wat_cropped.png'
                    },
                    always_compute=False
                )
                
            pipeline.add_step(adipocyte_detection,
                inputs={
                    'cropped_file_x20': inference_dir / f'{folder_name}_x20_cropped.tif',
                    'model_path': model_path_outlier,
                    'wat_mask_file': inference_dir / f'{folder_name}_wat_cropped.png'
                },
                outputs={
                    'shape_data_path': inference_dir / f'{folder_name}_shape_data.npz',
                    'aggregated_features_path': inference_dir / f'{folder_name}_mean.csv',
                    'all_features_path': inference_dir / f'{folder_name}_all.csv',
                    'wat_region_x20': inference_dir / f'{folder_name}_x20_wat.jpg',
                    'cells_path': inference_dir / f'{folder_name}_cells.jpg'
                },
                always_compute=False
            )
            
            try:
                pipeline.run()
                exit_code = 0
            except ValueError as e:
                print(
                    f'The skin region could not be identified in the file. '
                    f'The file {img_path} could not be processed successfully',
                    file=sys.stderr)
                print(str(e), file=sys.stderr)
                exit_code = 1
            print("***************************************")
            print(f"Processing completed for folder: {folder_name}")
            print("***************************************")
    sys.exit(exit_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adipocyte detection pipeline using microscopy images."
    )
    parser.add_argument("--ndpi_input",
        type=str,
        required=True,
        help=(
            "Path to a single .ndpi file, a directory of .ndpi files, "
            "a CSV listing .ndpi file paths, or a comma-separated list of .ndpi paths."
        )
    )
    parser.add_argument("--inf_dir",
        type=str,
        required=True,
        help="Inference directory to store output files."
    )
    parser.add_argument("--tissue_type",
        type=str,
        required=True,
        help="Tissue type (e.g., 'sWAT' or 'pWAT') that determines the configuration."
    )
    parser.add_argument("--parent_dir",
        type=str,
        required=True,
        help="Parent directory containing model files and other resources."
    )
    args = parser.parse_args()
    main(args.ndpi_input, args.inf_dir, args.tissue_type, args.parent_dir)