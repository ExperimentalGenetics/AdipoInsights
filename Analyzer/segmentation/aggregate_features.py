from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


def aggregate_features(shape_features: Dict[str, np.ndarray], outliers_score: np.ndarray, mask_20x: np.ndarray,
                       aggregated_features_path: Path, all_features_path: Path, config: dict) -> None:
    """
    Filters outliers in the data and aggregates the shape information across all cells into a single number (e.g. mean cell area).

    Args:
        shape_features: Calculated shape features.
        outliers_score: Outlier score for each cell.
        aggregated_features_path: Path to the csv file where aggregated features (mean over all cells) will be stored.
        all_features_path: Path to the csv file where all features (for each cell) will be stored.
        config: Configuration parameters (using outlier_threshold which determines how many cells will be considered as outlier, larger values closer to zero remove more cells).
    """
    shape_data_mean = {}
    shape_data_all = {}
    for key in shape_features.keys():
        # print("key -", key)
        if key != 'centroid' and key != 'bbox-0' and key != 'bbox-1' and key != 'bbox-2' and key != 'bbox-3':
            if 'outlier_threshold' in config:
                shape_features[key] = shape_features[key][outliers_score > config['outlier_threshold'], :]
                outliers = outliers_score[outliers_score > config['outlier_threshold']]
                shape_data_all['outlier_score'] = outliers

            if shape_features[key].shape[1] > 1:
                for i in range(shape_features[key].shape[1]):
                    shape_data_mean[f'{key}_{i}'] = np.mean(shape_features[key][:, i])
                    shape_data_all[f'{key}_{i}'] = shape_features[key][:, i]
            else:
                shape_data_mean[key] = np.mean(shape_features[key])
                shape_data_all[key] = shape_features[key].squeeze()  # should result in 1D array


    # Calculate cell density
    nr_cells = len(shape_features['area'])
    total_cell_area = shape_features['area'].sum()
    white_pixel_count = np.sum(mask_20x != 0)
    shape_data_mean["TotalWATArea"] = white_pixel_count
    shape_data_mean["TotalAdipocyteArea"] = total_cell_area
    shape_data_mean["TotalAdipocyteCount"] = nr_cells
    shape_data_mean["cell_density"] = (nr_cells / total_cell_area) * 1e4  # nr of cells per 10000 pixels

    if nr_cells > 0:
        df = pd.DataFrame({k: [v] for k, v in shape_data_mean.items()})
        df.to_csv(aggregated_features_path, index=None)

        df = pd.DataFrame(shape_data_all)
        df.to_csv(all_features_path, index=None)
        return total_cell_area ### new line added
    else:
        raise ValueError('Could not detect any cells in the image. No features are calculated')