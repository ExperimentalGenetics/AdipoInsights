import os
import pickle
import joblib
import hashlib
import numpy as np
from pathlib import Path
from sklearn.base import OutlierMixin
from sklearn.ensemble import IsolationForest

def get_shape_matrix(file: Path) -> np.array:
    shape = np.load(file, allow_pickle=True)['features'].item()
    data_matrix = []
    for name, features in shape.items():
        # print("name - ",name)
        if name not in ['centroid', 'cell_index', 'bbox']:
            data_matrix.append(features)
    data_matrix = np.concatenate(data_matrix, axis=1)
    return data_matrix.astype(np.float32)

def detect_outliers(file: Path, model_path: Path) -> np.ndarray:
    clf = joblib.load(model_path)
    data_matrix = get_shape_matrix(file)
    outliers_score = clf.score_samples(data_matrix)
    return outliers_score

def train_outlier_detector(shape_dir: Path, model_dir: Path) -> OutlierMixin:
    """
    Trains an outlier detector based on the shape information of training files (unsupervised).

    Args:
        shape_dir: Path to the shape information files (training data).
        model_dir: Path to the directory to story the model file.

    Returns:
        OutlierMixin: Trained outlier detector.
    """
    # We use the shape information from all files for training
    all_shapes = []
    for file in shape_dir.glob('*.npz'):
        all_shapes.append(get_shape_matrix(file))
    all_shapes = np.concatenate(all_shapes)
    clf = IsolationForest(random_state=0, n_estimators=200, n_jobs=-1).fit(all_shapes)
    joblib.dump(clf, model_dir / "WAT_IsolationForest.pkl")
    
    
def filter_outliers(segments: np.ndarray, outliers: np.ndarray, unique_segments: np.ndarray) -> np.ndarray:
    """Filter out the outlier segments from the segmentation."""
    filtered_segments = np.copy(segments)
    unique_segments_no_bg = unique_segments[unique_segments != 0]
    outlier_segments = unique_segments_no_bg[outliers]
    for segment in outlier_segments:
        filtered_segments[filtered_segments == segment] = 0 
    return filtered_segments
