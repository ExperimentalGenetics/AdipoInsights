import os,pickle
import numpy as np
from typing import Dict
from pathlib import Path
from p_tqdm import p_map
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from skimage.measure import regionprops, regionprops_table
from skimage import io,color,measure
from Analyzer.segmentation import segment_cells

from skimage.color import rgb2hsv
from scipy.stats import entropy

def shape_features(segments: np.ndarray, img: np.ndarray, shape_data_path: Path) -> Dict[str, np.ndarray]:
    
    n_regions = max(np.unique(segments))
    features = {}
    feature_spec = [('area', 1), ('perimeter',1), ('eccentricity', 1), ('solidity', 1), #('extent',1),
                    ('centroid', 2), ('moments_hu', 7), ('bbox',4),('euler_number',1),('feret_diameter_max',1)]
    for name, shape in feature_spec:
        features[name] = np.zeros((n_regions, shape))
    for region in regionprops(segments, intensity_image=img):
        for name, _ in feature_spec:
            try:
                features[name][region.label - 1, :] = region[name]
            except ValueError as e:
                print("Error assigning feature:", name)
                print("region[name]:", region[name])
                print("Expected shape:", features[name].shape)
                print("Region label:", region.label)
                raise e

    X = features['centroid']
    k = 5
    nbrs = NearestNeighbors(n_neighbors=min(k, X.shape[0]), algorithm='ball_tree').fit(X)
    distances, _ = nbrs.kneighbors(X)
    features['mean_dist'] = np.mean(distances, axis=1).reshape(-1, 1)
    features['cell_index'] = np.arange(n_regions).reshape(-1, 1)
    cells = segments - 1  # background=-1, first_cell=0
    np.savez_compressed(shape_data_path, features=features, cells=cells)
    return features

def normalize_staining(img):

    Io = 240 
    alpha = 1 
    beta = 0.15
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
    maxCRef = np.array([1.9705, 1.0308])
    h, w, c = img.shape
    img = img.reshape((-1,3))
    OD = -np.log10((img.astype(np.float64)+1)/Io) 
    ODhat = OD[~np.any(OD < beta, axis=1)] 
    if ODhat.size == 0:
        raise ValueError("ODhat is empty. No data for SVD calculation.")
    
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T)) 
    if eigvals.min() < 0:
        raise ValueError("Invalid eigenvalues. SVD did not converge.")
    
    That = ODhat.dot(eigvecs[:,1:3]) 
    phi = np.arctan2(That[:,1],That[:,0]) 
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    if vMin[0] > vMax[0]:    
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    Y = np.reshape(OD, (-1, 3)).T
    
    if Y.shape[1] != HE.shape[0]:
        raise ValueError("Number of samples in Y does not match the number of rows in HE.")
    
    try:
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    except np.linalg.LinAlgError as e:
        print("SVD did not converge in Linear Least Squares")
        print("HE shape:", HE.shape)
        print("Y shape:", Y.shape)
        raise e
    
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2))) # Step 8: Convert extreme values back to OD space
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    return Inorm

def calculate_shape_features_train(
        dataX20cropped_dir: Path,
        datamask_dir: Path,
        shape_dir: Path,
        config: dict) -> None:
    """
    Calculate shape features for a set of training files (includes cell segmentation).

    Args:
        dataX20cropped_dir: directory of the cropped x20 images.
        datamask_dir: directory of the masks
        shape_dir: output directory where the shape files will be stored.
        config: Configuration parameters for the cell segmentation.
    """
    def shape_features_file(cropped_file: Path, mask_file: Path) -> None:
        img_20x = io.imread(str(cropped_file))
        mask_20x = io.imread(str(mask_file))
        
        if mask_20x is not None:
            if img_20x[:2].shape != mask_20x.shape[:2]:  # Compare only the first two dimensions
                mask_20x = Image.fromarray(mask_20x)
                mask_20x = mask_20x.resize((img_20x.shape[1], img_20x.shape[0]))
                mask_20x = np.array(mask_20x)
            img_20x[mask_20x[:, :, 0] == 0] = 0
            
        img_20x_gray = color.rgb2gray(img_20x)

        segments = segment_cells(img_20x_gray, config)
        segments_values = np.unique(segments)
        if len(segments_values) == 1 and segments_values[0] == 0:
            print(f'The file {cropped_file} does not contain a WAT region. No shape features are calculates for this file.')
        else:
            shape_features_path = shape_dir / f'{cropped_file.stem}.npz'
            shape_features(segments, img_20x, shape_features_path, config)
    
    cropped_files = list(dataX20cropped_dir.glob('*.tif'))
    mask_files = list(datamask_dir.glob('*.png'))
    p_map(shape_features_file, cropped_files, mask_files, num_cpus=2)
