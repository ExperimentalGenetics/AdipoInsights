# **AdipoInsights**

Is a command-line pipeline for adipocyte detection and analysis from `.ndpi` digital histological whole slide images of adipocyte depots. 

---

## 🚀 Features

* The `.ndpi` should have good quality and high resoution (20× magnification or above) for good detection of adipocytes
* Accepts a single `.ndpi` file, a directory of `.ndpi` slides, a CSV list of `.ndpi` paths, *or* a comma-separated list on the command line
* Generates TIFFs, tissue masks, cropped regions, and adipocyte detection outputs
* Configurable for subcutaneous white adipose tissue (`sWAT`) or perigonadal white adipose tissue (`pWAT`)
* Automatic filename sanitization: replaces any non-alphanumeric character with `_`, collapses repeats, and trims


## 🛠️ Requirements
1. Python>=3.9
2. [ndpisplit](ndpisplit) -> move file to ```/usr/local/bin```
3. TensorFlow / Keras (required by segmentation models)


## 📁 Repository Structure

```text
AdipoInsights/
├── Analyzer/
│   ├── configs/                     # JSON config files
│   ├── data_extraction/             # Scripts for data extraction
│   ├── optimize/                    # Optimization scripts
│   ├── segmentation/                # Segmentation model code
│   ├── training/                    # Training scripts and notebooks
│   └── utils/                       # Utility functions and helpers
├── ndpisplit/                       # Helpers to split large .ndpi files (via NDPITools)
├── parent_dir/
│   └── models/                      # Pretrained models (download via Zenodo)
├── requirements.txt                 # Python dependencies
├── script_notebooks/
│   ├── run_adipocytes_pipeline.sh   # Bash wrapper to launch the pipeline
│   └── testing_WAT_inf.py           # Main Python inference script
```

## ⚙️ Configuration

1. **Model directory**: Place your pretrained models under [models](parent_dir/models):

   * `adipocytes_IsolationForest.pkl`
   * `skin_tissue_segmentation.ckpt`
   * `wat_tissue_segmentation.ckpt`

**Model weights download:** You can download the pretrained model weights from Zenodo: [https://doi.org/10.5281/zenodo.15321697](https://doi.org/10.5281/zenodo.15321697)

2. **Config files**: Update thresholds and parameters in `Analyzer/configs/*.json` if needed.

### 🔧 Config File Parameters

Each JSON config (e.g., `Analyzer/configs/subcutaneous_wat.json`) defines key parameters that you can tweak directly:

* **`tissue_type`**: Type of WAT to process. Options: `"sWAT"` or `"pWAT"`.
* **`shape_ratio_thresh`**: Maximum ratio of minor/major axis for cell shape filtering (lower values enforce rounder cells).
* **`max_20xsize`**: Maximum allowed pixel area at 20× magnification; cells larger than this are ignored.
* **`device`**: Compute device for models. Use `"cpu"` or e.g. `"gpu"` if you have a GPU, or `"mps"` for macOS.
* **`num_tissues`**: Number of separate tissue regions to segment per slide.
* **`edge_detector`**: Method for edge detection (`"canny","otsu"`, etc.).

* **`cell_segmentation_interval_binary`**: `[min, max]` pixel-area range for initial binary segmentation.

* **`cell_segmentation_interval_binary_inv`**: `[min, max]` range for inverse segmentation mask.

* **`min_adipocyte_intensity`**: Normalized intensity threshold (0–1) for removal of non-adipocyte structures.

* **`outlier_threshold`**: Isolation forest outlier cutoff for cell outlier removal.

You can edit these values directly in the JSON to customize processing without changing any code.

---

### Installation
#### Using miniconda

1. Clone github repository to local :
    ```python 
    git clone git@github.com:ExperimentalGenetics/AdipoInsights.git
    ```

2. Using conda create a virtual environment

    a. Install Miniconda (if required)

    ```python 
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    source miniconda3/etc/profile.d/conda.sh
    ```

    b. Create conda environment and install dependencies. 
    ```python 
    conda create --name wat -y python=3.9
    conda activate wat
    pip install -r requirements.txt
    ```


##  🏃‍♂️ Usage

```python 
cd /path/to/AdipoInsights
```

#### 1. Directory of slides

```bash
script_notebooks/run_adipocytes_pipeline.sh \
  /path/to/ndpi_folder \
  /path/to/output_dir \
  sWAT \
  /path/to/parent_dir
```

#### 2. CSV listing slides

Create a CSV (`slides.csv`) with one column of `.ndpi` paths:

```csv
ndpi_path
/data/A.ndpi
/data/B.ndpi
```

Run:

```bash
script_notebooks/run_adipocytes_pipeline.sh slides.csv /path/to/output sWAT /path/to/parent_dir
```

#### 3. Comma-separated list

```bash
script_notebooks/run_adipocytes_pipeline.sh "A.ndpi,B.ndpi,C.ndpi" /out sWAT /parent_dir
```

#### 4. Single file

```bash
script_notebooks/run_adipocytes_pipeline.sh /data/slideX.ndpi /out pWAT /parent_dir
```

## 📝 Output

For each slide, the pipeline creates a folder named after the sanitized filename. Inside you will find:

* `*_fixed.tif`, `*_resized.tif`  – intermediate images
* `*_wat.png`, `*_wat_post.png`   – tissue masks
* `*_x5_cropped.tif`, `*_x20_cropped.tif` – cropped ROIs
* `*_wat_cropped.png`            – final mask
* `*_shape_data.npz`              – raw adipocyte shape data
* `*_mean.csv`, `*_all.csv`       – aggregated and per-cell features
* `*_x20_wat.jpg`, `*_cells.jpg`  – visualizations


## 🐛 Troubleshooting

* **"skin region could not be identified"**: Try adjusting mask thresholds in your config JSON.
* **Missing dependencies**: Ensure `openslide-python` and all Python packages are installed.


## 📄 License

[MIT License](LICENSE)










