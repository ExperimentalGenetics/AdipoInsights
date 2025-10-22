# **AdipoInsights**

Is a command-line pipeline for adipocyte detection and analysis from `.ndpi` digital histological images (DHIs) of adipocyte depots. It  performs automated segmentation, quantification, and analysis of adipocytes. It supports **subcutaneous (sWAT)**, **perigonadal (pWAT)**, and **inguinal (iWAT)** white adipose tissues, producing both **numerical data** and **visual summaries** for downstream analysis.


---

## 🚀 Features

### 🔹 Core Pipeline
* The `.ndpi` should have good quality and high resoution (20× magnification or above) for good detection of adipocytes
* Accepts a single `.ndpi` file, a directory of `.ndpi` slides, a CSV list of `.ndpi` paths, *or* a comma-separated list on the command line
* Generates TIFFs, tissue masks, cropped regions, and adipocyte detection outputs
* Configurable for subcutaneous white adipose tissue (`sWAT`) or perigonadal white adipose tissue (`pWAT`)
* Automatic filename sanitization: replaces any non-alphanumeric character with `_`, collapses repeats, and trims


### 🔹 Adipo Viewer (Optional GUI)
- Streamlit-based interface to **explore results interactively**
- Accepts either:
  - Folder path containing outputs, or
  - Uploaded `.csv` + image files
- Visualizes histograms, box/violin plots, scatter plots
- Built-in filters for `outlier_score` and `mean_dist`
- Statistical tests: Welch t-test, Mann–Whitney U, KS, ANOVA/LM
- Optional metadata integration (`case_id`, `Sex`, `Genotype`, or any columns)


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
├── adipo_viewer/                    # Streamlit-based interactive viewer
│   ├── app.py                       # Streamlit entrypoint
│   ├── engine.py                    # Core logic: discovery, plotting, stats
│   ├── ui.py                        # UI and controller
│   └── requirements.txt             # Minimal viewer dependencies
├── ndpisplit/                       # Helpers to split large .ndpi files (via NDPITools)
├── parent_dir/
│   └── models/                      # Pretrained models (download via Zenodo)
├── requirements.txt                 # Full pipeline dependencies
├── script_notebooks/
│   ├── run_adipocytes_pipeline.sh   # Bash wrapper for CLI pipeline
│   └── testing_WAT_inf.py           # Main Python inference script
└── README.md
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


## ⚙️ Installation
### Using miniconda

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

First go into the cloned directory :
```python 
cd /path/to/AdipoInsights
```

#### 1. Directory of multiple DHIs

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
script_notebooks/run_adipocytes_pipeline.sh /path/to/slides.csv /path/to/output sWAT /path/to/parent_dir
```

#### 3. Comma-separated list

```bash
script_notebooks/run_adipocytes_pipeline.sh "A.ndpi,B.ndpi,C.ndpi" /path/to/output_dir sWAT /parent_dir
```

#### 4. Single file

```bash
script_notebooks/run_adipocytes_pipeline.sh /data/slideX.ndpi /path/to/output_dir pWAT /parent_dir
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



## 🧭 Adipo Viewer — Interactive Visualization

Once your pipeline generates the *_all.csv and *_mean.csv files, you can explore results using the Adipo Viewer GUI.

###  ▶️ Run the Viewer

```bash
cd adipo_viewer
streamlit run app.py
```
This launches the app at http://localhost:8501.

### 💡 Usage
1.	Choose input mode:
    * Folder path containing your outputs, or
    * Upload *_all.csv, *_mean.csv, and/or images.
2.	Optionally upload metadata (only case_id required)
3.	Adjust filters for outlier_score and mean_dist.
4.	Explore the following tabs:
    * Analysis: Box, violin, histogram plots
    * Outlier: Inspect outlier_score distribution
    * Mean Distance: Analyze spatial thresholds
    * Stats: Run t-test, Mann–Whitney, KS, or ANOVA/LM

## 🧱 Example Visuals

Below are examples of the visualizations available through the **Adipo Viewer** interface:

| Visualization | Description |
|----------------|--------------|
| **📊 Box & Violin Plots** | Compare per-cell features such as *area*, *perimeter*, or *solidity* between groups or metadata categories. |
| **📈 Scatter Plots** | Explore relationships between quantitative variables (e.g., mean distance vs. cell area). |
| **📉 Histograms** | View the distribution of numerical metrics like *area*, *outlier_score*, or *mean_dist*. |
| **🧮 Statistical Tests** | Quickly run built-in comparisons including *t-test*, *Mann–Whitney U*, *Kolmogorov–Smirnov*, and *ANOVA/Linear Model*. |
| **🧬 Metadata-based Views** | Overlay visualizations by `case_id`, `Sex`, `Genotype`, or any other metadata column for grouped summaries. |

---

✅ **Tip:** Each visualization updates interactively as you modify filters or metadata selections in the Streamlit sidebar.

### 🧹 Cleanup

If you use the upload option, clear temporary staged files using “Clear uploaded temp files” in the sidebar.


## 🐛 Troubleshooting

Below are common issues you might encounter while running **AdipoInsights** or the **Adipo Viewer**, along with suggested fixes:

| Issue | Possible Fix |
|--------|---------------|
| **"Skin region could not be identified"** | Adjust mask thresholds in your configuration JSON file under `Analyzer/configs/`. |
| **Missing dependencies or import errors** | Ensure all packages are installed. Run `pip install -r requirements.txt`. For the viewer, install `streamlit`, `plotly`, and `statsmodels` if not already included. |
| **`openslide-python` errors** | Install both `openslide-python` and the `openslide-bin` system library. On macOS: `brew install openslide`. |
| **Streamlit fails to launch** | Verify that you are inside the `adipo_viewer` directory and the virtual environment is active. Then run `streamlit run app.py`. |
| **Plots not displaying or empty** | Check that your folder contains valid `*_all.csv` and `*_mean.csv` files with correct column headers. |
| **"Permission denied" or file access errors** | Ensure you have read/write permissions for the input/output directories. |
| **Viewer not updating after upload** | Use the sidebar option **“Clear uploaded temp files”** to reset cached data. |
| **Slow or unresponsive performance** | Limit the number of files in your inference directory or disable heavy plots temporarily. |


## 📄 License

[MIT License](LICENSE)










