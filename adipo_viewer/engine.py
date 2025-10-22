# engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable
import re, io, shutil, tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# Config & helpers
# =========================
@dataclass
class AppConfig:
    """
    Generic file discovery configuration.
    Assumes each 'case' generates <case_id>_all.csv / <case_id>_mean.csv and images nearby.
    """
    all_csv_glob: str = "*_all.csv"
    mean_csv_glob: str = "*_mean.csv"
    image_globs: Tuple[str, ...] = ("*_x20_wat.jpg","*_cells.jpg","*_wat_cropped.png","*_x20_cropped.tif")
    case_regex: str = r"^([A-Za-z0-9\-]+)[_\-].*$"  # first token before '_' or '-' -> case_id
    recursive: bool = True  # scan subfolders

def _read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try: return pd.read_csv(p)
    except Exception: return None

def _newest(paths: Iterable[Path]) -> Optional[Path]:
    paths = list(paths)
    if not paths: return None
    return max(paths, key=lambda p: p.stat().st_mtime)

def _case_from_name(name: str, cfg: AppConfig) -> Optional[str]:
    m = re.match(cfg.case_regex, name)
    if m: return str(m.group(1))
    # fallback: bare name without suffixes
    stem = Path(name).stem
    return stem.split("_")[0] if "_" in stem else stem

def _ensure_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str) if s.dtype != object else s

# =========================
# Domain: repository & metadata
# =========================
class InferenceRepository:
    """Discovers per-case artifacts (all/mean CSV and images) in a folder or in staged uploads."""
    def __init__(self, root: Path, cfg: AppConfig):
        self.root = Path(root)
        self.cfg = cfg
        if not self.root.exists():
            raise FileNotFoundError(f"Path does not exist: {self.root}")

    def _walk(self) -> List[Path]:
        if self.root.is_file(): return [self.root]
        if self.cfg.recursive: return list(self.root.rglob("*"))
        return list(self.root.glob("*"))

    def discover(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        Returns: { case_id: { 'all': [paths], 'mean': [paths], 'images': [paths], 'others':[paths] } }
        """
        bucket: Dict[str, Dict[str, List[Path]]] = {}
        files = [p for p in self._walk() if p.is_file()]
        for p in files:
            case_id = _case_from_name(p.name, self.cfg) or _case_from_name(p.parent.name, self.cfg)
            if not case_id: continue
            slot = bucket.setdefault(case_id, {"all":[], "mean":[], "images":[], "others":[]})
            if p.match(self.cfg.all_csv_glob): slot["all"].append(p)
            elif p.match(self.cfg.mean_csv_glob): slot["mean"].append(p)
            elif any(p.match(gl) for gl in self.cfg.image_globs): slot["images"].append(p)
            elif p.suffix.lower() in {".csv",".tif",".tiff",".png",".jpg",".jpeg",".npz"}: slot["others"].append(p)
        return bucket

class MetadataRepository:
    """Holds minimal metadata; only 'case_id' is required. Columns like Sex/Genotype are optional."""
    def __init__(self, df: Optional[pd.DataFrame]=None):
        self._df = (df.copy() if df is not None else pd.DataFrame(columns=["case_id","Sex","Genotype"]))
        if "case_id" not in self._df.columns:
            self._df["case_id"] = pd.Series(dtype=str)

    @staticmethod
    def from_csv(path_or_buffer, column_map: Dict[str, Optional[str]]):
        peek = pd.read_csv(path_or_buffer)
        out = pd.DataFrame()
        # map requested columns -> dataset columns; if missing, fill None
        for key in ("case_id","Sex","Genotype"):
            src = column_map.get(key)
            if src and src in peek.columns: out[key] = peek[src]
            else: out[key] = pd.Series([None]*len(peek))
        out = out.dropna(subset=["case_id"]).copy()
        out["case_id"] = _ensure_str_series(out["case_id"]).str.strip()
        return MetadataRepository(out.drop_duplicates(subset=["case_id"], keep="first"))

    def frame(self) -> pd.DataFrame:
        df = self._df.copy()
        if "case_id" in df.columns:
            df["case_id"] = _ensure_str_series(df["case_id"]).str.strip()
        return df

# =========================
# Assembly
# =========================
class DataAssembler:
    """
    Loads CSVs, attaches 'case_id', and returns combined per-row and per-mean DataFrames.
    Also returns image mapping & csv file map.
    """
    def __init__(self, repo: InferenceRepository, meta_repo: Optional[MetadataRepository]):
        self.repo = repo
        self.meta_repo = meta_repo

    def build(self):
        disc = self.repo.discover()
        all_parts, mean_parts = [], []
        images_by_case: Dict[str, List[Path]] = {}
        csv_map: Dict[str, Dict[str, Optional[Path]]] = {}

        for case_id, groups in disc.items():
            # newest matching files if multiple
            all_csv = _newest(groups["all"])
            mean_csv = _newest(groups["mean"])
            images_by_case[case_id] = groups["images"]
            csv_map[case_id] = {"all": all_csv, "mean": mean_csv}

            if all_csv:
                df = _read_csv_safe(all_csv)
                if df is not None and not df.empty:
                    df = df.copy(); df["case_id"] = case_id
                    all_parts.append(df)
            if mean_csv:
                md = _read_csv_safe(mean_csv)
                if md is not None and not md.empty:
                    md = md.copy(); md["case_id"] = case_id
                    mean_parts.append(md)

        all_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
        mean_df = pd.concat(mean_parts, ignore_index=True) if mean_parts else pd.DataFrame()

        # Merge metadata here too (ui will re-merge defensively)
        if self.meta_repo is not None:
            meta = self.meta_repo.frame()
            if not meta.empty:
                if not all_df.empty:
                    all_df["case_id"] = _ensure_str_series(all_df["case_id"])
                    meta["case_id"] = _ensure_str_series(meta["case_id"])
                    all_df = all_df.merge(meta, on="case_id", how="left")
                if not mean_df.empty:
                    mean_df["case_id"] = _ensure_str_series(mean_df["case_id"])
                    meta["case_id"] = _ensure_str_series(meta["case_id"])
                    mean_df = mean_df.merge(meta, on="case_id", how="left")

        return all_df, mean_df, images_by_case, csv_map

# =========================
# Filters
# =========================
class FilterSpec:
    def __init__(self, outlier_min: float=-0.4, mean_dist_max: float=140.0):
        self.outlier_min = outlier_min
        self.mean_dist_max = mean_dist_max

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        d = df.copy()
        # tolerate absence
        if "outlier_score" not in d.columns: d["outlier_score"]=np.nan
        if "mean_dist" not in d.columns: d["mean_dist"]=np.nan
        return d[(d["outlier_score"].fillna(-np.inf)>=self.outlier_min) &
                 (d["mean_dist"].fillna(np.inf)<=self.mean_dist_max)]

# =========================
# Plotting
# =========================
class PlotService:
    """Plotly/Mpl plotting utilities. Always checks presence of columns."""
    def histogram(self, df: pd.DataFrame, x: str, color: Optional[str]=None, bins:int=60):
        if df.empty or x not in df.columns: return None
        if color and color not in df.columns: color=None
        fig = px.histogram(df, x=x, color=color, nbins=bins, opacity=0.75, barmode="overlay")
        fig.update_layout(xaxis_title=x, yaxis_title="count")
        return fig

    def box(self, df: pd.DataFrame, y: str, x: Optional[str]=None, color: Optional[str]=None):
        if df.empty or y not in df.columns: return None
        if x and x not in df.columns: x=None
        if color and color not in df.columns: color=None
        fig = px.box(df, x=x, y=y, color=color, points="outliers")
        return fig

    def violin(self, df: pd.DataFrame, y: str, x: Optional[str]=None, color: Optional[str]=None):
        if df.empty or y not in df.columns: return None
        if x and x not in df.columns: x=None
        if color and color not in df.columns: color=None
        fig = px.violin(df, x=x, y=y, color=color, box=True, points="all")
        return fig

    def bar_total_vs_binned(self, df: pd.DataFrame, col: str, bins: int=20):
        """Matplotlib bar: counts per bin for numeric column."""
        if df.empty or col not in df.columns: return None
        if not pd.api.types.is_numeric_dtype(df[col]): return None
        binned = df.groupby(pd.cut(df[col], bins=bins))["case_id" if "case_id" in df else col].count()
        fig, ax = plt.subplots(figsize=(10,4))
        binned.plot(kind="bar", ax=ax)
        ax.set_xlabel(f"{col} bins"); ax.set_ylabel("count"); ax.set_title(f"Count vs {col} bins")
        plt.xticks(rotation=75)
        return fig

# =========================
# Statistics
# =========================
class StatsService:
    """Small façade over SciPy/StatsModels with defensive checks."""
    @staticmethod
    def _groups(df: pd.DataFrame, y: str, g: str, a: str, b: str):
        if df.empty or y not in df.columns or g not in df.columns: return None, None
        d = df[[y,g]].dropna()
        A = d[d[g].astype(str)==str(a)][y].astype(float)
        B = d[d[g].astype(str)==str(b)][y].astype(float)
        return A,B

    def ttest(self, df, y, g, a, b):
        A,B = self._groups(df,y,g,a,b)
        if A is None or len(A)<2 or len(B)<2: return {}
        t,p = stats.ttest_ind(A,B,equal_var=False,nan_policy="omit")
        return {"test":"Welch t-test","feature":y,"group_col":g,"A":a,"B":b,"t_stat":t,"p_value":p,
                "n_A":int(A.notna().sum()),"n_B":int(B.notna().sum())}

    def mannwhitney(self, df, y, g, a, b):
        A,B = self._groups(df,y,g,a,b)
        if A is None or len(A)<1 or len(B)<1: return {}
        u,p = stats.mannwhitneyu(A,B,alternative="two-sided")
        return {"test":"Mann–Whitney U","feature":y,"group_col":g,"A":a,"B":b,"U":u,"p_value":p,
                "n_A":int(A.notna().sum()),"n_B":int(B.notna().sum())}

    def ks(self, df, y, g, a, b):
        A,B = self._groups(df,y,g,a,b)
        if A is None or len(A)<1 or len(B)<1: return {}
        d,p = stats.ks_2samp(A,B,alternative="two-sided",mode="auto")
        return {"test":"KS 2-sample","feature":y,"group_col":g,"A":a,"B":b,"D":d,"p_value":p,
                "n_A":int(A.notna().sum()),"n_B":int(B.notna().sum())}

    def anova_df(self, df: pd.DataFrame, y: str, factors: List[str]) -> pd.DataFrame:
        """
        Fits OLS with up to 2 categorical factors:
          y ~ C(f1) (+ C(f2) + C(f1):C(f2))
        Returns ANOVA table (Type II).
        """
        if df.empty or y not in df.columns: return pd.DataFrame()
        facs = [f for f in factors if f in df.columns][:2]
        if not facs: return pd.DataFrame()
        d = df[[y]+facs].dropna().copy()
        # cast factors as string categories
        for f in facs: d[f] = d[f].astype(str)
        if len(facs)==1:
            formula = f"{y} ~ C({facs[0]}, Sum)"
        else:
            f1,f2 = facs[0], facs[1]
            formula = f"{y} ~ C({f1}, Sum) + C({f2}, Sum) + C({f1}, Sum):C({f2}, Sum)"
        try:
            model = smf.ols(formula, data=d).fit()
            aov = sm.stats.anova_lm(model, typ=2)
        except Exception:
            return pd.DataFrame()
        # make compact table
        out = aov.reset_index().rename(columns={"index":"term","PR(>F)":"p_value"})
        return out[["term","sum_sq","df","F","p_value"]].sort_values("p_value", na_position="last")

# =========================
# Upload staging (acts like 'browse' for directories)
# =========================
class UploadStager:
    """
    Stages uploaded Streamlit files into a temp directory so the app can treat them like a folder.
    """
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="adipo_stage_"))

    def stage(self, files: List[io.BytesIO]) -> Path:
        for f in files:
            name = getattr(f, "name", None) or f"file_{hash(f)}"
            out = self.root / Path(name).name
            with open(out, "wb") as w:
                w.write(f.read())
        return self.root

    def cleanup(self):
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
