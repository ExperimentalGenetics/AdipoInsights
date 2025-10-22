from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from engine import (
    AppConfig,
    InferenceRepository,
    MetadataRepository,
    DataAssembler,
    FilterSpec,
    PlotService,
    StatsService,
    UploadStager,
)

class AppController:
    def __init__(self):
        self.cfg = AppConfig()
        self.plot = PlotService()
        self.stats = StatsService()

    # -------------------- source intake --------------------
    def _source_ui(self) -> Path | None:
        st.subheader("Inputs")
        src = st.radio("Choose data source", ["Folder path", "Upload files"], horizontal=True)
        if src == "Folder path":
            p = st.text_input("Path to inference directory", "", placeholder="e.g. /path/to/outputs")
            return Path(p).expanduser() if p else None
        else:
            st.caption(
                "Upload any mix of your output files (e.g. *_all.csv, *_mean.csv, *_x20_wat.jpg, *_cells.jpg, *_wat_cropped.png)."
            )
            files = st.file_uploader("Drop files here", accept_multiple_files=True)
            if not files:
                return None
            if "_stager" not in st.session_state:
                st.session_state["_stager"] = UploadStager()
            staged_dir = st.session_state["_stager"].stage(files)
            st.success(f"Staged {len(files)} file(s) to {staged_dir}")
            return staged_dir

    # -------------------- metadata intake --------------------
    def _metadata_ui(self) -> MetadataRepository | None:
        st.subheader("Metadata (optional)")
        mode = st.radio(
            "Provide metadata?",
            ["None", "Upload CSV", "Path to CSV", "Inline editor"],
            horizontal=True,
            index=0,
        )

        if mode == "Upload CSV":
            up = st.file_uploader("Upload metadata CSV", type=["csv"], key="meta_up")
            if not up:
                return None
            peek = pd.read_csv(up, nrows=50)
            cols = [None] + list(peek.columns)
            c1, c2, c3 = st.columns(3)
            with c1: cid = st.selectbox("case_id column", cols)
            with c2: sex = st.selectbox("Sex column (optional)", cols)
            with c3: gty = st.selectbox("Genotype column (optional)", cols)
            up.seek(0)
            return MetadataRepository.from_csv(up, {"case_id": cid, "Sex": sex, "Genotype": gty})

        if mode == "Path to CSV":
            path = st.text_input("Path to metadata CSV", "", placeholder="e.g. /path/to/meta.csv")
            if not path:
                return None
            try:
                peek = pd.read_csv(path, nrows=50)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return None
            cols = [None] + list(peek.columns)
            c1, c2, c3 = st.columns(3)
            with c1: cid = st.selectbox("case_id column", cols, key="cid_path")
            with c2: sex = st.selectbox("Sex column (optional)", cols, key="sex_path")
            with c3: gty = st.selectbox("Genotype column (optional)", cols, key="gty_path")
            return MetadataRepository.from_csv(path, {"case_id": cid, "Sex": sex, "Genotype": gty})

        if mode == "Inline editor":
            st.caption("Create or paste a minimal table (only case_id required).")
            df = st.data_editor(
                pd.DataFrame({"case_id": [], "Sex": [], "Genotype": []}),
                num_rows="dynamic",
                key="meta_inline",
            )
            return MetadataRepository(df)

        return None

    # -------------------- pages --------------------
    def _page_analysis(self, df: pd.DataFrame):
        st.subheader("Per-row Analysis")
        if df.empty:
            st.info("No data to analyze.")
            return
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            st.info("No numeric columns found.")
            return
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != "case_id"]

        y = st.selectbox("Feature (numeric)", num_cols, index=num_cols.index("area") if "area" in num_cols else 0)
        x = st.selectbox("Group (categorical, optional)", [None] + cat_cols)
        color = st.selectbox("Color (categorical, optional)", [None] + [c for c in cat_cols if c != x])

        c1, c2 = st.columns(2)
        with c1:
            fig_b = self.plot.box(df, y=y, x=x, color=color)
            if fig_b: st.plotly_chart(fig_b, use_container_width=True)
            fig_v = self.plot.violin(df, y=y, x=x, color=color)
            if fig_v: st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            fig_h = self.plot.histogram(df, x=y, color=color, bins=60)
            if fig_h: st.plotly_chart(fig_h, use_container_width=True)

    def _page_outlier(self, df: pd.DataFrame):
        st.subheader("Explore Outlier Threshold")
        if df.empty or "outlier_score" not in df.columns:
            st.info("No outlier_score column found.")
            return
        bins = st.slider("Bins", 5, 50, 20, key="bins_outlier")
        fig = self.plot.bar_total_vs_binned(df, col="outlier_score", bins=bins)
        if fig: st.pyplot(fig)

    def _page_mean_dist(self, df: pd.DataFrame):
        st.subheader("Explore Mean Distance Threshold")
        if df.empty or "mean_dist" not in df.columns:
            st.info("No mean_dist column found.")
            return
        st.dataframe(df["mean_dist"].describe(percentiles=[.25, .5, .75, .90, .95, .99]).to_frame().T)
        bins = st.slider("Bins", 5, 50, 20, key="bins_mean")
        fig = self.plot.bar_total_vs_binned(df, col="mean_dist", bins=bins)
        if fig: st.pyplot(fig)

    def _page_stats(self, df: pd.DataFrame):
        st.subheader("Statistical Tests")
        if df.empty:
            st.info("No data for statistics.")
            return
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != "case_id"]
        if not num_cols or not cat_cols:
            st.info("Need at least one numeric feature and one categorical column.")
            return

        feature = st.selectbox("Feature (numeric)", num_cols)
        test = st.selectbox("Test", ["t-test (Welch)", "Mann–Whitney U", "KS test", "Two-way ANOVA / Linear Model"])

        if test == "Two-way ANOVA / Linear Model":
            factors = st.multiselect("Choose factors (≤2)", cat_cols, max_selections=2)
            if not factors:
                st.info("Select one or two factors.")
                return
            table = self.stats.anova_df(df, y=feature, factors=factors)
            if table.empty:
                st.warning("Model could not be fit (check data and factor levels).")
            else:
                st.dataframe(table)
            return

        group_col = st.selectbox("Group column (categorical)", cat_cols)
        levels = [str(x) for x in df[group_col].dropna().astype(str).unique().tolist()]
        if len(levels) < 2:
            st.info("Need at least two groups in the selected column.")
            return
        g1 = st.selectbox("Group A", levels)
        g2 = st.selectbox("Group B", [l for l in levels if l != g1])
        if test == "t-test (Welch)":
            res = self.stats.ttest(df, feature, group_col, g1, g2)
        elif test == "Mann–Whitney U":
            res = self.stats.mannwhitney(df, feature, group_col, g1, g2)
        else:
            res = self.stats.ks(df, feature, group_col, g1, g2)
        st.dataframe(pd.DataFrame([res]) if res else pd.DataFrame())

    # -------------------- main --------------------
    def run(self):
        st.set_page_config(page_title="Adipocyte Analysis Viewer", layout="wide")
        st.title("Adipocyte Analysis")

        root = self._source_ui()
        st.markdown("---")
        meta_repo = self._metadata_ui()

        if not root:
            st.info("Select a folder or upload a few files to proceed.")
            return

        # Discover & load
        repo = InferenceRepository(root, self.cfg)
        assembler = DataAssembler(repo, meta_repo)
        all_df, mean_df, images_by_case, csv_map = assembler.build()

        # Merge metadata here (ensures it applies even if assembler didn't merge)
        if meta_repo is not None:
            meta = meta_repo.frame()
            if not meta.empty:
                if not all_df.empty:
                    all_df["case_id"] = all_df["case_id"].astype(str)
                    meta["case_id"] = meta["case_id"].astype(str)
                    all_df = all_df.merge(meta, on="case_id", how="left")
                if not mean_df.empty:
                    mean_df["case_id"] = mean_df["case_id"].astype(str)
                    meta["case_id"] = meta["case_id"].astype(str)
                    mean_df = mean_df.merge(meta, on="case_id", how="left")

        # Minimal inline metadata when none provided
        if meta_repo is None:
            existing_ids = sorted(
                pd.concat(
                    [
                        all_df.get("case_id", pd.Series(dtype=str)).dropna().astype(str),
                        mean_df.get("case_id", pd.Series(dtype=str)).dropna().astype(str),
                    ]
                )
                .unique()
                .tolist()
            )
            if existing_ids:
                with st.expander("Optional: add minimal metadata (case_id, Sex, Genotype)"):
                    tmp = pd.DataFrame({"case_id": existing_ids, "Sex": None, "Genotype": None})
                    tmp = st.data_editor(tmp, num_rows="dynamic", key="meta_fallback")
                    if not all_df.empty:
                        all_df = all_df.merge(tmp, on="case_id", how="left")
                    if not mean_df.empty:
                        mean_df = mean_df.merge(tmp, on="case_id", how="left")

        # Sidebar filters
        st.sidebar.header("Filters")
        out_min = st.sidebar.number_input("Outlier score ≥", value=-0.40, step=0.05, format="%.3f")
        md_max = st.sidebar.number_input("Mean distance ≤", value=140.0, step=1.0)
        filtered_df = FilterSpec(outlier_min=out_min, mean_dist_max=md_max).apply(all_df)

        # Case selection
        case_ids = sorted(filtered_df.get("case_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel_all = st.checkbox("Select all cases", value=True)
        selected = case_ids if sel_all else st.multiselect("Choose cases", case_ids, default=case_ids[: min(20, len(case_ids))])
        if selected:
            filtered_df = filtered_df[filtered_df["case_id"].astype(str).isin(selected)]

        c1, c2, c3 = st.columns(3)
        c1.metric("Cases", len(selected))
        c2.metric("Rows (filtered)", len(filtered_df))
        c3.metric("Columns", filtered_df.shape[1] if not filtered_df.empty else 0)

        # Tabs
        tabs = st.tabs(["Analysis", "Outlier", "Mean distance", "Stats"])
        with tabs[0]:
            self._page_analysis(filtered_df)
        with tabs[1]:
            self._page_outlier(all_df)
        with tabs[2]:
            self._page_mean_dist(all_df)
        with tabs[3]:
            self._page_stats(filtered_df)

        # Cleanup staged uploads on user request
        if "_stager" in st.session_state:
            if st.sidebar.button("Clear uploaded temp files"):
                try:
                    st.session_state["_stager"].cleanup()
                    del st.session_state["_stager"]
                    st.sidebar.success("Cleared staged files. You can upload again.")
                except Exception:
                    st.sidebar.warning("Could not clear staged files.")
