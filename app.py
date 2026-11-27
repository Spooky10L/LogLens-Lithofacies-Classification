"""LogLens dashboard - lithofacies classification from wireline logs (FORCE 2020).

Launch (from LogLens_AI/, with the project venv active):
    streamlit run app.py

The dashboard only READS results saved by `python -m src.force_v1`; it never
retrains the model, never changes saved predictions or metrics, and never
touches the original LAS files.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dashboard_data as dd  # noqa: E402
from src import dashboard_plots as dp  # noqa: E402
from src.force_v1 import LABEL, LITHOLOGY_NAMES  # noqa: E402

st.set_page_config(page_title="LogLens | Lithofacies from Wireline Logs", page_icon="🪨",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
 #MainMenu, footer {visibility: hidden;}
 .block-container {padding-top: 1.2rem; max-width: 1500px;}
 .ll-hero {border-left: 4px solid #f2a93b; padding: 4px 0 4px 16px; margin-bottom: 10px;}
 .ll-hero h1 {margin: 0; font-size: 2.0rem; letter-spacing: .3px;}
 .ll-hero p {margin: 2px 0 0 0; color: #9fb0c0; font-size: .95rem;}
 .ll-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 10px 0 18px 0;}
 .ll-card {background: linear-gradient(160deg,#18232f,#131c26); border: 1px solid #263544; border-radius: 10px; padding: 14px 18px;}
 .ll-card .lab {font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; color: #8fa1b3; white-space: nowrap;}
 .ll-card .val {font-size: 1.9rem; font-weight: 600; color: #f3f6f9; line-height: 1.25; white-space: nowrap;}
 .ll-card .sub {font-size: .8rem; color: #8fa1b3;}
 .ll-card.good .val {color: #7bd88f;} .ll-card.warn .val {color: #f2a93b;}
 .stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #263544;}
 .stTabs [data-baseweb="tab"] {padding: 8px 16px; font-weight: 500;}
 h2, h3 {letter-spacing: .2px;}
</style>
""", unsafe_allow_html=True)

NAMES = list(LITHOLOGY_NAMES.values())
CODES = list(LITHOLOGY_NAMES)
PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"scale": 2}}


def cards(items: list[tuple]) -> None:
    """Metric cards in a responsive grid (labels never wrap). items = (label, value, sub, tone)."""
    out = ['<div class="ll-grid">']
    for it in items:
        label, value, sub, tone = (list(it) + ["", ""])[:4]
        out.append(f'<div class="ll-card {tone}"><div class="lab">{html.escape(str(label))}</div>'
                   f'<div class="val">{html.escape(str(value))}</div>'
                   f'<div class="sub">{html.escape(str(sub))}&nbsp;</div></div>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Cached loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading saved results...")
def _results(model_dir: str):
    return dd.load_results(Path(model_dir))


@st.cache_data(show_spinner=False)
def _stats(audit_dir: str):
    return dd.load_dataset_stats(Path(audit_dir))


@st.cache_data(show_spinner="Loading all labelled wells...")
def _all_labelled(path: str):
    return dd.load_labelled_all(Path(path))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🪨 LogLens")
st.sidebar.caption("Student research prototype · decision support only")
model_dir = st.sidebar.text_input("Results folder", str(ROOT / "models" / "v1"))
try:
    res = _results(model_dir)
except (dd.ResultsNotFound, dd.SchemaError) as e:
    st.title("🪨 LogLens")
    st.error(str(e))
    st.code("python -m src.force_v1 --data data/force_las --output models/v1", language="bash")
    st.stop()

stats = _stats(str(ROOT / "data" / "audit"))
all_df = _all_labelled(str(ROOT / "data" / "processed" / "force_v1_labelled.csv.gz"))
meta = res["metrics"]
catalog = dd.well_catalog(res, all_df)
if all_df is None:
    st.sidebar.info("Only the 24 held-out wells are listed. Training wells appear once "
                    "`data/processed/force_v1_labelled.csv.gz` exists (created by training).")

st.sidebar.markdown("---")
split_filter = st.sidebar.radio("Wells to show", ["validation (held-out)", "training (in-sample)", "all"], index=0)
cat = catalog if split_filter == "all" else catalog[catalog.split == split_filter]
if cat.empty:
    st.sidebar.warning("No wells in this group.")
    well = None
else:
    well = st.sidebar.selectbox("Well", cat["well"].tolist())
st.sidebar.markdown("---")
st.sidebar.caption("Dataset: FORCE 2020 well log and lithofacies dataset (CC BY 4.0), "
                   "doi.org/10.5281/zenodo.4351156")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="ll-hero"><h1>LogLens</h1>'
    "<p>Lithofacies classification from wireline logs · Random Forest on GR, RHOB, NPHI, RDEP, DTC, CALI · "
    "FORCE 2020 public benchmark (118 Norwegian Sea wells)</p></div>", unsafe_allow_html=True)
st.warning(meta["claim_scope"] + " Predictions support interpretation; they do not replace "
           "petrophysical QC or geoscientist review.", icon="⚠️")

tabs = st.tabs(["Overview", "Well viewer", "Model performance", "Per-class", "Feature importance",
                "Limitations"])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------
with tabs[0]:
    pre, sp = meta["preprocessing"], meta["split"]
    n_wells = sp["n_train_wells"] + sp["n_val_wells"]
    cards([("Wells", n_wells, "FORCE 2020, Norwegian Sea"),
           ("Depth samples", f"{pre['rows_total']:,}", "all curves, 0.152 m step"),
           ("Labelled samples", f"{pre['rows_labelled']:,}",
            f"{100 * pre['rows_labelled'] / pre['rows_total']:.1f}% of all samples")])
    cards([("Training wells", sp["n_train_wells"], f"{sp['train_rows_full']:,} labelled samples"),
           ("Validation wells", sp["n_val_wells"], f"{sp['val_rows']:,} labelled samples (all used)"),
           ("Lithofacies classes", len(CODES), f"{len(sp['classes_scored_in_validation'])} occur in validation wells")])

    left, right = st.columns([3, 2], gap="large")
    cov = res.get("coverage")
    with left:
        if cov is not None:
            cov2 = cov.assign(total=cov.train_samples + cov.val_samples)
            fig = go.Figure(go.Bar(x=cov2["class_name"], y=cov2["total"],
                                   marker_color=[dd.LITHOLOGY_COLOURS[n] for n in cov2["class_name"]],
                                   customdata=cov2[["wells_total"]].to_numpy(),
                                   hovertemplate="%{x}<br>%{y:,} samples<br>in %{customdata[0]} wells<extra></extra>"))
            fig.update_yaxes(type="log", title="labelled samples (log scale)", gridcolor="#2a3745")
            fig.update_xaxes(tickangle=-35, automargin=True)
            fig.update_layout(title="Class distribution (strongly imbalanced)", height=430, template="plotly_dark",
                              paper_bgcolor=dp.BG, plot_bgcolor=dp.PANEL, margin=dict(t=50, b=20))
            st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    with right:
        st.markdown("#### How this was built")
        st.markdown(
            "- Raw LAS files read-only; −999.25 and invalid values → NaN\n"
            "- Only rows with an interpreted lithology are used\n"
            "- Median imputation + missing-value indicators\n"
            "- **Well-wise** 80/20 split: no well is in both partitions\n"
            f"- Training capped at ~{meta['train_sampling']['max_samples']:,} rows "
            f"(sampled per well × class; {meta['train_sampling']['n_after']:,} used)\n"
            "- Evaluated on **all** labelled rows of the held-out wells\n"
            "- Depth, location and the confidence curve are **not** model inputs")
    if cov is not None:
        st.markdown("#### Class coverage and rare classes")
        st.caption("Nothing was removed to improve scores. Rare classes are flagged, and a class absent from "
                   "validation is reported as not scorable.")
        st.dataframe(cov.drop(columns=["class_code"]), width="stretch", hide_index=True, height=38 + 35 * len(cov))
    with st.expander("Held-out validation wells"):
        st.write(", ".join(res["val_wells"]))
    m = stats.get("manifest")
    need = ["well_from_filename", "n_samples", "n_labelled_samples", "pct_labelled", "depth_min_data",
            "depth_max_data", "n_log_curves"]
    if m is not None and all(c in m.columns for c in need):
        with st.expander("Dataset audit (all 118 wells)"):
            st.dataframe(m[need], width="stretch", hide_index=True)

# ----------------------------------------------------------------------------
# Well viewer
# ----------------------------------------------------------------------------
with tabs[1]:
    if well is None:
        st.info("Choose a well in the sidebar.")
    else:
        is_val = well in res["val_wells"]
        raw = dd.well_frame(res, all_df, well)
        if is_val:
            st.success(f"**{well}** is a held-out validation well: the model never saw it during training.")
        else:
            st.warning(f"**{well}** is a training well (in-sample). Predictions here are NOT a fair "
                       "performance estimate; run the model only for demonstration.")
            model_path = Path(model_dir) / "random_forest_pipeline.joblib"
            if model_path.exists() and st.button("Run saved Random Forest on this well"):
                try:
                    raw["pred_code"] = dd.predict_with_model(model_path, raw)
                    raw["pred_name"] = raw["pred_code"].map(LITHOLOGY_NAMES)
                except Exception as exc:  # e.g. scikit-learn missing
                    st.error(f"Could not run the model: {exc}")
        dmin, dmax = float(raw["DEPT"].min()), float(raw["DEPT"].max())
        info = st.columns([2, 3])
        info[0].markdown(f"**{len(raw):,}** labelled samples · **{dmin:,.0f}–{dmax:,.0f} m**")
        lo, hi = info[1].slider("Depth window (m)", dmin, dmax, (dmin, dmax), step=5.0, format="%.0f")
        g = dd.regularise_depth(raw)
        fig = dp.well_figure(g, (lo, hi))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.caption("Gaps = unlabelled or missing intervals (only labelled depths are stored). GR axis is clipped at "
                   "150 gAPI; NPHI increases to the left (neutron convention); RDEP is logarithmic. Match strip: "
                   "green = predicted equals actual, red = different.")

        if raw["pred_code"].notna().any():
            win = raw[(raw["DEPT"] >= lo) & (raw["DEPT"] <= hi)]
            mw, _ = dd.metrics_for_subset(win)
            cards([("Accuracy (this window)", f"{mw['accuracy']:.3f}", ""),
                   ("Balanced accuracy", f"{mw['balanced_accuracy']:.3f}", f"{mw['n_classes_scored']} classes present"),
                   ("Macro-F1", f"{mw['macro_f1']:.3f}", "present classes only")])
            st.caption("Single-well / window metrics use only the classes present and can differ a lot from the "
                       "24-well result. Do not quote them as model performance.")
        else:
            st.info("No predictions for this well (training well). Actual interpretation only.")

# ----------------------------------------------------------------------------
# Model performance
# ----------------------------------------------------------------------------
with tabs[2]:
    mv = meta["metrics_validation"]
    rf, base = mv["random_forest"], mv["dummy_most_frequent"]
    st.subheader("Held-out validation wells (24 wells, all labelled samples)")
    cards([("Accuracy", f"{rf['accuracy']:.4f}", f"{rf['accuracy'] - base['accuracy']:+.4f} vs always-Shale"),
           ("Balanced accuracy", f"{rf['balanced_accuracy']:.4f}",
            f"{rf['balanced_accuracy'] - base['balanced_accuracy']:+.4f} vs always-Shale", "good"),
           ("Macro-F1", f"{rf['macro_f1']:.4f}", f"{rf['macro_f1'] - base['macro_f1']:+.4f} vs always-Shale", "good"),
           ("Weighted-F1", f"{rf['weighted_f1']:.4f}", f"{rf['weighted_f1'] - base['weighted_f1']:+.4f} vs always-Shale")])
    st.info(
        f"Macro averages cover the {rf['n_classes_scored']} classes that occur in the validation wells. "
        "Basement exists in only one well (a training well), so it cannot be scored: it is reported as "
        "'not scorable', not as 0. Accuracy alone is misleading: always predicting Shale already gives "
        f"{base['accuracy']:.3f} accuracy but a balanced accuracy of only {base['balanced_accuracy']:.3f}.")
    table = pd.DataFrame(mv).T.rename_axis("model").reset_index()
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%.4f")
                                for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")})

    cm = res.get("confusion")
    if cm is not None:
        st.subheader("Confusion matrix")
        mode = st.radio("Display", ["Row-normalised (recall per true class)", "Sample counts"], horizontal=True)
        fig, dropped = dp.confusion_figure(cm, normalise=mode.startswith("Row"))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        n_shown = len(cm) - len(dropped)
        note = f"All {n_shown} classes present in the validation wells are shown."
        if dropped:
            note += (f" Not shown: {', '.join(dropped)} (no validation samples and never predicted; "
                     "it is not scorable, and it has not been removed from the data).")
        st.caption(note + " Rows = true class, columns = predicted class; the diagonal is correct.")
    if "audit" in res:
        st.markdown("**Largest confusions (Random Forest)**")
        st.dataframe(pd.DataFrame(res["audit"]["top_confusions_random_forest"]), width="stretch", hide_index=True)
    pw = res["per_well"].sort_values("accuracy")
    fig = go.Figure(go.Bar(x=pw["well"], y=pw["accuracy"], marker_color="#5aa9ff",
                           customdata=pw[["n_labelled"]].to_numpy(),
                           hovertemplate="%{x}<br>accuracy %{y:.3f}<br>%{customdata[0]:,} samples<extra></extra>"))
    fig.add_hline(y=rf["accuracy"], line_dash="dot", line_color="#f2a93b",
                  annotation_text=f"overall {rf['accuracy']:.3f}", annotation_position="top left")
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(title="Accuracy varies a lot from well to well", height=400, template="plotly_dark",
                      paper_bgcolor=dp.BG, plot_bgcolor=dp.PANEL, yaxis_range=[0, 1], margin=dict(t=50))
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)

# ----------------------------------------------------------------------------
# Per-class
# ----------------------------------------------------------------------------
with tabs[3]:
    pc = res["per_class"].copy()
    scored = pc[pc.support_val > 0]
    fig = go.Figure()
    for metric, colour in [("precision", "#5aa9ff"), ("recall", "#7bd88f"), ("f1", "#f2a93b")]:
        fig.add_trace(go.Bar(x=pc["class_name"], y=pc[metric], name=metric, marker_color=colour))
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(barmode="group", height=440, template="plotly_dark", yaxis_range=[0, 1],
                      paper_bgcolor=dp.BG, plot_bgcolor=dp.PANEL,
                      title="Per-class precision / recall / F1 (held-out wells)", margin=dict(t=50))
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    shown = pc.copy()  # display copy only: a class with no validation samples shows "n/a", never 0
    for k in ("precision", "recall", "f1"):
        shown[k] = shown[k].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    shown["status"] = shown["status"].str.replace("NOT SCORABLE: absent from validation wells",
                                                  "not scorable", regex=False)
    st.dataframe(shown, width="stretch", hide_index=True, height=38 + 35 * len(shown))
    weak = scored[scored.f1 < 0.35].sort_values("f1")
    if len(weak):
        st.markdown("**Classes with F1 below 0.35** (computed from the table above)")
        for _, r in weak.iterrows():
            cw, extra = res.get("coverage"), ""
            if cw is not None:
                row = cw[cw.class_name == r["class_name"]].iloc[0]
                extra = f"; in {int(row.train_wells)} training wells and {int(row.val_wells)} validation wells"
            st.markdown(f"- **{r['class_name']}**: F1 {r['f1']:.2f}, precision {r['precision']:.2f}, "
                        f"recall {r['recall']:.2f}, {int(r['support_val']):,} validation samples{extra}")
    st.caption("Small supports (few wells) make per-class scores unstable. For example, a class present in a "
               "single validation well mostly measures that one well. n/a = not scorable.")

# ----------------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------------
with tabs[4]:
    fi = res.get("importance")
    if fi is None:
        st.info("feature_importance.csv not found.")
    else:
        s = fi["importance"].sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                               marker_color=["#f2a93b" if not i.startswith("missingindicator") else "#6b7a8a"
                                             for i in s.index]))
        fig.update_layout(height=440, template="plotly_dark", paper_bgcolor=dp.BG, plot_bgcolor=dp.PANEL,
                          title="Random Forest impurity importance", xaxis_title="mean decrease in impurity",
                          margin=dict(t=50))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        st.caption("Importance is spread fairly evenly across the six logs, so no single curve dominates. "
                   "Impurity importance is biased towards continuous features and splits credit among "
                   "correlated curves; it shows what the forest used, not geological causality. "
                   "Grey bars are missing-value indicators.")

# ----------------------------------------------------------------------------
# Limitations
# ----------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
**What this is:** LogLens is an independent student research prototype developed for educational and research purposes. It is a machine-learning-based lithofacies classification and decision-support application, not a commercial-grade petrophysical interpretation system.

It is trained on the public FORCE 2020 dataset (Bormann et al., Zenodo, CC BY 4.0).

**Known limitations**
- One 80/20 well-wise split (seed {meta['seed']}); no repeated cross-validation, so the scores carry split-to-split uncertainty.
- Not the official FORCE 2020 competition test scoring; numbers are not comparable with leaderboard results.
- Each depth sample is classified independently; the vertical sequence of beds is ignored.
- Basement (1 well) and Halite (4 wells) are very rare; Chalk (15 wells) and Anhydrite (10 wells) are uncommon. Basement cannot be scored at all.
- Labels are expert interpretations, not core-verified truth. The confidence curve is saved but unused.
- Training used a ~300k-row sample of the 1.14M training rows; no hyper-parameter tuning was done.
- Missing log values are median-imputed, which is not a petrophysical estimate.
- Predictions are decision-support only and must be reviewed by a petrophysicist / geoscientist.
""")
    st.caption(f"Software: Python {meta['versions']['python']}, scikit-learn {meta['versions']['sklearn']}, "
               f"pandas {meta['versions']['pandas']}")
