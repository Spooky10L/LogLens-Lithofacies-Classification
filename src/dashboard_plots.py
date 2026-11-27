"""Plotly figure builders for the LogLens dashboard (no Streamlit import, so they are testable).

Design notes
------------
* Log curves use plain SVG `go.Scatter`, NOT `go.Scattergl`. WebGL traces did not draw in the
  embedded browser and are blank in browser print/PDF export; SVG draws everywhere.
* Every track shares one fixed depth axis (explicit range, reversed: depth increases downwards).
* Lithology strips are drawn as 2-column heatmaps (a 1-column heatmap has degenerate width).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dashboard_data import LITHOLOGY_COLOURS, lithology_index
from .force_v1 import LABEL, LITHOLOGY_NAMES

CODES = list(LITHOLOGY_NAMES)
NAMES = list(LITHOLOGY_NAMES.values())
BG = "#0f1720"
PANEL = "#131c26"
GRID = "#2a3745"
TEXT = "#d5dde5"

# (column, title, colour, axis kwargs)
CURVE_TRACKS = [
    ("GR", "GR · gAPI", "#7bd88f", dict(range=[0, 150])),
    ("RHOB", "RHOB · g/cm³", "#ff6b6b", dict(range=[1.8, 3.0])),
    ("NPHI", "NPHI · v/v", "#5aa9ff", dict(range=[0.6, -0.1])),  # neutron: increases to the left
    ("RDEP", "RDEP · Ω·m", "#f2a93b", dict(type="log", range=[np.log10(0.2), np.log10(2000)])),
]
ACTUAL_COLOUR, PRED_COLOUR, MATCH_COLOUR = "#4fd1c5", "#f2a93b", "#9aa7b4"


def _discrete_scale():
    k = len(CODES)
    scale = []
    for i, name in enumerate(NAMES):
        c = LITHOLOGY_COLOURS[name]
        scale += [[i / k, c], [(i + 1) / k, c]]
    return scale


def lithology_strip(codes: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(z, names) 2-column arrays for a heatmap strip. z is the class index + 0.5 (NaN = no data)."""
    idx = lithology_index(codes).to_numpy(dtype=float)
    z = np.repeat((idx + 0.5)[:, None], 2, axis=1)
    names = np.array([[NAMES[int(i)] if np.isfinite(i) else ""] * 2 for i in idx], dtype=object)
    return z, names


def agreement_strip(actual: pd.Series, pred: pd.Series) -> np.ndarray:
    ok = np.where(actual.notna() & pred.notna(), (actual == pred).astype(float), np.nan)
    return np.repeat(ok[:, None], 2, axis=1)


def well_figure(g: pd.DataFrame, depth_range: tuple[float, float] | None = None,
                height: int = 860) -> go.Figure:
    """g = a depth-regularised well frame (see dashboard_data.regularise_depth)."""
    d0, d1 = depth_range if depth_range else (float(g["DEPT"].min()), float(g["DEPT"].max()))
    g = g[(g["DEPT"] >= d0) & (g["DEPT"] <= d1)]
    depth = g["DEPT"]
    has_pred = "pred_code" in g and g["pred_code"].notna().any()

    titles = [t[1] for t in CURVE_TRACKS] + ["ACTUAL", "PREDICTED", "MATCH"]
    fig = make_subplots(rows=1, cols=7, shared_yaxes=True, horizontal_spacing=0.007,
                        column_widths=[1, 1, 1, 1, 0.55, 0.55, 0.2], subplot_titles=titles)
    for i, (col, _, colour, axis) in enumerate(CURVE_TRACKS, start=1):
        fig.add_trace(go.Scatter(x=g[col], y=depth, mode="lines", line=dict(color=colour, width=1.1),
                                 name=col, showlegend=False, connectgaps=False,
                                 hovertemplate=f"{col} %{{x:.3g}}<br>%{{y:.1f}} m<extra></extra>"),
                      row=1, col=i)
        fig.update_xaxes(row=1, col=i, showgrid=True, gridcolor=GRID, showline=True, linecolor=GRID,
                         mirror=True, side="bottom", ticks="outside", tickfont=dict(size=10), **axis)

    scale = _discrete_scale()
    z_act, n_act = lithology_strip(g[LABEL])
    pred = g["pred_code"] if "pred_code" in g else pd.Series(np.nan, index=g.index)
    z_pred, n_pred = lithology_strip(pred)
    for col_i, (z, names, label) in [(5, (z_act, n_act, "Actual")), (6, (z_pred, n_pred, "RF predicted"))]:
        fig.add_trace(go.Heatmap(z=z, x=[0, 1], y=depth, colorscale=scale, zmin=0, zmax=len(CODES),
                                 showscale=False, text=names, xgap=0, ygap=0,
                                 hovertemplate=f"{label}: %{{text}}<br>%{{y:.1f}} m<extra></extra>"),
                      row=1, col=col_i)
    fig.add_trace(go.Heatmap(z=agreement_strip(g[LABEL], pred), x=[0, 1], y=depth, zmin=0, zmax=1,
                             colorscale=[[0, "#d9534f"], [0.5, "#d9534f"], [0.5, "#3fb27f"], [1, "#3fb27f"]],
                             showscale=False, hoverinfo="skip"), row=1, col=7)
    for col_i in (5, 6, 7):
        fig.update_xaxes(row=1, col=col_i, range=[-0.5, 1.5], showticklabels=False, showgrid=False,
                         showline=True, linecolor=GRID, mirror=True)

    for nme in NAMES:  # legend entries for the lithology colours
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nme,
                                 marker=dict(size=11, color=LITHOLOGY_COLOURS[nme], symbol="square",
                                             line=dict(color="#000", width=0.5))), row=1, col=5)

    fig.update_yaxes(range=[d1, d0], showgrid=True, gridcolor=GRID, showline=True, linecolor=GRID,
                     mirror=True, ticks="outside", dtick=None)
    fig.update_yaxes(title_text="Measured depth (m)", row=1, col=1)
    for c in range(2, 8):
        fig.update_yaxes(showticklabels=False, row=1, col=c)
    # Coloured headers so the three lithology columns cannot be confused.
    header_colours = {"ACTUAL": ACTUAL_COLOUR, "PREDICTED": PRED_COLOUR, "MATCH": MATCH_COLOUR}
    for a in fig.layout.annotations:
        a.font = dict(size=12, color=header_colours.get(a.text, TEXT))
        a.y = a.y + 0.012
    if not has_pred:
        fig.add_annotation(xref="x6 domain", yref="y6 domain", x=0.5, y=0.5, showarrow=False,
                           text="no prediction<br>for this well", font=dict(color="#8b98a5", size=11))
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL,
                      margin=dict(t=60, b=120, l=70, r=15), font=dict(color=TEXT, size=11),
                      legend=dict(orientation="h", y=-0.075, x=0.0, font=dict(size=11),
                                  bgcolor="rgba(0,0,0,0)"),
                      hovermode="closest")
    return fig


def drop_empty_classes(cm: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove classes that have NO true samples AND were never predicted (reported, not hidden)."""
    empty = [c for c in cm.index if cm.loc[c].sum() == 0 and cm[c].sum() == 0]
    return cm.drop(index=empty, columns=empty), empty


def confusion_figure(cm: pd.DataFrame, normalise: bool = True) -> tuple[go.Figure, list[str]]:
    cm, dropped = drop_empty_classes(cm)
    arr = cm.to_numpy().astype(float)
    n = len(cm)
    if normalise:
        rs = arr.sum(1, keepdims=True)
        z = np.where(rs > 0, arr / np.where(rs == 0, 1, rs), np.nan)
        text = np.where(np.isnan(z), "", np.char.mod("%.2f", z))
        bar_title = "share of true class"
    else:
        z = np.where(arr > 0, arr, np.nan)
        text = np.where(arr > 0, np.char.mod("%d", arr), "")
        bar_title = "samples"
    fig = go.Figure(go.Heatmap(
        z=z, x=list(cm.columns), y=list(cm.index), text=text, texttemplate="%{text}",
        textfont=dict(size=12), colorscale="Blues", xgap=2, ygap=2,
        colorbar=dict(title=bar_title, thickness=14),
        hovertemplate="true: %{y}<br>predicted: %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="True lithology (FORCE interpretation)", automargin=True,
                     tickfont=dict(size=12))
    fig.update_xaxes(title="Predicted lithology (Random Forest)", side="bottom", tickangle=-40,
                     automargin=True, tickfont=dict(size=12))
    fig.update_layout(height=max(560, 52 * n + 190), template="plotly_dark", paper_bgcolor=BG,
                      plot_bgcolor=PANEL, margin=dict(t=20, l=10, r=10, b=10))
    return fig, dropped
