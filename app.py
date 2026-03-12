import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ─── DATA LOADING & CLEANING ───────────────────────────────────────────────
df_raw = pd.read_csv(
    "1773255985411_exp1_14drivers_14cars_dailyRoutes.csv",
    low_memory=False
)

def clean_pct(series):
    return pd.to_numeric(
        series.astype(str).str.replace("%", "").str.replace(",", ".").str.strip(),
        errors="coerce"
    )

for col in ["FUEL_LEVEL", "ENGINE_LOAD", "THROTTLE_POS", "TIMING_ADVANCE", "EQUIV_RATIO"]:
    if col in df_raw.columns:
        df_raw[col] = clean_pct(df_raw[col])

for col in ["SPEED", "ENGINE_RPM", "ENGINE_COOLANT_TEMP", "MAF",
            "INTAKE_MANIFOLD_PRESSURE", "AIR_INTAKE_TEMP", "BAROMETRIC_PRESSURE(KPA)"]:
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

df_raw["TIMESTAMP"] = pd.to_numeric(df_raw["TIMESTAMP"], errors="coerce")
df_raw["datetime"] = pd.to_datetime(df_raw["TIMESTAMP"], unit="ms", errors="coerce")
df_raw = df_raw[df_raw["VEHICLE_ID"].notna() & df_raw["VEHICLE_ID"].str.startswith("car", na=False)].copy()

VEHICLES = sorted(df_raw["VEHICLE_ID"].unique())

# ─── PRECOMPUTE KPIs ────────────────────────────────────────────────────────
def compute_kpis(df):
    total_records = len(df)
    avg_speed     = df["SPEED"].mean()
    max_speed     = df["SPEED"].max()
    avg_rpm       = df["ENGINE_RPM"].mean()
    avg_load      = df["ENGINE_LOAD"].mean()
    avg_coolant   = df["ENGINE_COOLANT_TEMP"].mean()
    n_vehicles    = df["VEHICLE_ID"].nunique()
    return {
        "records": total_records,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "avg_rpm": avg_rpm,
        "avg_load": avg_load,
        "avg_coolant": avg_coolant,
        "n_vehicles": n_vehicles,
    }

# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────
ACCENT   = "#00D4FF"
ACCENT2  = "#FF6B35"
ACCENT3  = "#A78BFA"
BG       = "#0A0E1A"
CARD_BG  = "#111827"
BORDER   = "#1F2937"
TEXT     = "#E5E7EB"
MUTED    = "#6B7280"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="'JetBrains Mono', monospace"),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
        colorway=[ACCENT, ACCENT2, ACCENT3, "#34D399", "#FBBF24", "#F87171"],
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
)

def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig

