import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# ─── DATA LOADING & CLEANING ───────────────────────────────────────────────
df_raw = pd.read_csv(
    "exp1_14drivers_14cars_dailyRoutes.csv",
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

# ─── APP INIT ───────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap"
    ],
    title="Fleet Analytics · OBD-II Dashboard"
)

# ─── LAYOUT ─────────────────────────────────────────────────────────────────
def kpi_card(title, value_id, unit="", icon=""):
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px", "opacity": "0.9"}),
        html.Div(title, style={"fontSize": "11px", "color": MUTED, "letterSpacing": "0.12em",
                                "textTransform": "uppercase", "marginBottom": "4px", "fontFamily": "'JetBrains Mono'"}),
        html.Div([
            html.Span(id=value_id, style={"fontSize": "26px", "fontWeight": "700",
                                           "color": ACCENT, "fontFamily": "'JetBrains Mono'"}),
            html.Span(f" {unit}", style={"fontSize": "12px", "color": MUTED, "marginLeft": "2px"}),
        ])
    ], style={
        "background": CARD_BG, "border": f"1px solid {BORDER}",
        "borderRadius": "12px", "padding": "18px 20px",
        "borderTop": f"2px solid {ACCENT}",
        "flex": "1", "minWidth": "140px"
    })

app.layout = html.Div([

    # ── HEADER ──
    html.Div([
        html.Div([
            html.Span("◈ ", style={"color": ACCENT, "fontSize": "28px"}),
            html.Span("FLEET", style={"fontFamily": "'JetBrains Mono'", "fontWeight": "700",
                                       "fontSize": "24px", "letterSpacing": "0.15em", "color": TEXT}),
            html.Span("ANALYTICS", style={"fontFamily": "'JetBrains Mono'", "fontWeight": "300",
                                           "fontSize": "24px", "letterSpacing": "0.15em", "color": ACCENT,
                                           "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div("OBD-II · 14 Vehicles · Real-Time Telemetry",
                 style={"color": MUTED, "fontSize": "12px", "fontFamily": "'JetBrains Mono'",
                        "letterSpacing": "0.08em", "marginTop": "4px"}),
    ], style={
        "background": f"linear-gradient(135deg, {BG} 0%, #0D1526 100%)",
        "borderBottom": f"1px solid {BORDER}", "padding": "24px 32px",
        "display": "flex", "flexDirection": "column"
    }),

    # ── FILTERS BAR ──
    html.Div([
        html.Div([
            html.Label("VEHICLE", style={"color": MUTED, "fontSize": "10px",
                                          "fontFamily": "'JetBrains Mono'", "letterSpacing": "0.1em",
                                          "display": "block", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="vehicle-filter",
                options=[{"label": "All Vehicles", "value": "ALL"}] +
                        [{"label": v.upper(), "value": v} for v in VEHICLES],
                value="ALL", clearable=False, multi=False,
                style={"background": CARD_BG, "minWidth": "160px"},
                className="dark-dropdown"
            ),
        ]),
        html.Div([
            html.Label("METRIC (Time Series)", style={"color": MUTED, "fontSize": "10px",
                                                       "fontFamily": "'JetBrains Mono'", "letterSpacing": "0.1em",
                                                       "display": "block", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="metric-selector",
                options=[
                    {"label": "Speed (km/h)", "value": "SPEED"},
                    {"label": "Engine RPM", "value": "ENGINE_RPM"},
                    {"label": "Engine Load (%)", "value": "ENGINE_LOAD"},
                    {"label": "Coolant Temp (°C)", "value": "ENGINE_COOLANT_TEMP"},
                    {"label": "Throttle Position (%)", "value": "THROTTLE_POS"},
                    {"label": "Air Intake Temp (°C)", "value": "AIR_INTAKE_TEMP"},
                    {"label": "MAF (g/s)", "value": "MAF"},
                ],
                value="SPEED", clearable=False,
                style={"background": CARD_BG, "minWidth": "200px"},
                className="dark-dropdown"
            ),
        ]),
        html.Div([
            html.Label("X-AXIS (Scatter)", style={"color": MUTED, "fontSize": "10px",
                                                   "fontFamily": "'JetBrains Mono'", "letterSpacing": "0.1em",
                                                   "display": "block", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="scatter-x",
                options=[
                    {"label": "Engine RPM", "value": "ENGINE_RPM"},
                    {"label": "Throttle Position (%)", "value": "THROTTLE_POS"},
                    {"label": "Engine Load (%)", "value": "ENGINE_LOAD"},
                    {"label": "Air Intake Temp (°C)", "value": "AIR_INTAKE_TEMP"},
                ],
                value="ENGINE_RPM", clearable=False,
                style={"background": CARD_BG, "minWidth": "200px"},
                className="dark-dropdown"
            ),
        ]),
    ], style={
        "background": "#0D1526", "borderBottom": f"1px solid {BORDER}",
        "padding": "16px 32px", "display": "flex", "gap": "24px",
        "alignItems": "flex-end", "flexWrap": "wrap"
    }),

   