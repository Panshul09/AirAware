from __future__ import annotations

import html
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry
import requests
import streamlit as st
from branca.element import Element
from folium.raster_layers import ImageOverlay

# ============================================================
# AIR AWARE — GLOBAL ENVIRONMENTAL INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="AirAware | Global Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIMEZONE_DEFAULT = "UTC"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CITY_DATA_URL = "https://raw.githubusercontent.com/datasets/world-cities/main/data/world-cities.csv"
REGION_PLACEHOLDER = "Select a state / region"
CITY_PLACEHOLDER = "Select a city"
NO_REGION = "Other / no region listed"

# These are representative major cities used for the comparison panel.
FEATURED_CITIES = [
    ("New York", "United States", 40.7128, -74.0060),
    ("Los Angeles", "United States", 34.0522, -118.2437),
    ("Toronto", "Canada", 43.6532, -79.3832),
    ("Vancouver", "Canada", 49.2827, -123.1207),
    ("Mexico City", "Mexico", 19.4326, -99.1332),
    ("São Paulo", "Brazil", -23.5505, -46.6333),
    ("Buenos Aires", "Argentina", -34.6037, -58.3816),
    ("London", "United Kingdom", 51.5072, -0.1276),
    ("Paris", "France", 48.8566, 2.3522),
    ("Berlin", "Germany", 52.5200, 13.4050),
    ("Madrid", "Spain", 40.4168, -3.7038),
    ("Rome", "Italy", 41.9028, 12.4964),
    ("Istanbul", "Türkiye", 41.0082, 28.9784),
    ("Cairo", "Egypt", 30.0444, 31.2357),
    ("Lagos", "Nigeria", 6.5244, 3.3792),
    ("Nairobi", "Kenya", -1.2921, 36.8219),
    ("Johannesburg", "South Africa", -26.2041, 28.0473),
    ("Dubai", "United Arab Emirates", 25.2048, 55.2708),
    ("Riyadh", "Saudi Arabia", 24.7136, 46.6753),
    ("Karachi", "Pakistan", 24.8607, 67.0011),
    ("Lahore", "Pakistan", 31.5204, 74.3587),
    ("Delhi", "India", 28.6139, 77.2090),
    ("Mumbai", "India", 19.0760, 72.8777),
    ("Dhaka", "Bangladesh", 23.8103, 90.4125),
    ("Bangkok", "Thailand", 13.7563, 100.5018),
    ("Singapore", "Singapore", 1.3521, 103.8198),
    ("Jakarta", "Indonesia", -6.2088, 106.8456),
    ("Beijing", "China", 39.9042, 116.4074),
    ("Shanghai", "China", 31.2304, 121.4737),
    ("Tokyo", "Japan", 35.6762, 139.6503),
    ("Seoul", "South Korea", 37.5665, 126.9780),
    ("Sydney", "Australia", -33.8688, 151.2093),
    ("Melbourne", "Australia", -37.8136, 144.9631),
    ("Auckland", "New Zealand", -36.8509, 174.7645),
]

AQI_COLORS = {
    "Good": "#16845B",
    "Moderate": "#B28A00",
    "Unhealthy for sensitive groups": "#CA6A13",
    "Unhealthy": "#D64545",
    "Very unhealthy": "#8E44AD",
    "Hazardous": "#712638",
    "No data": "#7B8794",
}

AQI_DESCRIPTIONS = {
    "Good": "The estimate is in the Good category of the US AQI scale.",
    "Moderate": "The estimate is Moderate. Sensitive people should continue monitoring local conditions.",
    "Unhealthy for sensitive groups": "Sensitive people may wish to reduce prolonged or strenuous outdoor activity.",
    "Unhealthy": "Check trusted local public-health guidance and monitor changing conditions.",
    "Very unhealthy": "The model estimates very high pollution. Follow trusted public-health guidance.",
    "Hazardous": "The estimate is in the highest US AQI category. Consult trusted public-health guidance.",
    "No data": "No usable AQI estimate is available for this location.",
}

# Gradient is intentionally similar to the supplied reference: blue -> green ->
# yellow -> orange -> red -> violet. It colours an interpolated raster overlay,
# not individual city markers.
HEAT_STOPS = np.array([0, 30, 60, 100, 140, 180, 230, 300], dtype=float)
HEAT_RGB = np.array(
    [
        [48, 82, 210],
        [34, 120, 232],
        [38, 190, 112],
        [155, 215, 68],
        [245, 222, 65],
        [245, 153, 55],
        [228, 65, 57],
        [116, 42, 144],
    ],
    dtype=float,
)

# ============================================================
# LIGHT THEME + CONSISTENT COMPONENT STYLING
# A .streamlit/config.toml is also included in the ZIP so native
# Streamlit widgets start in light mode rather than OS dark mode.
# ============================================================

st.markdown(
    """
    <style>
    :root { color-scheme: light !important; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        color-scheme: light !important; background:#F6F8F6 !important;
        color:#17251D !important; font-family:Inter,"Segoe UI",Arial,sans-serif !important;
    }
    [data-testid="stHeader"] { background:#F6F8F6 !important; }
    [data-testid="stSidebar"] { background:#FFFFFF !important; border-right:1px solid #E2E8E2 !important; }
    [data-testid="stSidebar"] .block-container { padding:1.25rem .95rem 2rem !important; }
    h1,h2,h3,h4 { color:#17251D !important; letter-spacing:-.025em !important; }
    p, label, [data-testid="stWidgetLabel"] p { color:#34443A !important; }
    [data-testid="stCaptionContainer"] { color:#68766D !important; }

    /* Force all Streamlit buttons to a white surface with dark text. */
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button,
    .stApp [data-testid="stFormSubmitButton"] button,
    .stApp button[data-testid^="stBaseButton-"] {
        appearance:none !important; background:#FFFFFF !important;
        background-color:#FFFFFF !important; background-image:none !important;
        color:#111B14 !important; -webkit-text-fill-color:#111B14 !important;
        border:1px solid #CDD8CE !important; border-radius:10px !important;
        box-shadow:0 1px 2px rgba(20,35,24,.05) !important;
        opacity:1 !important; font-weight:650 !important; min-height:40px !important;
    }
    .stApp [data-testid="stButton"] button *,
    .stApp [data-testid="stDownloadButton"] button *,
    .stApp [data-testid="stFormSubmitButton"] button *,
    .stApp button[data-testid^="stBaseButton-"] * {
        color:#111B14 !important; -webkit-text-fill-color:#111B14 !important;
        fill:#111B14 !important; opacity:1 !important;
    }
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover,
    .stApp [data-testid="stFormSubmitButton"] button:hover,
    .stApp button[data-testid^="stBaseButton-"]:hover {
        background:#F1F4F1 !important; background-color:#F1F4F1 !important;
        border-color:#9DAD9F !important; color:#101A13 !important;
        -webkit-text-fill-color:#101A13 !important;
    }
    .stApp [data-testid="stButton"] button:disabled,
    .stApp button[data-testid^="stBaseButton-"]:disabled {
        background:#F5F7F5 !important; color:#7B867D !important;
        -webkit-text-fill-color:#7B867D !important;
    }

    /* Native dropdowns and input menus remain white on dark-mode systems. */
    .stApp [data-baseweb="select"] > div,
    .stApp [data-baseweb="input"] > div,
    .stApp [data-baseweb="input"] input {
        background:#FFFFFF !important; background-color:#FFFFFF !important;
        color:#17251D !important; -webkit-text-fill-color:#17251D !important;
        border-color:#D6E0D7 !important; border-radius:10px !important;
        box-shadow:none !important;
    }
    .stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] *,
    .stApp [data-baseweb="menu"] *, .stApp [data-baseweb="popover"] * {
        color:#17251D !important; -webkit-text-fill-color:#17251D !important;
    }
    .stApp [data-baseweb="menu"], .stApp [data-baseweb="popover"],
    .stApp [role="listbox"] { background:#FFFFFF !important; }
    .stApp [role="option"]:hover, .stApp [role="option"][aria-selected="true"] { background:#EEF3EF !important; }
    .stApp [data-baseweb="select"] svg { fill:#46554A !important; color:#46554A !important; }

    /* Quiet grey/white forecast-range radios. */
    .stApp [data-testid="stRadio"] label, .stApp [data-testid="stRadio"] label * { color:#29372E !important; }
    .stApp [data-testid="stRadio"] [role="radio"] {
        background:#FFFFFF !important; border:1.5px solid #A5B0A7 !important; box-shadow:none !important;
    }
    .stApp [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        background:#E7EDE8 !important; border-color:#7D8B80 !important;
    }
    [data-testid="stMetric"] { background:#FFFFFF !important; border:1px solid #E2E8E2 !important; border-radius:14px !important; padding:16px 18px !important; }
    [data-testid="stMetricLabel"] { color:#66746A !important; }
    [data-testid="stMetricValue"] { color:#17251D !important; font-weight:740 !important; }
    [data-testid="stPlotlyChart"] { border-radius:14px; overflow:hidden; }
    hr { border-color:#E4EAE4 !important; }

    .air-brand { color:#216B45; font-size:24px; font-weight:820; letter-spacing:-.8px; }
    .air-brand-sub { color:#879188; font-size:9px; letter-spacing:1.35px; text-transform:uppercase; margin-top:2px; }
    .air-hero { background:linear-gradient(120deg,#FFFFFF 0%,#F0F6F0 100%); border:1px solid #E2E9E2; border-radius:19px; padding:26px 30px; margin:4px 0 21px; }
    .air-eyebrow { color:#28734A; font-size:10px; font-weight:800; letter-spacing:1.65px; text-transform:uppercase; margin-bottom:10px; }
    .air-title { color:#17251D; font-size:clamp(31px,4vw,46px); font-weight:820; line-height:1.1; letter-spacing:-1.9px; margin-bottom:10px; }
    .air-subtitle { color:#67756C; font-size:14px; line-height:1.75; max-width:780px; }
    .air-section-label { color:#28734A; font-size:10px; font-weight:800; letter-spacing:1.65px; text-transform:uppercase; margin-bottom:5px; }
    .air-muted { color:#758178; font-size:12px; line-height:1.75; }
    .air-selected { background:#F7FAF7; border:1px solid #E1E8E1; border-radius:11px; padding:11px 12px; }
    .air-status { background:#FFFFFF; border:1px solid #E2E9E2; border-left:5px solid #28734A; border-radius:13px; padding:16px 19px; margin:5px 0 17px; }
    .air-status-label { color:#7B887E; font-size:10px; font-weight:750; letter-spacing:1.2px; text-transform:uppercase; }
    .air-status-title { font-size:22px; font-weight:800; margin:4px 0; }
    .air-info { background:#FFFFFF; border:1px solid #E1E8E1; border-radius:13px; padding:18px; min-height:112px; }
    .air-legend { display:flex; align-items:center; gap:10px; flex-wrap:wrap; font-size:11px; color:#65736A; margin:9px 0 12px; }
    .air-gradient { height:10px; width:220px; border-radius:20px; border:1px solid rgba(0,0,0,.08); background:linear-gradient(90deg,#3052D2,#2278E8,#26BE70,#9BD744,#F5DE41,#F59937,#E44139,#742A90); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CITY CATALOG: WORLD CITIES WITH COUNTRY + FIRST-LEVEL REGION
# Dataset: GeoNames, published as the DataHub world-cities dataset.
# Includes cities generally above 15,000 residents plus capitals.
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def load_city_catalog() -> pd.DataFrame:
    catalog = pd.read_csv(CITY_DATA_URL, keep_default_na=False)
    required = {"name", "country", "subcountry", "geonameid"}
    if not required.issubset(catalog.columns):
        raise ValueError("The global city dataset has an unexpected format.")
    catalog = catalog[list(required)].copy()
    for column in ("name", "country", "subcountry"):
        catalog[column] = catalog[column].astype(str).str.strip()
    catalog["geonameid"] = pd.to_numeric(catalog["geonameid"], errors="coerce")
    catalog = catalog.dropna(subset=["geonameid"])
    catalog["geonameid"] = catalog["geonameid"].astype("int64")
    catalog = catalog[(catalog["name"] != "") & (catalog["country"] != "")]
    catalog["region"] = catalog["subcountry"].replace("", NO_REGION)
    catalog = catalog.drop_duplicates(subset=["geonameid"])
    return catalog.sort_values(["country", "region", "name"], kind="stable").reset_index(drop=True)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("-", " ").replace("_", " ").split())


def country_code_for_name(country_name: str) -> str | None:
    if normalize_text(country_name) == "kosovo":
        return "XK"
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_catalog_city(city_name: str, country_name: str, region_name: str, geoname_id: int) -> dict | None:
    country_code = country_code_for_name(country_name)
    if not country_code:
        return None
    response = requests.get(
        GEOCODING_URL,
        params={"name": city_name, "count": 100, "language": "en", "countryCode": country_code},
        timeout=25,
    )
    response.raise_for_status()
    results = response.json().get("results", []) or []
    if not results:
        return None

    # Prefer the exact GeoNames ID from the city dropdown's catalog row.
    for item in results:
        if int(item.get("id", -1)) == int(geoname_id):
            return item

    city_norm = normalize_text(city_name)
    region_norm = normalize_text(region_name)
    exact_name = [item for item in results if normalize_text(item.get("name")) == city_norm]
    if region_norm and region_norm != normalize_text(NO_REGION):
        same_region = [
            item for item in exact_name
            if normalize_text(item.get("admin1", "")) == region_norm
            or region_norm in normalize_text(item.get("admin1", ""))
            or normalize_text(item.get("admin1", "")) in region_norm
        ]
        if same_region:
            exact_name = same_region
    candidates = exact_name or results
    return max(candidates, key=lambda item: int(item.get("population") or 0))


def reset_region_and_city() -> None:
    st.session_state["region_name"] = REGION_PLACEHOLDER
    st.session_state["city_geoname_id"] = None


def reset_city_only() -> None:
    st.session_state["city_geoname_id"] = None


# Load the global city catalog before drawing dependent selectboxes.
with st.spinner("Loading the worldwide city directory..."):
    try:
        city_catalog = load_city_catalog()
    except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error("AirAware could not load its global city directory. Check your internet connection and restart the app.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

country_options = sorted(city_catalog["country"].dropna().unique().tolist(), key=str.casefold)
if not country_options:
    st.error("The city directory did not contain any countries.")
    st.stop()

if "country_name" not in st.session_state:
    st.session_state["country_name"] = "Pakistan" if "Pakistan" in country_options else country_options[0]
if st.session_state["country_name"] not in country_options:
    st.session_state["country_name"] = country_options[0]

country_rows = city_catalog[city_catalog["country"] == st.session_state["country_name"]]
region_options_current = sorted(country_rows["region"].dropna().unique().tolist(), key=str.casefold)
if st.session_state.get("region_name") not in ([REGION_PLACEHOLDER] + region_options_current):
    preferred = "Sindh" if st.session_state["country_name"] == "Pakistan" and "Sindh" in region_options_current else REGION_PLACEHOLDER
    st.session_state["region_name"] = preferred
if "city_geoname_id" not in st.session_state:
    st.session_state["city_geoname_id"] = None
if "forecast_window" not in st.session_state:
    st.session_state["forecast_window"] = "48 hours"

# Clean, compact sidebar: only the three location selectors and forecast range.
st.sidebar.markdown(
    '<div class="air-brand">🌍 AirAware</div>'
    '<div class="air-brand-sub">Global environmental intelligence</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("STABILIZED BUILD · v2")
st.sidebar.markdown("---")
st.sidebar.markdown("**LOCATION**")
selected_country_name = st.sidebar.selectbox(
    "Country",
    country_options,
    key="country_name",
    on_change=reset_region_and_city,
)
country_rows = city_catalog[city_catalog["country"] == selected_country_name]
region_options_current = sorted(country_rows["region"].dropna().unique().tolist(), key=str.casefold)
region_widget_options = [REGION_PLACEHOLDER] + region_options_current
if st.session_state.get("region_name") not in region_widget_options:
    st.session_state["region_name"] = REGION_PLACEHOLDER
selected_region_name = st.sidebar.selectbox(
    "State / province / region",
    region_widget_options,
    key="region_name",
    on_change=reset_city_only,
    disabled=len(region_options_current) == 0,
)

if selected_region_name == REGION_PLACEHOLDER:
    region_city_rows = country_rows.iloc[0:0].copy()
else:
    region_city_rows = country_rows[country_rows["region"] == selected_region_name].copy()

region_city_rows = region_city_rows.sort_values("name", key=lambda col: col.str.casefold(), kind="stable")
# The widget stores GeoNames IDs, so two cities with identical names remain distinct.
city_id_options = [int(value) for value in region_city_rows["geonameid"].tolist()]
city_display_by_id: dict[int, str] = {}
name_counts = region_city_rows["name"].value_counts().to_dict()
for _, row in region_city_rows.iterrows():
    name = str(row["name"])
    city_display_by_id[int(row["geonameid"])] = (
        f"{name} · place {int(row['geonameid'])}" if name_counts.get(name, 0) > 1 else name
    )

if st.session_state.get("city_geoname_id") not in city_id_options:
    st.session_state["city_geoname_id"] = None

selected_city_id = st.sidebar.selectbox(
    "City",
    options=[None] + city_id_options,
    format_func=lambda value: CITY_PLACEHOLDER if value is None else city_display_by_id.get(value, str(value)),
    key="city_geoname_id",
    disabled=selected_region_name == REGION_PLACEHOLDER or not city_id_options,
    help="Choose a city from the selected country and region. The map moves to it automatically.",
)

selected_city_row = None
selected_place = None
if selected_city_id is not None:
    row_matches = city_catalog[city_catalog["geonameid"] == int(selected_city_id)]
    if not row_matches.empty:
        selected_city_row = row_matches.iloc[0]
        try:
            selected_place = geocode_catalog_city(
                str(selected_city_row["name"]),
                str(selected_city_row["country"]),
                str(selected_city_row["region"]),
                int(selected_city_row["geonameid"]),
            )
        except (requests.RequestException, ValueError, KeyError):
            selected_place = None

if selected_city_row is not None:
    st.sidebar.markdown(
        '<div class="air-selected">'
        '<div style="font-size:9px;letter-spacing:1.35px;color:#7A887E;font-weight:750;">SELECTED LOCATION</div>'
        f'<div style="font-size:15px;font-weight:750;color:#17251D;margin-top:4px;">{html.escape(str(selected_city_row["name"]))}</div>'
        f'<div style="font-size:11px;color:#68766D;margin-top:2px;">{html.escape(str(selected_city_row["region"]))}, {html.escape(str(selected_city_row["country"]))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if selected_place is None:
        st.sidebar.caption("The city was selected, but its map coordinates could not be resolved. Check your connection.")
else:
    st.sidebar.caption("Choose a city to zoom into its local air-quality heatmap. With no city selected, the world view is shown.")

st.sidebar.markdown("---")
st.sidebar.markdown("**FORECAST TRENDS**")
forecast_window = st.sidebar.radio(
    "Chart range",
    ["24 hours", "48 hours", "5 days"],
    key="forecast_window",
    label_visibility="collapsed",
)
refresh_requested = st.sidebar.button("Refresh data", use_container_width=True, type="secondary")


# ============================================================
# API DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_batch(points: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    points = tuple((float(lat), float(lon)) for lat, lon in points)
    if not points:
        return pd.DataFrame(columns=["latitude", "longitude", "us_aqi", "pm2_5", "pm10", "time"])
    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
        "current": "us_aqi,pm2_5,pm10",
        "cell_selection": "nearest",
    }
    response = requests.get(AIR_URL, params=params, timeout=40)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Unexpected air-quality API response format.")
    rows = []
    for index, (requested_lat, requested_lon) in enumerate(points):
        item = payload[index] if index < len(payload) and isinstance(payload[index], dict) else {}
        current = item.get("current", {}) or {}
        rows.append({
            "latitude": safe_number(item.get("latitude")) if item.get("latitude") is not None else requested_lat,
            "longitude": safe_number(item.get("longitude")) if item.get("longitude") is not None else requested_lon,
            "requested_latitude": requested_lat,
            "requested_longitude": requested_lon,
            "us_aqi": safe_number(current.get("us_aqi")),
            "pm2_5": safe_number(current.get("pm2_5")),
            "pm10": safe_number(current.get("pm10")),
            "time": current.get("time", ""),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hourly_air(latitude: float, longitude: float, timezone_name: str) -> pd.DataFrame:
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "hourly": "us_aqi,pm2_5,pm10",
        "forecast_days": 5,
        "timezone": timezone_name or TIMEZONE_DEFAULT,
        "cell_selection": "land",
    }
    response = requests.get(AIR_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("Unexpected hourly air-quality response.")
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).reset_index(drop=True)
    for column in ("us_aqi", "pm2_5", "pm10"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(latitude: float, longitude: float, timezone_name: str) -> dict:
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "timezone": timezone_name or TIMEZONE_DEFAULT,
    }
    response = requests.get(WEATHER_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "current" not in payload:
        raise ValueError("Unexpected weather response.")
    return payload["current"]


def safe_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def aqi_category(value: Any) -> str:
    value = safe_number(value)
    if value is None:
        return "No data"
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for sensitive groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very unhealthy"
    return "Hazardous"


def format_value(value: Any, suffix: str = "", decimals: int = 1) -> str:
    number = safe_number(value)
    return "—" if number is None else f"{number:,.{decimals}f}{suffix}"


def generate_global_grid() -> list[tuple[float, float]]:
    # Representative featured-city samples keep the global view lightweight.
    # Interpolated areas are a visual summary, not measurements between cities.
    points = [(float(lat), float(lon)) for _, _, lat, lon in FEATURED_CITIES]
    return list(dict.fromkeys(points))


def generate_local_grid(latitude: float, longitude: float) -> list[tuple[float, float]]:
    # A 9 x 9 local sample. Model resolution varies, so nearby points can map
    # to the same underlying forecast cell; raster smoothing is visualization only.
    offsets = np.linspace(-0.55, 0.55, 9)
    cosine = max(abs(math.cos(math.radians(latitude))), 0.2)
    longitude_span = min(0.55 / cosine, 3.0)
    points = []
    for dlat in offsets:
        for dlon in np.linspace(-longitude_span, longitude_span, 9):
            lat = max(-84.5, min(84.5, latitude + float(dlat)))
            lon = ((longitude + float(dlon) + 180) % 360) - 180
            points.append((round(lat, 4), round(lon, 4)))
    return list(dict.fromkeys(points))


def fetch_chunk(points: tuple[tuple[float, float], ...]) -> tuple[pd.DataFrame, str | None]:
    try:
        return fetch_current_batch(tuple(points)), None
    except (requests.RequestException, ValueError, KeyError) as exc:
        return pd.DataFrame(), str(exc)


def collect_samples(points: list[tuple[float, float]], chunk_size: int = 36) -> tuple[pd.DataFrame, int]:
    chunks = [tuple(points[i:i + chunk_size]) for i in range(0, len(points), chunk_size)]
    if not chunks:
        return pd.DataFrame(), 0
    frames: list[pd.DataFrame] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame, error in executor.map(fetch_chunk, chunks):
            if error or frame.empty:
                failures += 1
            else:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(), failures
    return pd.concat(frames, ignore_index=True), failures


def interpolate_raster(
    frame: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    smoothing_km: float,
) -> np.ndarray | None:
    """Interpolate sparse AQI samples into an RGBA geographic raster using IDW.

    bounds are (west, south, east, north). The raster is a visualization surface,
    not evidence of measurements at every pixel.
    """
    if frame is None or frame.empty:
        return None
    valid = frame.copy()
    for column in ("latitude", "longitude", "us_aqi"):
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=["latitude", "longitude", "us_aqi"])
    if valid.empty:
        return None

    west, south, east, north = bounds
    longitudes = np.linspace(west, east, width, dtype=np.float32)
    latitudes = np.linspace(north, south, height, dtype=np.float32)
    grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
    weighted_values = np.zeros((height, width), dtype=np.float32)
    weight_total = np.zeros((height, width), dtype=np.float32)
    sigma2 = max(float(smoothing_km), 1.0) ** 2

    # Aggregate duplicate model-grid locations before interpolation.
    valid["lat_bin"] = valid["latitude"].round(3)
    valid["lon_bin"] = valid["longitude"].round(3)
    valid = valid.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        us_aqi=("us_aqi", "mean"),
    )

    for sample in valid.itertuples(index=False):
        lat0 = float(sample.latitude)
        lon0 = float(sample.longitude)
        value = float(sample.us_aqi)
        dlat_km = (grid_lat - lat0) * 111.32
        dlon = (grid_lon - lon0 + 180.0) % 360.0 - 180.0
        mean_lat = np.deg2rad((grid_lat + lat0) / 2.0)
        dlon_km = dlon * 111.32 * np.cos(mean_lat)
        distance2 = dlat_km * dlat_km + dlon_km * dlon_km
        weight = 1.0 / (distance2 + sigma2)
        weighted_values += weight * value
        weight_total += weight

    surface = np.divide(
        weighted_values,
        np.maximum(weight_total, 1e-12),
        out=np.zeros_like(weighted_values),
        where=weight_total > 0,
    )
    surface = np.clip(surface, 0, 300)

    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        channel_values = np.interp(surface, HEAT_STOPS, HEAT_RGB[:, channel])
        rgb[:, :, channel] = np.clip(channel_values, 0, 255).astype(np.uint8)

    # Feather transparency around the sample footprint for a soft, geographic
    # heat surface rather than a hard-edged rectangle.
    nearest_distance2 = np.full((height, width), np.inf, dtype=np.float32)
    for sample in valid.itertuples(index=False):
        lat0 = float(sample.latitude)
        lon0 = float(sample.longitude)
        dlat_km = (grid_lat - lat0) * 111.32
        dlon = (grid_lon - lon0 + 180.0) % 360.0 - 180.0
        mean_lat = np.deg2rad((grid_lat + lat0) / 2.0)
        dlon_km = dlon * 111.32 * np.cos(mean_lat)
        distance2 = dlat_km * dlat_km + dlon_km * dlon_km
        nearest_distance2 = np.minimum(nearest_distance2, distance2)
    feather_radius = max(float(smoothing_km) * 2.8, 30.0)
    alpha_float = 190.0 * np.exp(-nearest_distance2 / (2.0 * feather_radius ** 2))
    alpha_float[nearest_distance2 > (feather_radius * 2.2) ** 2] = 0.0
    alpha = np.clip(alpha_float, 0, 190).astype(np.uint8)
    return np.dstack((rgb, alpha))


def build_folium_map(
    frame: pd.DataFrame,
    center: tuple[float, float],
    zoom: int,
    bounds: tuple[float, float, float, float],
    smoothing_km: float,
    raster_width: int,
    raster_height: int,
    height: int = 650,
) -> folium.Map:
    """Always return a fixed-height street basemap; overlay AQI raster when available."""
    fmap = folium.Map(
        location=[float(center[0]), float(center[1])],
        zoom_start=int(zoom),
        width="100%",
        height=f"{int(height)}px",
        tiles=None,
        control_scale=True,
        zoom_control=True,
        scrollWheelZoom=True,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street map",
        control=False,
        show=True,
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
    ).add_to(fmap)

    raster = interpolate_raster(
        frame, bounds=bounds, width=raster_width, height=raster_height,
        smoothing_km=smoothing_km,
    )
    if raster is not None:
        west, south, east, north = bounds
        ImageOverlay(
            image=raster,
            bounds=[[south, west], [north, east]],
            opacity=0.92,
            interactive=False,
            cross_origin=False,
            zindex=350,
            name="Interpolated AQI heat layer",
        ).add_to(fmap)

    legend_html = """
    <div style="position:absolute;z-index:9999;right:16px;bottom:18px;background:rgba(255,255,255,.96);border:1px solid #DCE4DC;border-radius:10px;padding:10px 12px;font:11px Arial,sans-serif;color:#29372E;box-shadow:0 2px 10px rgba(0,0,0,.10);">
      <div style="font-weight:700;margin-bottom:7px;">Modelled US AQI</div>
      <div style="width:190px;height:10px;border-radius:9px;background:linear-gradient(90deg,#3052D2,#2278E8,#26BE70,#9BD744,#F5DE41,#F59937,#E44139,#742A90);"></div>
      <div style="display:flex;justify-content:space-between;margin-top:5px;color:#56645A;"><span>0</span><span>100</span><span>200</span><span>300+</span></div>
    </div>
    """
    fmap.get_root().html.add_child(Element(legend_html))

    map_id = fmap.get_name()
    explicit_size_css = Element(
        f"<style>html,body{{width:100%!important;height:100%!important;margin:0!important;padding:0!important;overflow:hidden!important;}}"
        f"#{map_id}{{width:100%!important;height:{int(height)}px!important;min-height:{int(height)}px!important;position:relative!important;}}"
        f".folium-map{{width:100%!important;height:{int(height)}px!important;min-height:{int(height)}px!important;}}</style>"
    )
    fmap.get_root().header.add_child(explicit_size_css)
    return fmap


def display_map(fmap: folium.Map, height: int = 650) -> None:
    """Display a Folium HTML document in a full-width Streamlit iframe."""
    st.iframe(
        fmap.get_root().render(),
        width="stretch",
        height=int(height),
        alt="Interactive geographic air-quality heatmap",
    )


def make_chart_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=12, r=14, t=24, b=12),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#17251D", size=12),
        hovermode="x unified",
        legend=dict(font=dict(color="#17251D", size=11)),
    )
    fig.update_xaxes(
        gridcolor="#E8ECE8", tickfont=dict(color="#17251D", size=11),
        title_font=dict(color="#17251D", size=12), linecolor="#AAB4AC",
        tickcolor="#78847B",
    )
    fig.update_yaxes(
        gridcolor="#E8ECE8", tickfont=dict(color="#17251D", size=11),
        title_font=dict(color="#17251D", size=12), linecolor="#AAB4AC",
        tickcolor="#78847B",
    )
    return fig


def render_comparison_table(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.info("Global city comparison is temporarily unavailable.")
        return
    view = frame.copy().sort_values("us_aqi", ascending=False, na_position="last")
    rows = []
    for row in view.itertuples(index=False):
        category = aqi_category(row.us_aqi)
        tint = AQI_COLORS[category]
        rows.append(
            "<tr>"
            f"<td style='font-weight:650;color:#17251D'>{html.escape(str(row.city))}</td>"
            f"<td>{html.escape(str(row.country))}</td>"
            f"<td style='font-weight:750;font-variant-numeric:tabular-nums'>{format_value(row.us_aqi, decimals=0)}</td>"
            f"<td>{format_value(row.pm2_5)}</td><td>{format_value(row.pm10)}</td>"
            f"<td><span style='display:inline-block;border:1px solid {tint};color:{tint};border-radius:99px;padding:4px 8px;font-size:10px;white-space:nowrap'>{html.escape(category)}</span></td>"
            "</tr>"
        )
    table = (
        "<div style='overflow-x:auto;border:1px solid #E1E8E1;border-radius:13px;background:#FFFFFF;'>"
        "<table style='width:100%;border-collapse:collapse;background:#FFFFFF;color:#35443A;font:13px Inter,Segoe UI,Arial,sans-serif;'>"
        "<thead><tr style='background:#F4F7F4;color:#647269;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.7px;'>"
        "<th style='padding:12px 14px'>City</th><th style='padding:12px 14px'>Country</th><th style='padding:12px 14px'>US AQI</th>"
        "<th style='padding:12px 14px'>PM2.5</th><th style='padding:12px 14px'>PM10</th><th style='padding:12px 14px'>Category</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)
    st.caption("Modelled forecasts at representative city coordinates; not direct sensor readings.")


# Refresh clears only API caches; the city catalog stays cached for 24 hours.
if refresh_requested:
    fetch_current_batch.clear()
    fetch_hourly_air.clear()
    fetch_weather.clear()
    st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="air-hero">
      <div class="air-eyebrow">Global environmental intelligence</div>
      <div class="air-title">Understand the air around you.</div>
      <div class="air-subtitle">Explore modelled air-quality forecasts on a continuous geographic heat layer, select a city from its country and region, and compare conditions across major cities worldwide.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# WORLD / LOCAL MAP
# ============================================================

st.markdown('<div class="air-section-label">AIR QUALITY MAP</div>', unsafe_allow_html=True)
st.markdown('<div class="air-legend"><div class="air-gradient"></div><span>Lower estimate</span><span>→</span><span>Higher estimate</span></div>', unsafe_allow_html=True)

if selected_city_id is None:
    st.subheader("Global air-quality overview")
    st.caption("A continuous raster-style visualization interpolated from a coarse global sample. Select a city in the sidebar to zoom into its local heat layer.")
    with st.spinner("Loading representative global air-quality forecasts..."):
        global_points = generate_global_grid()
        global_map_frame, global_failures = collect_samples(global_points, chunk_size=36)

    if not global_map_frame.empty:
        global_map_frame = global_map_frame.dropna(subset=["latitude", "longitude", "us_aqi"])
        global_map_frame["lat_bin"] = global_map_frame["latitude"].round(2)
        global_map_frame["lon_bin"] = global_map_frame["longitude"].round(2)
        global_map_frame = global_map_frame.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
            latitude=("latitude", "mean"), longitude=("longitude", "mean"), us_aqi=("us_aqi", "mean"),
            pm2_5=("pm2_5", "mean"), pm10=("pm10", "mean"),
        )
    else:
        global_map_frame = pd.DataFrame()

    world_map = build_folium_map(
        global_map_frame,
        center=(18, 0),
        zoom=2,
        bounds=(-180, -60, 180, 80),
        smoothing_km=1450,
        raster_width=720,
        raster_height=360,
        height=650,
    )
    display_map(world_map, height=650)
    if global_map_frame.empty:
        st.warning("The street basemap is visible, but live AQI samples were unavailable for the heat layer. Check your internet connection and refresh.")
    else:
        st.caption("Global heat regions are interpolated from representative city forecasts. They are a broad visual overview, not measurements at every map pixel.")

else:
    if selected_place is None:
        st.warning("The selected city could not be resolved to map coordinates. The world street basemap is shown until the location service responds.")
        fallback_map = build_folium_map(
            pd.DataFrame(), center=(18, 0), zoom=2,
            bounds=(-180, -60, 180, 80), smoothing_km=1450,
            raster_width=720, raster_height=360, height=650,
        )
        display_map(fallback_map, height=650)
    else:
        place_name = str(selected_place.get("name") or selected_city_row["name"])
        place_region = str(selected_place.get("admin1") or selected_city_row["region"])
        place_country = str(selected_place.get("country") or selected_city_row["country"])
        lat = float(selected_place["latitude"])
        lon = float(selected_place["longitude"])
        st.subheader(f"Air quality around {place_name}")
        st.caption(f"{place_region}, {place_country} · The map automatically centres on the selected city. No city markers are added; the geographic surface is the heat layer.")
        with st.spinner(f"Building the local heat layer around {place_name}..."):
            local_frame, local_failures = collect_samples(generate_local_grid(lat, lon), chunk_size=36)
        local_frame = local_frame.dropna(subset=["latitude", "longitude", "us_aqi"]) if not local_frame.empty else pd.DataFrame()
        half_lat = 0.85
        half_lon = min(0.85 / max(abs(math.cos(math.radians(lat))), 0.2), 4.0)
        local_map = build_folium_map(
            local_frame,
            center=(lat, lon),
            zoom=9,
            bounds=(lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat),
            smoothing_km=22,
            raster_width=640,
            raster_height=540,
            height=650,
        )
        display_map(local_map, height=650)
        if local_frame.empty:
            st.warning("The street basemap is visible, but no usable AQI samples were returned for this city right now.")
        else:
            st.caption("Local heat layer is interpolated from nearby forecast-grid samples. Its spatial detail is limited by the atmospheric model resolution.")


# ============================================================
# SELECTED CITY CONDITIONS + FORECAST TRENDS
# ============================================================

st.markdown("---")
st.markdown('<div class="air-section-label">SELECTED LOCATION</div>', unsafe_allow_html=True)
if selected_city_id is None:
    st.subheader("World overview")
    st.caption("No city is selected. Choose a country, region and city in the sidebar to display local conditions and forecasts.")
else:
    if selected_place is None:
        st.subheader(str(selected_city_row["name"]))
        st.caption(f"{selected_city_row['region']}, {selected_city_row['country']} · Coordinates are currently unavailable.")
    else:
        selected_name = str(selected_place.get("name") or selected_city_row["name"])
        selected_country_for_data = str(selected_place.get("country") or selected_city_row["country"])
        selected_region_for_data = str(selected_place.get("admin1") or selected_city_row["region"])
        st.subheader(selected_name)
        st.caption(f"{selected_region_for_data}, {selected_country_for_data}")

        selected_timezone = str(selected_place.get("timezone") or TIMEZONE_DEFAULT)
        selected_hourly = pd.DataFrame()
        selected_weather: dict = {}
        hourly_error = None
        weather_error = None
        try:
            selected_hourly = fetch_hourly_air(
                float(selected_place["latitude"]),
                float(selected_place["longitude"]),
                selected_timezone,
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            hourly_error = str(exc)
        try:
            selected_weather = fetch_weather(
                float(selected_place["latitude"]),
                float(selected_place["longitude"]),
                selected_timezone,
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            weather_error = str(exc)

        if not selected_hourly.empty:
            local_now = pd.Timestamp.now(tz=selected_timezone).tz_localize(None)
            usable = selected_hourly.dropna(subset=["us_aqi"])
            source = usable if not usable.empty else selected_hourly
            nearest_idx = (source["time"] - local_now).abs().idxmin()
            current = source.loc[nearest_idx]
            current_aqi = safe_number(current.get("us_aqi"))
            category = aqi_category(current_aqi)
            st.markdown(
                f"""
                <div class="air-status" style="border-left-color:{AQI_COLORS[category]};">
                  <div class="air-status-label">Nearest hourly model forecast · US AQI</div>
                  <div class="air-status-title" style="color:{AQI_COLORS[category]};">{html.escape(category)}{' · AQI ' + str(round(current_aqi)) if current_aqi is not None else ''}</div>
                  <div class="air-muted">{html.escape(AQI_DESCRIPTIONS[category])}</div>
                  <div class="air-muted" style="margin-top:4px;">Forecast hour: {html.escape(pd.Timestamp(current['time']).strftime('%d %b %Y, %I:%M %p'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Modelled US AQI", format_value(current_aqi, decimals=0))
            with metric_cols[1]:
                st.metric("PM2.5", format_value(current.get("pm2_5"), " µg/m³"))
            with metric_cols[2]:
                st.metric("PM10", format_value(current.get("pm10"), " µg/m³"))
            with metric_cols[3]:
                st.metric("Temperature", format_value(selected_weather.get("temperature_2m"), " °C"))

            weather_cols = st.columns(3)
            with weather_cols[0]:
                st.metric("Relative humidity", format_value(selected_weather.get("relative_humidity_2m"), " %"))
            with weather_cols[1]:
                st.metric("Wind speed", format_value(selected_weather.get("wind_speed_10m"), " km/h"))
            with weather_cols[2]:
                st.metric("Feels-like temperature", format_value(selected_weather.get("apparent_temperature"), " °C"))

            st.markdown("### Forecast trends")
            st.caption("Hourly model estimates in the selected location's time zone.")
            future = selected_hourly[selected_hourly["time"] >= local_now.floor("h")].copy()
            if future.empty:
                future = selected_hourly.copy()
            hours = {"24 hours": 24, "48 hours": 48, "5 days": 120}[forecast_window]
            chart_frame = future.head(hours)

            aqi_fig = go.Figure()
            aqi_fig.add_trace(go.Scatter(
                x=chart_frame["time"], y=chart_frame["us_aqi"], mode="lines",
                line={"color": "#28734A", "width": 2.7},
                fill="tozeroy", fillcolor="rgba(40,115,74,.08)",
                connectgaps=False, name="US AQI",
                hovertemplate="%{x|%d %b, %H:%M}<br>US AQI: %{y:.0f}<extra></extra>",
            ))
            aqi_fig.update_layout(showlegend=False)
            aqi_fig.update_yaxes(title_text="US AQI")
            make_chart_layout(aqi_fig)

            pm_fig = go.Figure()
            pm_fig.add_trace(go.Scatter(
                x=chart_frame["time"], y=chart_frame["pm2_5"], mode="lines", name="PM2.5",
                line={"color": "#28734A", "width": 2.5}, connectgaps=False,
            ))
            pm_fig.add_trace(go.Scatter(
                x=chart_frame["time"], y=chart_frame["pm10"], mode="lines", name="PM10",
                line={"color": "#6498C9", "width": 2.5}, connectgaps=False,
            ))
            pm_fig.update_layout(legend={"orientation": "h", "y": 1.08, "x": 0})
            pm_fig.update_yaxes(title_text="Concentration (µg/m³)")
            make_chart_layout(pm_fig)

            left_chart, right_chart = st.columns(2, gap="large")
            with left_chart:
                st.markdown("##### US AQI trend")
                st.plotly_chart(aqi_fig, use_container_width=True, config={"displaylogo": False})
            with right_chart:
                st.markdown("##### Particulate matter")
                st.plotly_chart(pm_fig, use_container_width=True, config={"displaylogo": False})
        else:
            st.info("The city was selected, but hourly air-quality forecast data is temporarily unavailable.")
            if hourly_error:
                with st.expander("Technical details"):
                    st.code(hourly_error)
        if weather_error:
            st.caption("Weather data is temporarily unavailable for this location.")

# ============================================================
# GLOBAL MAJOR-CITY COMPARISON
# ============================================================

st.markdown("---")
st.markdown('<div class="air-section-label">WORLD CITY COMPARISON</div>', unsafe_allow_html=True)
st.subheader("Air quality in major cities")
st.caption("A weather-app-style comparison of selected major cities across continents. Values are modelled estimates and the list is a representative selection, not every city worldwide.")

with st.spinner("Loading comparison city forecasts..."):
    comparison_points = [(lat, lon) for _, _, lat, lon in FEATURED_CITIES]
    comparison_frame, comparison_failures = collect_samples(comparison_points, chunk_size=36)

if not comparison_frame.empty:
    count = min(len(comparison_frame), len(FEATURED_CITIES))
    comparison_frame = comparison_frame.iloc[:count].copy()
    comparison_frame["city"] = [item[0] for item in FEATURED_CITIES[:count]]
    comparison_frame["country"] = [item[1] for item in FEATURED_CITIES[:count]]
    render_comparison_table(comparison_frame[["city", "country", "us_aqi", "pm2_5", "pm10"]])
else:
    st.info("Global city comparison is temporarily unavailable. Try Refresh data.")

# ============================================================
# DEFINITIONS + ATTRIBUTION
# ============================================================

st.markdown("---")
st.subheader("Understand the data")
info_a, info_b = st.columns(2, gap="large")
with info_a:
    st.markdown(
        "<div class='air-info'><strong>PM2.5</strong><div class='air-muted' style='margin-top:8px;'>Fine airborne particles with diameters of 2.5 micrometres or less. Concentration is expressed in micrograms per cubic metre (µg/m³).</div></div>",
        unsafe_allow_html=True,
    )
with info_b:
    st.markdown(
        "<div class='air-info'><strong>US AQI</strong><div class='air-muted' style='margin-top:8px;'>The US Air Quality Index is a communication scale. Higher values indicate greater concern on that scale; other countries may use different official indexes.</div></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="air-muted" style="margin-top:20px;">
      <strong>Important limitations:</strong> AirAware uses atmospheric-model forecasts, not verified local sensor readings. The map is a smooth interpolation between sampled forecast values; it does not imply that each pixel is measured. Spatial detail depends on the underlying atmospheric model. Do not use this prototype as an official warning or medical assessment.<br><br>
      <strong>Air quality and weather:</strong> <a href="https://open-meteo.com/" target="_blank">Open-Meteo</a> and
      <a href="https://atmosphere.copernicus.eu/" target="_blank">Copernicus Atmosphere Monitoring Service</a>.<br>
      <strong>City directory:</strong> <a href="https://github.com/datasets/world-cities" target="_blank">DataHub world-cities dataset</a>, sourced from GeoNames. Includes major cities generally above 15,000 residents plus capitals; licensed CC BY 4.0.<br>
      <strong>Map tiles:</strong> <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap contributors</a>.<br><br>
      AirAware · Global environmental intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
