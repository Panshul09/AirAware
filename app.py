from __future__ import annotations

import html
import io
import math
import time
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
# AIR AWARE — GLOBAL ENVIRONMENTAL INTELLIGENCE | RELEASE CANDIDATE
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

# Additional map-only sampling locations. These improve geographic coverage
# without making the city-comparison table unnecessarily long.
GLOBAL_MAP_EXTRA_POINTS = [
    ("Seattle", "United States", 47.6062, -122.3321),
    ("Chicago", "United States", 41.8781, -87.6298),
    ("Boston", "United States", 42.3601, -71.0589),
    ("Miami", "United States", 25.7617, -80.1918),
    ("Houston", "United States", 29.7604, -95.3698),
    ("Montreal", "Canada", 45.5019, -73.5674),
    ("Calgary", "Canada", 51.0447, -114.0719),
    ("Santiago", "Chile", -33.4489, -70.6693),
    ("Medellin", "Colombia", 6.2442, -75.5812),
    ("Lisbon", "Portugal", 38.7223, -9.1393),
    ("Amsterdam", "Netherlands", 52.3676, 4.9041),
    ("Warsaw", "Poland", 52.2297, 21.0122),
    ("Athens", "Greece", 37.9838, 23.7275),
    ("Casablanca", "Morocco", 33.5731, -7.5898),
    ("Accra", "Ghana", 5.6037, -0.1870),
    ("Addis Ababa", "Ethiopia", 8.9806, 38.7578),
    ("Tehran", "Iran", 35.6892, 51.3890),
    ("Kathmandu", "Nepal", 27.7172, 85.3240),
    ("Hanoi", "Vietnam", 21.0278, 105.8342),
    ("Manila", "Philippines", 14.5995, 120.9842),
    ("Taipei", "Taiwan", 25.0330, 121.5654),
    ("Osaka", "Japan", 34.6937, 135.5023),
    ("Perth", "Australia", -31.9505, 115.8605),
    ("Brisbane", "Australia", -27.4698, 153.0251),
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
    :root, html, body { color-scheme: light !important; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        color-scheme: light !important;
        background: #F6F8F6 !important;
        color: #17251D !important;
        font-family: Inter, "Segoe UI", Arial, sans-serif !important;
    }
    [data-testid="stHeader"] { background: #F6F8F6 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background: #FFFFFF !important;
        border-right: 1px solid #E2E8E2 !important;
    }
    [data-testid="stSidebar"] .block-container { padding: 1.25rem .95rem 2rem !important; }
    h1, h2, h3, h4 { color: #17251D !important; letter-spacing: -.025em !important; }
    p, label, [data-testid="stWidgetLabel"] p { color: #34443A !important; }
    [data-testid="stCaptionContainer"] { color: #68766D !important; }

    /* Style only real Streamlit action buttons. Do not style every button on
       the page: BaseWeb uses button-like elements inside select widgets. */
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button,
    .stApp [data-testid="stFormSubmitButton"] button {
        appearance: none !important;
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        background-image: none !important;
        color: #111B14 !important;
        -webkit-text-fill-color: #111B14 !important;
        border: 1px solid #CDD8CE !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(20,35,24,.05) !important;
        opacity: 1 !important;
        font-weight: 650 !important;
        min-height: 40px !important;
    }
    .stApp [data-testid="stButton"] button p,
    .stApp [data-testid="stButton"] button span,
    .stApp [data-testid="stDownloadButton"] button p,
    .stApp [data-testid="stDownloadButton"] button span,
    .stApp [data-testid="stFormSubmitButton"] button p,
    .stApp [data-testid="stFormSubmitButton"] button span {
        color: #111B14 !important;
        -webkit-text-fill-color: #111B14 !important;
    }
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover,
    .stApp [data-testid="stFormSubmitButton"] button:hover {
        background: #F1F4F1 !important;
        background-color: #F1F4F1 !important;
        border-color: #9DAD9F !important;
        color: #101A13 !important;
        -webkit-text-fill-color: #101A13 !important;
    }
    .stApp [data-testid="stButton"] button:disabled,
    .stApp [data-testid="stDownloadButton"] button:disabled {
        background: #F5F7F5 !important;
        color: #7B867D !important;
        -webkit-text-fill-color: #7B867D !important;
    }

    /* Light select controls. Keep the inner arrow area transparent and style
       its icon as a small grey chevron instead of a dark square. */
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #17251D !important;
        border-color: #D1DBD2 !important;
        box-shadow: none !important;
    }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"] * {
        color: #17251D !important;
        -webkit-text-fill-color: #17251D !important;
    }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        color: #68766D !important;
        min-height: 0 !important;
        padding: 0 !important;
    }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] svg {
        color: #68766D !important;
        fill: #68766D !important;
        stroke: #68766D !important;
        background: transparent !important;
    }
    .stApp [data-testid="stSelectbox"] [data-baseweb="select"] input {
        color: #17251D !important;
        -webkit-text-fill-color: #17251D !important;
        background: transparent !important;
    }
    .stApp [data-testid="stSelectbox"] [aria-disabled="true"] {
        background: #F4F6F4 !important;
        color: #879188 !important;
        -webkit-text-fill-color: #879188 !important;
    }
    /* The select popup is rendered outside the sidebar DOM, so target it globally. */
    body div[data-baseweb="popover"],
    body div[data-baseweb="menu"],
    body ul[role="listbox"],
    body div[role="listbox"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        color: #17251D !important;
        border: 1px solid #DCE5DD !important;
    }
    body div[data-baseweb="popover"] *,
    body div[data-baseweb="menu"] *,
    body [role="listbox"] * {
        color: #17251D !important;
        -webkit-text-fill-color: #17251D !important;
    }
    body [role="option"]:hover,
    body [role="option"][aria-selected="true"] {
        background: #EDF3EE !important;
        background-color: #EDF3EE !important;
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


    /* Subtle motion: page entrance, cards, charts and controls. */
    @keyframes airFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes airSoftFade {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    .block-container { animation: airFadeUp .48s cubic-bezier(.2,.7,.2,1) both; }
    .air-hero { animation: airFadeUp .55s cubic-bezier(.2,.7,.2,1) both; }
    .air-status, .air-info, .air-selected { animation: airSoftFade .55s ease-out both; }
    [data-testid="stMetric"] {
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(24, 43, 29, .07) !important;
        border-color: #C8D8CA !important;
    }
    .stApp [data-testid="stButton"] button,
    .stApp [data-testid="stDownloadButton"] button,
    .stApp [data-testid="stFormSubmitButton"] button {
        transition: background-color .16s ease, border-color .16s ease,
                    box-shadow .16s ease, transform .16s ease !important;
    }
    .stApp [data-testid="stButton"] button:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(24, 43, 29, .07) !important;
    }
    .air-selected { transition: border-color .18s ease, box-shadow .18s ease; }
    .air-selected:hover { border-color: #C8D8CA; box-shadow: 0 4px 12px rgba(24,43,29,.045); }
    [data-testid="stPlotlyChart"] { animation: airSoftFade .65s ease-out both; }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
            scroll-behavior: auto !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CITY CATALOG: WORLD CITIES WITH COUNTRY + FIRST-LEVEL REGION
# Dataset: GeoNames, published as the DataHub world-cities dataset.
# Includes cities generally above 15,000 residents plus capitals.
# ============================================================

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def request_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    attempts: int = 3,
) -> requests.Response:
    """GET with bounded retries for transient network and server failures."""
    attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=(8, int(timeout)))
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise
            time.sleep(min(0.5 * (2 ** attempt), 2.0))
            continue

        if response.status_code in RETRYABLE_HTTP_STATUS and attempt < attempts - 1:
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = 0.5 * (2 ** attempt)
            time.sleep(min(max(delay, 0.25), 2.5))
            continue

        response.raise_for_status()
        return response

    if last_error is not None:
        raise last_error
    raise requests.RequestException("The request could not be completed after retries.")


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
    attempts: int = 3,
) -> Any:
    """Request JSON and turn API-level error objects into readable exceptions."""
    response = request_with_retry(url, params=params, timeout=timeout, attempts=attempts)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("The data service returned an invalid JSON response.") from exc
    if isinstance(payload, dict) and payload.get("error"):
        message = payload.get("reason") or payload.get("message") or "The data service reported an error."
        raise ValueError(str(message))
    return payload


def safe_timezone_name(value: Any) -> str:
    """Return an API timezone that pandas can use, falling back to UTC."""
    candidate = str(value or TIMEZONE_DEFAULT).strip() or TIMEZONE_DEFAULT
    try:
        pd.Timestamp.now(tz=candidate)
        return candidate
    except (TypeError, ValueError, KeyError):
        return TIMEZONE_DEFAULT


@st.cache_data(ttl=86400, show_spinner=False)
def load_city_catalog() -> pd.DataFrame:
    catalog_response = request_with_retry(CITY_DATA_URL, timeout=35)
    catalog = pd.read_csv(io.StringIO(catalog_response.text), keep_default_na=False)
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
    payload = request_json(
        GEOCODING_URL,
        params={"name": city_name, "count": 100, "language": "en", "countryCode": country_code},
        timeout=25,
    )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    results = results or []
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


def reset_region_and_city_day4() -> None:
    st.session_state["region_name_day4"] = None
    st.session_state["city_geoname_id_day4"] = None


def reset_city_only_day4() -> None:
    st.session_state["city_geoname_id_day4"] = None


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

# New state keys intentionally prevent a previous session's Pakistan/Hyderabad
# selection from carrying over. The initial experience opens on the world map.
if "country_name_day4" not in st.session_state or st.session_state["country_name_day4"] not in country_options:
    st.session_state["country_name_day4"] = None
if "region_name_day4" not in st.session_state:
    st.session_state["region_name_day4"] = None
if "city_geoname_id_day4" not in st.session_state:
    st.session_state["city_geoname_id_day4"] = None
# Clean, compact sidebar: three dependent selectors and forecast range.
st.sidebar.markdown(
    '<div class="air-brand">🌍 AirAware</div>'
    '<div class="air-brand-sub">Global environmental intelligence</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**LOCATION**")

selected_country_name = st.sidebar.selectbox(
    "Country",
    options=country_options,
    index=None,
    placeholder="Select a country",
    key="country_name_day4",
    on_change=reset_region_and_city_day4,
)

if selected_country_name is None:
    country_rows = city_catalog.iloc[0:0].copy()
    region_options_current: list[str] = []
else:
    country_rows = city_catalog[city_catalog["country"] == selected_country_name]
    region_options_current = sorted(country_rows["region"].dropna().unique().tolist(), key=str.casefold)

if st.session_state.get("region_name_day4") not in region_options_current:
    st.session_state["region_name_day4"] = None

selected_region_name = st.sidebar.selectbox(
    "State / province / region",
    options=region_options_current,
    index=None,
    placeholder="Select a state / region",
    key="region_name_day4",
    on_change=reset_city_only_day4,
    disabled=selected_country_name is None or not region_options_current,
)

if selected_region_name is None:
    region_city_rows = country_rows.iloc[0:0].copy()
else:
    region_city_rows = country_rows[country_rows["region"] == selected_region_name].copy()

region_city_rows = region_city_rows.sort_values("name", key=lambda col: col.str.casefold(), kind="stable")
city_id_options = [int(value) for value in region_city_rows["geonameid"].tolist()]
city_display_by_id: dict[int, str] = {}
name_counts = region_city_rows["name"].value_counts().to_dict()
for _, row in region_city_rows.iterrows():
    city_name = str(row["name"])
    city_display_by_id[int(row["geonameid"])] = (
        f"{city_name} · place {int(row['geonameid'])}" if name_counts.get(city_name, 0) > 1 else city_name
    )

if st.session_state.get("city_geoname_id_day4") not in city_id_options:
    st.session_state["city_geoname_id_day4"] = None

selected_city_id = st.sidebar.selectbox(
    "City",
    options=city_id_options,
    index=None,
    placeholder="Select a city",
    format_func=lambda value: city_display_by_id.get(value, str(value)),
    key="city_geoname_id_day4",
    disabled=selected_region_name is None or not city_id_options,
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
    region_label = str(selected_city_row["region"])
    if region_label == NO_REGION:
        region_label = "Region not listed"
    st.sidebar.markdown(
        '<div class="air-selected">'
        '<div style="font-size:9px;letter-spacing:1.35px;color:#7A887E;font-weight:750;">SELECTED LOCATION</div>'
        f'<div style="font-size:15px;font-weight:750;color:#17251D;margin-top:4px;">{html.escape(str(selected_city_row["name"]))}</div>'
        f'<div style="font-size:11px;color:#68766D;margin-top:2px;">{html.escape(region_label)}, {html.escape(str(selected_city_row["country"]))}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if selected_place is None:
        st.sidebar.caption("The city was selected, but its map coordinates could not be resolved. Check your connection.")
else:
    st.sidebar.caption("Choose a country, region and city to open a local heatmap. With no location selected, AirAware opens on the world view.")

st.sidebar.markdown("---")
st.sidebar.caption("Hourly and daily forecasts are both available on the main page.")
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
    payload = request_json(AIR_URL, params=params, timeout=40)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Unexpected air-quality API response format.")
    if len(payload) != len(points):
        raise ValueError(
            "The multi-location API response did not contain one result per requested coordinate. "
            "This batch has been marked unavailable to avoid mismatching city data."
        )
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
        "timezone": safe_timezone_name(timezone_name),
        "cell_selection": "land",
    }
    payload = request_json(AIR_URL, params=params, timeout=30)
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("Unexpected hourly air-quality response.")
    frame = pd.DataFrame(payload["hourly"])
    if "time" not in frame.columns:
        raise ValueError("The air-quality response did not include timestamps.")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time", kind="stable")
    frame = frame.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    for column in ("us_aqi", "pm2_5", "pm10"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_historical_air_quality(
    latitude: float,
    longitude: float,
    timezone_name: str,
    past_days: int,
) -> pd.DataFrame:
    """Fetch archived modelled air-quality data for the previous 7–90 days.

    Open-Meteo's Air Quality API past_days option returns archived model output.
    These values are not equivalent to independent ground-monitor observations.
    """
    days = int(past_days)
    if days not in (7, 30, 60, 90):
        raise ValueError("Choose a supported historical period: 7, 30, 60 or 90 days.")

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "hourly": "us_aqi,pm2_5,pm10",
        "past_days": days,
        "forecast_days": 0,
        "timezone": safe_timezone_name(timezone_name),
        "cell_selection": "land",
    }
    payload = request_json(AIR_URL, params=params, timeout=45)
    if "hourly" not in payload or "time" not in payload["hourly"]:
        message = payload.get("reason") or payload.get("error") or "Unexpected historical air-quality response."
        raise ValueError(str(message))

    frame = pd.DataFrame(payload["hourly"])
    if "time" not in frame.columns:
        raise ValueError("The historical response did not include timestamps.")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time", kind="stable")
    for column in ("us_aqi", "pm2_5", "pm10"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return frame


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hourly_weather(latitude: float, longitude: float, timezone_name: str) -> pd.DataFrame:
    """Fetch matching hourly weather forecasts for the relationship explorer."""
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "forecast_days": 5,
        "timezone": safe_timezone_name(timezone_name),
    }
    payload = request_json(WEATHER_URL, params=params, timeout=30)
    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("Unexpected hourly weather response.")
    frame = pd.DataFrame(payload["hourly"])
    if "time" not in frame.columns:
        raise ValueError("The weather response did not include timestamps.")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"]).sort_values("time", kind="stable")
    frame = frame.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    for column in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(latitude: float, longitude: float, timezone_name: str) -> dict:
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "timezone": safe_timezone_name(timezone_name),
    }
    payload = request_json(WEATHER_URL, params=params, timeout=30)
    if "current" not in payload:
        raise ValueError("Unexpected weather response.")
    return payload["current"]


def safe_number(value: Any) -> float | None:
    """Convert a scalar to a finite float; treat missing and non-finite values as unavailable."""
    try:
        if value is None:
            return None
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
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


def aggregate_daily_forecast(frame: pd.DataFrame, today: Any = None) -> pd.DataFrame:
    """Aggregate hourly model estimates into local calendar-day summaries.

    Open-Meteo's air-quality endpoint returns hourly pollutant/AQI forecast
    variables. Daily values shown by AirAware are calculated from those hourly
    values; they are not a separate native daily air-quality API product.
    """
    required = {"time", "us_aqi", "pm2_5", "pm10"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=[
                "forecast_date", "day_label", "mean_us_aqi", "min_us_aqi",
                "max_us_aqi", "mean_pm2_5", "mean_pm10", "aqi_hours",
                "available_hours", "aqi_category",
            ]
        )

    working = frame.copy()
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    for column in ("us_aqi", "pm2_5", "pm10"):
        working[column] = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=["time"]).sort_values("time")
    working = working.drop_duplicates(subset=["time"], keep="last")
    if working.empty:
        return pd.DataFrame()

    working["forecast_date"] = working["time"].dt.date
    daily = (
        working.groupby("forecast_date", as_index=False, sort=True)
        .agg(
            mean_us_aqi=("us_aqi", "mean"),
            min_us_aqi=("us_aqi", "min"),
            max_us_aqi=("us_aqi", "max"),
            mean_pm2_5=("pm2_5", "mean"),
            mean_pm10=("pm10", "mean"),
            aqi_hours=("us_aqi", "count"),
            available_hours=("time", "count"),
        )
    )
    today_date = today if today is not None else pd.Timestamp.now().date()
    daily["day_label"] = daily["forecast_date"].apply(
        lambda day: f"Today · {pd.Timestamp(day).strftime('%d %b')}"
        if day == today_date else pd.Timestamp(day).strftime("%a · %d %b")
    )
    daily["aqi_category"] = daily["mean_us_aqi"].apply(aqi_category)
    return daily.head(5).reset_index(drop=True)


def summarize_historical_air_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate archived AQI model output into local calendar-day summaries."""
    columns = ["date", "mean_us_aqi", "min_us_aqi", "max_us_aqi", "mean_pm2_5", "mean_pm10", "aqi_records", "total_records"]
    if frame is None or frame.empty or "time" not in frame.columns:
        return pd.DataFrame(columns=columns)

    working = frame.copy()
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    working = working.dropna(subset=["time"]).sort_values("time", kind="stable")
    working = working.drop_duplicates(subset=["time"], keep="last")
    for column in ("us_aqi", "pm2_5", "pm10"):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    working["date"] = working["time"].dt.date
    summary = (
        working.groupby("date", as_index=False, sort=True)
        .agg(
            mean_us_aqi=("us_aqi", "mean"),
            min_us_aqi=("us_aqi", "min"),
            max_us_aqi=("us_aqi", "max"),
            mean_pm2_5=("pm2_5", "mean"),
            mean_pm10=("pm10", "mean"),
            aqi_records=("us_aqi", "count"),
            total_records=("time", "count"),
        )
    )
    summary["aqi_category"] = summary["mean_us_aqi"].apply(aqi_category)
    summary["mean_7d_aqi"] = summary["mean_us_aqi"].rolling(7, min_periods=3).mean()
    return summary[columns + ["aqi_category", "mean_7d_aqi"]]


def infer_sample_interval(frame: pd.DataFrame) -> str:
    """Describe the typical archived-data cadence without assuming hourly data."""
    if frame is None or frame.empty or "time" not in frame.columns:
        return "Unavailable"
    times = pd.to_datetime(frame["time"], errors="coerce").dropna().drop_duplicates().sort_values()
    deltas = times.diff().dropna()
    if deltas.empty:
        return "Not enough timestamps"
    median_minutes = float(deltas.dt.total_seconds().median() / 60.0)
    if median_minutes <= 0:
        return "Irregular"
    if median_minutes < 90:
        return f"About {median_minutes:.0f} minutes"
    if median_minutes % 60 == 0:
        return f"About {median_minutes / 60:.0f} hours"
    return f"About {median_minutes:.0f} minutes"


def detect_elevated_aqi_episodes(
    frame: pd.DataFrame,
    threshold: float = 101.0,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, str]:
    """Find runs of elevated archived AQI samples without bridging time gaps.

    The detector uses the modal positive timestamp interval in the supplied
    archive (which may be hourly or multi-hourly depending on model coverage).
    It reports sample runs, not official pollution alerts or observed exposure.
    """
    columns = [
        "start_time", "end_time", "samples", "sample_interval",
        "observed_span_hours", "mean_us_aqi", "peak_us_aqi",
        "peak_pm2_5", "peak_pm10",
    ]
    if frame is None or frame.empty or not {"time", "us_aqi"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns), "Unavailable"

    working = frame.copy()
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    working["us_aqi"] = pd.to_numeric(working["us_aqi"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for column in ("pm2_5", "pm10"):
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=["time", "us_aqi"]).sort_values("time", kind="stable")
    working = working.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    working = working[(working["us_aqi"] >= 0) & (working["us_aqi"] <= 500)].copy()
    if len(working) < int(min_samples):
        return pd.DataFrame(columns=columns), "Not enough valid samples"

    deltas = working["time"].diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.DataFrame(columns=columns), "Not enough distinct timestamps"
    modal_interval = deltas.value_counts().index[0]
    cadence_label = (
        f"{modal_interval.total_seconds() / 3600:g} hour(s)"
        if modal_interval.total_seconds() % 3600 == 0
        else f"{modal_interval.total_seconds() / 60:g} minute(s)"
    )

    episodes: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []

    def close_episode(rows: list[dict[str, Any]]) -> None:
        if len(rows) < int(min_samples):
            return
        segment = pd.DataFrame(rows)
        start = pd.Timestamp(segment["time"].iloc[0])
        end = pd.Timestamp(segment["time"].iloc[-1])
        episodes.append({
            "start_time": start,
            "end_time": end,
            "samples": int(len(segment)),
            "sample_interval": cadence_label,
            "observed_span_hours": float((end - start).total_seconds() / 3600.0),
            "mean_us_aqi": float(segment["us_aqi"].mean()),
            "peak_us_aqi": float(segment["us_aqi"].max()),
            "peak_pm2_5": safe_number(segment["pm2_5"].max()),
            "peak_pm10": safe_number(segment["pm10"].max()),
        })

    previous_time = None
    for row in working.to_dict(orient="records"):
        timestamp = pd.Timestamp(row["time"])
        value = float(row["us_aqi"])
        is_elevated = value >= float(threshold)
        is_contiguous = previous_time is not None and timestamp - previous_time == modal_interval
        if not is_elevated:
            close_episode(current_rows)
            current_rows = []
        elif current_rows and not is_contiguous:
            close_episode(current_rows)
            current_rows = [row]
        elif is_elevated:
            current_rows.append(row)
        previous_time = timestamp

    close_episode(current_rows)
    result = pd.DataFrame(episodes, columns=columns)
    if not result.empty:
        result = result.sort_values(["start_time", "peak_us_aqi"], ascending=[True, False]).reset_index(drop=True)
    return result, cadence_label


def build_historical_analysis_note(
    *,
    city: str,
    region: str,
    country: str,
    period_days: int,
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    threshold: float,
    minimum_samples: int,
    cadence_label: str,
) -> str:
    """Create a small reproducible Markdown note for a historical analysis."""
    clean = frame.copy() if frame is not None else pd.DataFrame()
    if not clean.empty:
        clean["time"] = pd.to_datetime(clean["time"], errors="coerce")
        if "us_aqi" not in clean.columns:
            clean["us_aqi"] = np.nan
        clean["us_aqi"] = pd.to_numeric(clean["us_aqi"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        clean = clean.dropna(subset=["time"]).sort_values("time")
    valid_aqi = clean["us_aqi"].dropna() if "us_aqi" in clean else pd.Series(dtype=float)
    start = clean["time"].min() if not clean.empty else None
    end = clean["time"].max() if not clean.empty else None
    aqi_coverage = float(valid_aqi.notna().sum() / len(clean) * 100.0) if len(clean) else 0.0
    mean_aqi = float(valid_aqi.mean()) if not valid_aqi.empty else None
    peak_aqi = float(valid_aqi.max()) if not valid_aqi.empty else None
    lines = [
        "# AirAware historical analysis note",
        "",
        f"- Location: {city}, {region}, {country}",
        f"- Requested period: previous {period_days} days",
        f"- Returned local-time range: {start if start is not None else 'unavailable'} to {end if end is not None else 'unavailable'}",
        f"- Returned records: {len(clean)}",
        f"- AQI records: {len(valid_aqi)}",
        f"- AQI availability: {aqi_coverage:.1f}%",
        f"- Typical sample cadence: {cadence_label}",
        f"- Mean modelled US AQI: {mean_aqi:.1f}" if mean_aqi is not None else "- Mean modelled US AQI: unavailable",
        f"- Peak modelled US AQI: {peak_aqi:.1f}" if peak_aqi is not None else "- Peak modelled US AQI: unavailable",
        "",
        "## Elevated-sample run method",
        f"- Threshold: US AQI >= {threshold:.0f}",
        f"- Minimum consecutive samples: {minimum_samples}",
        f"- Detected runs: {len(episodes)}",
        "",
        "## Interpretation and limitations",
        "This analysis uses archived atmospheric-model output provided by Open-Meteo's Air Quality API. It is not a set of independent ground-monitor observations.",
        "The detector groups consecutive available samples using the modal timestamp interval in this returned dataset. It breaks a run at missing or irregular intervals.",
        "Detected runs are exploratory statistical summaries, not official pollution alerts, verified exposure periods, or medical guidance.",
        "For predictive-model validation, compare predictions against independent observations or a clearly defined benchmark dataset, not the same archived model output used as features.",
        "",
        "## Data source",
        "Open-Meteo Air Quality API: https://open-meteo.com/en/docs/air-quality-api",
        "",
    ]
    if not episodes.empty:
        lines.extend(["## Detected runs", "", "| Start | End | Samples | Mean AQI | Peak AQI |", "|---|---|---:|---:|---:|"])
        for item in episodes.itertuples(index=False):
            lines.append(
                f"| {item.start_time} | {item.end_time} | {item.samples} | {item.mean_us_aqi:.1f} | {item.peak_us_aqi:.1f} |"
            )
        lines.append("")
    else:
        lines.extend(["No runs met the selected threshold and minimum-sample settings.", ""])
    return "\n".join(lines)


def audit_forecast_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarise completeness and timestamp consistency without changing data."""
    if frame is None or frame.empty or "time" not in frame.columns:
        return {
            "rows": 0, "unique_timestamps": 0, "duplicate_timestamps": 0,
            "missing_hour_intervals": 0, "irregular_intervals": 0,
            "continuity_pct": 0.0, "aqi_coverage": 0.0,
            "pm25_coverage": 0.0, "pm10_coverage": 0.0,
            "invalid_values": 0, "valid_transitions": 0,
        }

    working = frame.copy()
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    working = working.dropna(subset=["time"])
    rows = int(len(working))
    duplicate_timestamps = int(working.duplicated(subset=["time"], keep="first").sum())
    unique_times = pd.Series(working["time"].drop_duplicates().sort_values().tolist())
    intervals = unique_times.diff().dropna()
    exact_hourly = intervals.eq(pd.Timedelta(hours=1))
    missing_hour_intervals = 0
    irregular_intervals = 0
    for delta in intervals:
        if delta == pd.Timedelta(hours=1):
            continue
        irregular_intervals += 1
        if delta > pd.Timedelta(hours=1):
            approximate_hours = int(round(delta / pd.Timedelta(hours=1)))
            missing_hour_intervals += max(approximate_hours - 1, 0)

    def numeric_series(column: str) -> pd.Series:
        if column not in working.columns:
            return pd.Series(np.nan, index=working.index, dtype=float)
        return pd.to_numeric(working[column], errors="coerce")

    def coverage(column: str) -> float:
        if rows == 0 or column not in working.columns:
            return 0.0
        numeric = numeric_series(column)
        finite = numeric.notna() & np.isfinite(numeric)
        return float(finite.sum() / rows * 100.0)

    invalid_values = 0
    if "us_aqi" in working.columns:
        aqi_values = numeric_series("us_aqi")
        finite_aqi = aqi_values.notna() & np.isfinite(aqi_values)
        invalid_values += int(np.isinf(aqi_values).sum())
        invalid_values += int((finite_aqi & ((aqi_values < 0) | (aqi_values > 500))).sum())
    for column in ("pm2_5", "pm10"):
        if column in working.columns:
            pollutant_values = numeric_series(column)
            finite_pollutant = pollutant_values.notna() & np.isfinite(pollutant_values)
            invalid_values += int(np.isinf(pollutant_values).sum())
            invalid_values += int((finite_pollutant & (pollutant_values < 0)).sum())

    continuity_pct = float(exact_hourly.mean() * 100.0) if len(intervals) else 0.0
    return {
        "rows": rows,
        "unique_timestamps": int(len(unique_times)),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_hour_intervals": int(missing_hour_intervals),
        "irregular_intervals": int(irregular_intervals),
        "continuity_pct": continuity_pct,
        "aqi_coverage": coverage("us_aqi"),
        "pm25_coverage": coverage("pm2_5"),
        "pm10_coverage": coverage("pm10"),
        "invalid_values": invalid_values,
        "valid_transitions": int(exact_hourly.sum()),
    }


def consecutive_hour_changes(frame: pd.DataFrame, value_column: str) -> pd.Series:
    """Return row-aligned changes only across exact one-hour intervals.

    Results are mapped by row position, not index label, so duplicate or
    non-monotonic DataFrame indexes cannot attach a change to the wrong row.
    The earlier row of a duplicate timestamp is left unavailable.
    """
    if frame is None or frame.empty or not {"time", value_column}.issubset(frame.columns):
        index = getattr(frame, "index", pd.RangeIndex(0))
        return pd.Series(np.nan, index=index, dtype=float)

    values = np.full(len(frame), np.nan, dtype=float)
    working = frame[["time", value_column]].copy()
    working["_source_position"] = np.arange(len(frame), dtype=int)
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=["time"]).sort_values("time", kind="stable")
    working = working.drop_duplicates(subset=["time"], keep="last")

    time_steps = working["time"].diff()
    value_steps = working[value_column].diff()
    changes = value_steps.where(time_steps.eq(pd.Timedelta(hours=1)))
    positions = working["_source_position"].to_numpy(dtype=int)
    values[positions] = changes.to_numpy(dtype=float)
    return pd.Series(values, index=frame.index, dtype=float)


def find_best_contiguous_window(frame: pd.DataFrame, window_size: int = 6) -> dict[str, Any] | None:
    """Find the lowest rolling AQI window without bridging missing-hour gaps."""
    if frame is None or frame.empty or not {"time", "us_aqi"}.issubset(frame.columns):
        return None
    working = frame[["time", "us_aqi"]].copy()
    working["time"] = pd.to_datetime(working["time"], errors="coerce")
    working["us_aqi"] = pd.to_numeric(working["us_aqi"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    working = working.dropna(subset=["time", "us_aqi"]).sort_values("time")
    working = working.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    if len(working) < window_size:
        return None

    working["segment"] = working["time"].diff().ne(pd.Timedelta(hours=1)).cumsum()
    best: dict[str, Any] | None = None
    for _, segment in working.groupby("segment", sort=False):
        if len(segment) < window_size:
            continue
        rolling_mean = segment["us_aqi"].rolling(window=window_size, min_periods=window_size).mean()
        if rolling_mean.notna().any():
            end_index = rolling_mean.idxmin()
            candidate = {
                "mean_aqi": float(rolling_mean.loc[end_index]),
                "end": pd.Timestamp(segment.loc[end_index, "time"]),
                "start": pd.Timestamp(segment.loc[end_index, "time"]) - pd.Timedelta(hours=window_size - 1),
                "window_size": int(window_size),
            }
            if best is None or candidate["mean_aqi"] < best["mean_aqi"]:
                best = candidate
    return best


def build_research_dataset(
    air_frame: pd.DataFrame,
    weather_frame: pd.DataFrame,
    *,
    city: str,
    region: str,
    country: str,
    latitude: float,
    longitude: float,
    timezone_name: str,
) -> pd.DataFrame:
    """Join forecast variables and create transparent, time-based research features."""
    if air_frame is None or air_frame.empty or "time" not in air_frame.columns:
        return pd.DataFrame()

    air = air_frame.copy()
    air["time"] = pd.to_datetime(air["time"], errors="coerce")
    air = air.dropna(subset=["time"]).sort_values("time")
    air = air.drop_duplicates(subset=["time"], keep="last")
    for column in ("us_aqi", "pm2_5", "pm10"):
        if column not in air.columns:
            air[column] = np.nan
        air[column] = pd.to_numeric(air[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    weather_columns = [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"
    ]
    weather = pd.DataFrame(columns=["time", *weather_columns])
    if weather_frame is not None and not weather_frame.empty and "time" in weather_frame.columns:
        weather = weather_frame.copy()
        weather["time"] = pd.to_datetime(weather["time"], errors="coerce")
        weather = weather.dropna(subset=["time"]).sort_values("time")
        weather = weather.drop_duplicates(subset=["time"], keep="last")
        for column in weather_columns:
            if column not in weather.columns:
                weather[column] = np.nan
            weather[column] = pd.to_numeric(weather[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        weather = weather[["time", *weather_columns]]

    result = air.merge(weather, on="time", how="left", validate="one_to_one")
    result["city"] = city
    result["region"] = region
    result["country"] = country
    result["latitude"] = float(latitude)
    result["longitude"] = float(longitude)
    result["timezone"] = timezone_name
    result["aqi_category"] = result["us_aqi"].apply(aqi_category)
    result["local_hour"] = result["time"].dt.hour
    result["day_of_week"] = result["time"].dt.day_name()
    result["month"] = result["time"].dt.month
    result["is_weekend"] = result["time"].dt.dayofweek >= 5
    result["pm25_to_pm10_ratio"] = np.where(
        result["pm10"].notna() & (result["pm10"] > 0) & result["pm2_5"].notna(),
        result["pm2_5"] / result["pm10"],
        np.nan,
    )
    result["aqi_change_1h"] = consecutive_hour_changes(result, "us_aqi").reindex(result.index)
    result["pm25_change_1h"] = consecutive_hour_changes(result, "pm2_5").reindex(result.index)
    result["wind_speed_change_1h"] = consecutive_hour_changes(result, "wind_speed_10m").reindex(result.index)

    ordered_columns = [
        "city", "region", "country", "latitude", "longitude", "timezone", "time",
        "local_hour", "day_of_week", "month", "is_weekend",
        "us_aqi", "aqi_category", "pm2_5", "pm10", "pm25_to_pm10_ratio",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
        "aqi_change_1h", "pm25_change_1h", "wind_speed_change_1h",
    ]
    return result[ordered_columns].reset_index(drop=True)


def generate_global_grid() -> list[tuple[float, float]]:
    # A broader set of map-only metropolitan samples gives more even coverage
    # across continents. The surface remains an interpolation, not a sensor map.
    locations = FEATURED_CITIES + GLOBAL_MAP_EXTRA_POINTS
    points = [(float(lat), float(lon)) for _, _, lat, lon in locations]
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
    """Fetch one coordinate batch and preserve rows even if the request fails."""
    try:
        frame = fetch_current_batch(tuple(points))
        return frame, None
    except (requests.RequestException, ValueError, KeyError) as exc:
        # Keep each requested coordinate in the result. This prevents a failed
        # middle batch from shifting city names against unrelated AQI values.
        fallback = pd.DataFrame([
            {
                "latitude": lat,
                "longitude": lon,
                "requested_latitude": lat,
                "requested_longitude": lon,
                "us_aqi": np.nan,
                "pm2_5": np.nan,
                "pm10": np.nan,
                "time": "",
            }
            for lat, lon in points
        ])
        return fallback, str(exc)


def collect_samples(points: list[tuple[float, float]], chunk_size: int = 36) -> tuple[pd.DataFrame, int]:
    """Collect multi-location samples while preserving the original point order."""
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    normalized_points: list[tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Each map sample must contain latitude and longitude.")
        latitude, longitude = float(point[0]), float(point[1])
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            continue
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            continue
        normalized_points.append((round(latitude, 4), round(longitude, 4)))
    points = list(dict.fromkeys(normalized_points))
    chunks = [tuple(points[i:i + chunk_size]) for i in range(0, len(points), chunk_size)]
    if not chunks:
        return pd.DataFrame(columns=["latitude", "longitude", "requested_latitude", "requested_longitude", "us_aqi", "pm2_5", "pm10", "time"]), 0
    frames: list[pd.DataFrame] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for frame, error in executor.map(fetch_chunk, chunks):
            if error:
                failures += 1
            if frame is not None and not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "requested_latitude", "requested_longitude", "us_aqi", "pm2_5", "pm10", "time"]), failures
    combined = pd.concat(frames, ignore_index=True)
    # Defensive sort by the requested coordinate sequence. The function's
    # callers can also join by coordinates, but the order is now deterministic.
    order = { (round(float(lat), 4), round(float(lon), 4)): i for i, (lat, lon) in enumerate(points) }
    if {"requested_latitude", "requested_longitude"}.issubset(combined.columns):
        combined["_point_order"] = combined.apply(
            lambda row: order.get((round(float(row["requested_latitude"]), 4), round(float(row["requested_longitude"]), 4)), len(order)),
            axis=1,
        )
        combined = combined.sort_values("_point_order", kind="stable").drop(columns="_point_order")
    return combined.reset_index(drop=True), failures


@st.cache_data(ttl=900, show_spinner=False)
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
    try:
        width, height = int(width), int(height)
        west, south, east, north = (float(value) for value in bounds)
        smoothing_km = float(smoothing_km)
    except (TypeError, ValueError):
        return None
    if width < 2 or height < 2 or not all(map(math.isfinite, (west, south, east, north, smoothing_km))):
        return None
    if east <= west or north <= south or smoothing_km <= 0:
        return None
    valid = frame.copy()
    if not {"latitude", "longitude", "us_aqi"}.issubset(valid.columns):
        return None
    for column in ("latitude", "longitude", "us_aqi"):
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=["latitude", "longitude", "us_aqi"])
    finite_mask = np.isfinite(valid[["latitude", "longitude", "us_aqi"]].to_numpy(dtype=float)).all(axis=1)
    valid = valid.loc[finite_mask].copy()
    valid = valid[
        valid["latitude"].between(-90, 90)
        & valid["longitude"].between(-180, 180)
        & valid["us_aqi"].between(0, 500)
    ]
    if valid.empty:
        return None

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
    fetch_historical_air_quality.clear()
    fetch_hourly_weather.clear()
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
      <div class="air-subtitle">Explore global air-quality forecasts, review hourly and daily outlooks, inspect archived trends, identify elevated pollution episodes, audit data quality, build research extracts, and compare major cities worldwide.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# WORLD / LOCAL MAP

st.markdown('<div class="air-section-label">AIR QUALITY MAP</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="air-legend"><div class="air-gradient"></div>'
    '<span>Lower estimate</span><span>→</span><span>Higher estimate</span></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Classic OpenStreetMap basemap with the interpolated AQI heat layer. "
    "This original map uses OpenStreetMap tiles and does not require a "
    "Mapbox, Esri, Carto, or other mapping-provider API key."
)

if selected_city_id is None:
    st.subheader("Global air-quality overview")
    st.caption(
        "The world view uses modelled AQI samples from representative major "
        "cities. Select a city in the sidebar to zoom into its local heat layer."
    )
    with st.spinner("Loading representative global air-quality forecasts..."):
        global_points = generate_global_grid()
        global_map_frame, global_failures = collect_samples(global_points, chunk_size=36)

    if not global_map_frame.empty:
        global_map_frame = global_map_frame.dropna(
            subset=["latitude", "longitude", "us_aqi"]
        )
        global_map_frame["lat_bin"] = global_map_frame["latitude"].round(2)
        global_map_frame["lon_bin"] = global_map_frame["longitude"].round(2)
        global_map_frame = global_map_frame.groupby(
            ["lat_bin", "lon_bin"], as_index=False
        ).agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            us_aqi=("us_aqi", "mean"),
            pm2_5=("pm2_5", "mean"),
            pm10=("pm10", "mean"),
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
        st.warning(
            "The OpenStreetMap basemap is visible, but no usable AQI samples "
            "were returned for the heat layer. Check your internet connection "
            "and try Refresh data."
        )
    else:
        valid_global_samples = int(global_map_frame["us_aqi"].notna().sum())
        map_metric_a, map_metric_b, map_metric_c = st.columns(3)
        with map_metric_a:
            st.metric("Valid map samples", valid_global_samples)
        with map_metric_b:
            st.metric("Sample locations requested", len(global_points))
        with map_metric_c:
            st.metric("Failed request batches", global_failures)
        st.caption(
            "The heat layer is an interpolation between modelled forecasts at "
            "representative city coordinates. It is a visual summary, not a "
            "measurement at every map pixel."
        )
        if global_failures:
            st.warning(
                f"{global_failures} map-data request batch(es) failed. "
                "Locations from failed batches are left without AQI values."
            )
else:
    if selected_place is None:
        st.warning(
            "The selected city could not be resolved to map coordinates. "
            "The global OpenStreetMap view is shown until location data is available."
        )
        fallback_map = build_folium_map(
            pd.DataFrame(),
            center=(18, 0),
            zoom=2,
            bounds=(-180, -60, 180, 80),
            smoothing_km=1450,
            raster_width=720,
            raster_height=360,
            height=650,
        )
        display_map(fallback_map, height=650)
    else:
        place_name = str(selected_place.get("name") or selected_city_row["name"])
        place_region = str(selected_place.get("admin1") or selected_city_row["region"])
        place_country = str(selected_place.get("country") or selected_city_row["country"])
        lat = float(selected_place["latitude"])
        lon = float(selected_place["longitude"])

        st.subheader(f"Air quality around {place_name}")
        st.caption(
            f"{place_region}, {place_country} · The map automatically centres "
            "on the selected city. The coloured surface is an interpolated "
            "modelled AQI heat layer, not street-level sensor coverage."
        )

        with st.spinner(f"Building the local heat layer around {place_name}..."):
            local_frame, local_failures = collect_samples(
                generate_local_grid(lat, lon), chunk_size=36
            )

        if not local_frame.empty:
            local_frame = local_frame.dropna(
                subset=["latitude", "longitude", "us_aqi"]
            )

        half_lat = 0.85
        half_lon = min(0.85 / max(abs(math.cos(math.radians(lat))), 0.2), 4.0)
        west = max(-179.8, lon - half_lon)
        east = min(179.8, lon + half_lon)
        south = max(-84.5, lat - half_lat)
        north = min(84.5, lat + half_lat)

        local_map = build_folium_map(
            local_frame,
            center=(lat, lon),
            zoom=9,
            bounds=(west, south, east, north),
            smoothing_km=22,
            raster_width=640,
            raster_height=540,
            height=650,
        )
        display_map(local_map, height=650)

        if local_frame.empty:
            st.warning(
                "The OpenStreetMap basemap is visible, but no usable local AQI "
                "samples were returned for the heat layer."
            )
        else:
            valid_local_samples = int(local_frame["us_aqi"].notna().sum())
            local_metric_a, local_metric_b = st.columns(2)
            with local_metric_a:
                st.metric("Valid local samples", valid_local_samples)
            with local_metric_b:
                st.metric("Failed request batches", local_failures)
            st.caption(
                "Nearby coordinates may resolve to the same atmospheric-model "
                "grid cell. The heat layer is therefore not equivalent to "
                "street-level sensor measurements."
            )
            if local_failures:
                st.warning(f"{local_failures} local map-data request batch(es) failed.")


# SELECTED CITY CONDITIONS + FORECAST TRENDS
# ============================================================

st.markdown("---")
st.markdown('<div class="air-section-label">SELECTED LOCATION</div>', unsafe_allow_html=True)
selected_aqi_for_benchmark: float | None = None
selected_city_for_benchmark: str | None = None
selected_country_for_benchmark: str | None = None
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

        selected_timezone = safe_timezone_name(selected_place.get("timezone"))
        selected_hourly = pd.DataFrame()
        selected_hourly_weather = pd.DataFrame()
        selected_weather: dict = {}
        hourly_error = None
        hourly_weather_error = None
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
        try:
            selected_hourly_weather = fetch_hourly_weather(
                float(selected_place["latitude"]),
                float(selected_place["longitude"]),
                selected_timezone,
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            hourly_weather_error = str(exc)

        if not selected_hourly.empty:
            local_now = pd.Timestamp.now(tz=selected_timezone).tz_localize(None)
            usable = selected_hourly.dropna(subset=["us_aqi"])
            source = usable if not usable.empty else selected_hourly
            nearest_idx = (source["time"] - local_now).abs().idxmin()
            current = source.loc[nearest_idx]
            current_aqi = safe_number(current.get("us_aqi"))
            selected_aqi_for_benchmark = current_aqi
            selected_city_for_benchmark = selected_name
            selected_country_for_benchmark = selected_country_for_data
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

            st.markdown("### Forecasts")
            future = (
                selected_hourly[selected_hourly["time"] >= local_now.floor("h")]
                .copy()
                .sort_values("time")
            )
            if future.empty:
                st.warning("The API returned no forecast timestamps at or after the current local hour. Past records are not relabelled as future forecasts.")

            # ------------------------------------------------------------
            # Forecast data health audit
            # ------------------------------------------------------------
            st.markdown('<div class="air-section-label">FORECAST DATA QUALITY</div>', unsafe_allow_html=True)
            st.markdown("#### Data availability and integrity")
            quality = audit_forecast_data(future)
            expected_rows = int(quality["rows"])
            aqi_coverage = float(quality["aqi_coverage"])
            pm25_coverage = float(quality["pm25_coverage"])
            pm10_coverage = float(quality["pm10_coverage"])

            quality_cols = st.columns(4)
            with quality_cols[0]:
                st.metric("Forecast timestamps", f"{expected_rows}")
            with quality_cols[1]:
                st.metric("AQI data coverage", f"{aqi_coverage:.0f}%")
            with quality_cols[2]:
                st.metric("PM2.5 data coverage", f"{pm25_coverage:.0f}%")
            with quality_cols[3]:
                st.metric("PM10 data coverage", f"{pm10_coverage:.0f}%")

            st.progress(min(max(aqi_coverage / 100.0, 0.0), 1.0))
            quality_rows = [
                {"Validation check": "Duplicate timestamps", "Result": quality["duplicate_timestamps"], "Status": "OK" if quality["duplicate_timestamps"] == 0 else "Review"},
                {"Validation check": "Missing hourly intervals", "Result": quality["missing_hour_intervals"], "Status": "OK" if quality["missing_hour_intervals"] == 0 else "Review"},
                {"Validation check": "Irregular time intervals", "Result": quality["irregular_intervals"], "Status": "OK" if quality["irregular_intervals"] == 0 else "Review"},
                {"Validation check": "Negative / out-of-range values", "Result": quality["invalid_values"], "Status": "OK" if quality["invalid_values"] == 0 else "Review"},
                {"Validation check": "Consecutive hourly transitions", "Result": quality["valid_transitions"], "Status": "Informational"},
            ]
            st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"Timestamp continuity: {quality['continuity_pct']:.0f}%. "
                "This audit checks missing values, duplicate timestamps and interval consistency. "
                "It does not measure forecast accuracy or prove that model estimates match conditions on the ground."
            )
            if quality["duplicate_timestamps"] or quality["missing_hour_intervals"] or quality["invalid_values"]:
                st.warning("The forecast extract has data-quality issues. Review the checks above before using it for analysis.")
            if aqi_coverage < 80 and expected_rows:
                st.warning("A noticeable share of AQI values is missing. Interpret the forecast charts with extra caution.")

            # ------------------------------------------------------------
            # HOURLY FORECAST — ALWAYS SHOWN ON THE MAIN PAGE
            # ------------------------------------------------------------
            st.markdown("---")
            st.markdown('<div class="air-section-label">HOURLY FORECAST</div>', unsafe_allow_html=True)
            st.markdown("### Next 48 hours")
            hourly_frame = future.head(48).copy()
            st.caption(
                "Hourly model estimates in the selected city's local time. "
                "Hover over a chart to inspect individual forecast timestamps."
            )

            if hourly_frame.empty:
                st.info("Hourly forecast data is not available for this location right now.")
            else:
                hourly_left, hourly_right = st.columns(2, gap="large")
                with hourly_left:
                    st.markdown("##### US AQI trend")
                    hourly_aqi_fig = go.Figure()
                    hourly_aqi_fig.add_trace(go.Scatter(
                        x=hourly_frame["time"], y=hourly_frame["us_aqi"], mode="lines",
                        line={"color": "#28734A", "width": 2.7},
                        fill="tozeroy", fillcolor="rgba(40,115,74,.08)",
                        connectgaps=False, name="US AQI",
                        hovertemplate="%{x|%d %b, %H:%M}<br>US AQI: %{y:.0f}<extra></extra>",
                    ))
                    hourly_aqi_fig.update_layout(showlegend=False)
                    hourly_aqi_fig.update_yaxes(title_text="US AQI")
                    make_chart_layout(hourly_aqi_fig, height=330)
                    st.plotly_chart(hourly_aqi_fig, use_container_width=True, config={"displaylogo": False})

                with hourly_right:
                    st.markdown("##### Particulate matter")
                    hourly_pm_fig = go.Figure()
                    hourly_pm_fig.add_trace(go.Scatter(
                        x=hourly_frame["time"], y=hourly_frame["pm2_5"], mode="lines", name="PM2.5",
                        line={"color": "#28734A", "width": 2.5}, connectgaps=False,
                    ))
                    hourly_pm_fig.add_trace(go.Scatter(
                        x=hourly_frame["time"], y=hourly_frame["pm10"], mode="lines", name="PM10",
                        line={"color": "#6498C9", "width": 2.5}, connectgaps=False,
                    ))
                    hourly_pm_fig.update_layout(legend={"orientation": "h", "y": 1.08, "x": 0})
                    hourly_pm_fig.update_yaxes(title_text="Concentration (µg/m³)")
                    make_chart_layout(hourly_pm_fig, height=330)
                    st.plotly_chart(hourly_pm_fig, use_container_width=True, config={"displaylogo": False})

                hourly_insights = hourly_frame.dropna(subset=["us_aqi"]).copy()
                if hourly_insights.empty:
                    st.info("AQI values are missing for the selected hourly forecast window.")
                else:
                    hourly_insights["aqi_category"] = hourly_insights["us_aqi"].apply(aqi_category)
                    hourly_metrics = st.columns(4)
                    hourly_metric_values = [
                        ("Average forecast AQI", format_value(hourly_insights["us_aqi"].mean(), decimals=0)),
                        ("Peak forecast AQI", format_value(hourly_insights["us_aqi"].max(), decimals=0)),
                        ("Hours above AQI 100", str(int((hourly_insights["us_aqi"] > 100).sum()))),
                        ("Lowest forecast AQI", format_value(hourly_insights["us_aqi"].min(), decimals=0)),
                    ]
                    for col, (label, value) in zip(hourly_metrics, hourly_metric_values):
                        with col:
                            st.metric(label, value)

                    # Search each uninterrupted hourly segment independently. Missing
                    # AQI values no longer prevent analysis of valid six-hour windows.
                    best_window = find_best_contiguous_window(hourly_frame, window_size=6)
                    if best_window is not None:
                        st.markdown(
                            f"<div class='air-info' style='min-height:0;margin:10px 0;'>"
                            f"<strong>Lowest six-consecutive-hour average:</strong> {best_window['mean_aqi']:.0f} US AQI, "
                            f"from {html.escape(best_window['start'].strftime('%d %b, %I:%M %p'))} "
                            f"to {html.escape(best_window['end'].strftime('%d %b, %I:%M %p'))}. "
                            "This is a comparison within the model forecast, not a guarantee of future conditions."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("A six-hour summary needs at least six consecutive hourly AQI readings in one uninterrupted segment.")

                    hourly_category_counts = hourly_insights["aqi_category"].value_counts()
                    category_order = [
                        "Good", "Moderate", "Unhealthy for sensitive groups",
                        "Unhealthy", "Very unhealthy", "Hazardous",
                    ]
                    hourly_category_counts = hourly_category_counts.reindex(category_order, fill_value=0)
                    hourly_category_counts = hourly_category_counts[hourly_category_counts > 0]
                    if not hourly_category_counts.empty:
                        hourly_category_fig = go.Figure(go.Bar(
                            x=hourly_category_counts.index.tolist(),
                            y=hourly_category_counts.values.tolist(),
                            marker_color=[AQI_COLORS[name] for name in hourly_category_counts.index],
                            hovertemplate="%{x}<br>%{y} forecast hours<extra></extra>",
                        ))
                        hourly_category_fig.update_layout(
                            title="Hourly forecast by US AQI category",
                            showlegend=False, yaxis_title="Number of hours", xaxis_title="", bargap=0.35,
                        )
                        make_chart_layout(hourly_category_fig, height=285)
                        st.plotly_chart(hourly_category_fig, use_container_width=True, config={"displaylogo": False})

                hourly_export = hourly_frame.copy()
                hourly_export["aqi_category"] = hourly_export["us_aqi"].apply(aqi_category)
                hourly_export = hourly_export.rename(columns={
                    "time": "local_time",
                    "us_aqi": "us_aqi_modelled",
                    "pm2_5": "pm2_5_ug_m3",
                    "pm10": "pm10_ug_m3",
                })
                city_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in selected_name).strip("_")
                st.download_button(
                    "Download hourly forecast (CSV)",
                    data=hourly_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"airaware_{city_slug}_hourly_forecast.csv",
                    mime="text/csv",
                    key="download_hourly_forecast_day5",
                    use_container_width=False,
                )

            # ------------------------------------------------------------
            # DAILY FORECAST — ALSO SHOWN ON THE MAIN PAGE
            # ------------------------------------------------------------
            st.markdown("---")
            st.markdown('<div class="air-section-label">DAILY FORECAST</div>', unsafe_allow_html=True)
            st.markdown("### Next 5 days")
            daily_frame = aggregate_daily_forecast(future, today=local_now.date())
            st.caption(
                "Daily statistics are aggregated from the hourly model forecasts. "
                "The first day may be partial because earlier hours are not included."
            )

            if daily_frame.empty or daily_frame["mean_us_aqi"].notna().sum() == 0:
                st.info("Daily forecast data is not available for this location right now.")
            else:
                daily_left, daily_right = st.columns(2, gap="large")
                with daily_left:
                    st.markdown("##### Daily AQI range")
                    daily_aqi_fig = go.Figure()
                    daily_aqi_fig.add_trace(go.Scatter(
                        x=daily_frame["day_label"], y=daily_frame["max_us_aqi"],
                        mode="lines+markers", name="Daily maximum",
                        line={"color": "#D97735", "width": 1.8, "dash": "dot"},
                        marker={"size": 6}, connectgaps=False,
                    ))
                    daily_aqi_fig.add_trace(go.Scatter(
                        x=daily_frame["day_label"], y=daily_frame["min_us_aqi"],
                        mode="lines+markers", name="Daily minimum",
                        line={"color": "#6498C9", "width": 1.8, "dash": "dot"},
                        marker={"size": 6}, connectgaps=False,
                    ))
                    daily_aqi_fig.add_trace(go.Scatter(
                        x=daily_frame["day_label"], y=daily_frame["mean_us_aqi"],
                        mode="lines+markers", name="Daily average",
                        line={"color": "#28734A", "width": 3}, marker={"size": 9},
                        connectgaps=False,
                        hovertemplate="%{x}<br>Average AQI: %{y:.0f}<extra></extra>",
                    ))
                    daily_aqi_fig.update_layout(
                        yaxis_title="US AQI", legend={"orientation": "h", "y": 1.12, "x": 0},
                    )
                    make_chart_layout(daily_aqi_fig, height=350)
                    st.plotly_chart(daily_aqi_fig, use_container_width=True, config={"displaylogo": False})

                with daily_right:
                    st.markdown("##### Daily particulate matter")
                    daily_pm_fig = go.Figure()
                    daily_pm_fig.add_trace(go.Bar(
                        x=daily_frame["day_label"], y=daily_frame["mean_pm2_5"],
                        name="Average PM2.5", marker_color="#28734A",
                    ))
                    daily_pm_fig.add_trace(go.Bar(
                        x=daily_frame["day_label"], y=daily_frame["mean_pm10"],
                        name="Average PM10", marker_color="#6498C9",
                    ))
                    daily_pm_fig.update_layout(
                        barmode="group", yaxis_title="Concentration (µg/m³)",
                        legend={"orientation": "h", "y": 1.12, "x": 0},
                    )
                    make_chart_layout(daily_pm_fig, height=350)
                    st.plotly_chart(daily_pm_fig, use_container_width=True, config={"displaylogo": False})

                valid_daily = daily_frame.dropna(subset=["mean_us_aqi"]).copy()
                daily_metrics = st.columns(4)
                daily_metric_values = [
                    ("Average daily AQI", format_value(valid_daily["mean_us_aqi"].mean(), decimals=0)),
                    ("Highest daily maximum", format_value(daily_frame["max_us_aqi"].max(), decimals=0)),
                    ("Days with mean AQI above 100", str(int((valid_daily["mean_us_aqi"] > 100).sum()))),
                    ("Days with AQI data", str(len(valid_daily))),
                ]
                for col, (label, value) in zip(daily_metrics, daily_metric_values):
                    with col:
                        st.metric(label, value)

                daily_table = daily_frame[[
                    "day_label", "mean_us_aqi", "min_us_aqi", "max_us_aqi",
                    "mean_pm2_5", "mean_pm10", "aqi_hours", "available_hours",
                ]].copy()
                daily_table["AQI coverage"] = np.where(
                    daily_table["available_hours"] > 0,
                    daily_table["aqi_hours"] / daily_table["available_hours"] * 100,
                    np.nan,
                )
                daily_table = daily_table.rename(columns={
                    "day_label": "Forecast day",
                    "mean_us_aqi": "Mean AQI",
                    "min_us_aqi": "Minimum AQI",
                    "max_us_aqi": "Maximum AQI",
                    "mean_pm2_5": "Mean PM2.5 (µg/m³)",
                    "mean_pm10": "Mean PM10 (µg/m³)",
                    "aqi_hours": "Hours with AQI data",
                    "available_hours": "Forecast timestamps",
                })
                st.markdown("##### Daily forecast details")
                st.dataframe(
                    daily_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Mean AQI": st.column_config.NumberColumn(format="%.0f"),
                        "Minimum AQI": st.column_config.NumberColumn(format="%.0f"),
                        "Maximum AQI": st.column_config.NumberColumn(format="%.0f"),
                        "Mean PM2.5 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                        "Mean PM10 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                        "AQI coverage": st.column_config.NumberColumn(format="%.0f%%"),
                    },
                )

                daily_export = daily_frame.copy().rename(columns={
                    "forecast_date": "local_date",
                    "mean_us_aqi": "mean_us_aqi_modelled",
                    "min_us_aqi": "minimum_us_aqi_modelled",
                    "max_us_aqi": "maximum_us_aqi_modelled",
                    "mean_pm2_5": "mean_pm2_5_ug_m3",
                    "mean_pm10": "mean_pm10_ug_m3",
                    "aqi_hours": "hours_with_aqi_data",
                })
                daily_export["aqi_category"] = daily_export["mean_us_aqi_modelled"].apply(aqi_category)
                st.download_button(
                    "Download daily forecast (CSV)",
                    data=daily_export.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"airaware_{city_slug}_daily_forecast.csv",
                    mime="text/csv",
                    key="download_daily_forecast_day5",
                    use_container_width=False,
                )

            st.caption(
                "All forecast values are model estimates, not direct observations from a local monitoring station. "
                "Daily values are calculated from the hourly response returned for this location."
            )

            # ============================================================
            # DAY 5 FEATURE: FORECAST VOLATILITY
            # Show how much the forecast varies over the next 48 hours.
            # This is a descriptive statistic, not a confidence interval.
            # ============================================================
            st.markdown("---")
            st.markdown('<div class="air-section-label">FORECAST STABILITY</div>', unsafe_allow_html=True)
            st.markdown("### How much does the forecast change?")
            volatility_frame = future.head(48).copy().sort_values("time")
            volatility_frame["aqi_change"] = consecutive_hour_changes(volatility_frame, "us_aqi").reindex(volatility_frame.index)
            valid_changes = volatility_frame["aqi_change"].dropna()
            available_aqi = pd.to_numeric(volatility_frame["us_aqi"], errors="coerce").dropna()
            if not valid_changes.empty:
                mean_absolute_change = valid_changes.abs().mean()
                largest_rise = max(0.0, float(valid_changes.max()))
                largest_drop = max(0.0, float(-valid_changes.min()))
                variability = available_aqi.std(ddof=0) if len(available_aqi) else np.nan
                volatility_cols = st.columns(4)
                with volatility_cols[0]:
                    st.metric("Average hourly AQI change", format_value(mean_absolute_change, decimals=1))
                with volatility_cols[1]:
                    st.metric("Largest hourly increase", format_value(largest_rise, decimals=1))
                with volatility_cols[2]:
                    st.metric("Largest hourly decrease", format_value(largest_drop, decimals=1))
                with volatility_cols[3]:
                    st.metric("Valid 1-hour comparisons", str(len(valid_changes)))
                st.caption(
                    f"The standard deviation of available AQI estimates is {format_value(variability, decimals=1)}. "
                    "Changes are calculated only between timestamps exactly one hour apart, so gaps are not treated as hourly changes. "
                    "These statistics do not quantify model uncertainty."
                )
            else:
                st.info("At least two non-missing AQI estimates exactly one hour apart are needed to calculate hourly forecast changes.")

            # ENVIRONMENTAL RELATIONSHIPS: WEATHER & POLLUTION EXPLORER
            # These are forecast-to-forecast associations, not causal claims.
            # ============================================================
            st.markdown("---")
            st.markdown('<div class="air-section-label">ENVIRONMENTAL RELATIONSHIPS</div>', unsafe_allow_html=True)
            st.markdown("### Weather & pollution explorer")
            st.caption(
                "Explore whether changes in the weather forecast move alongside changes in the air-quality forecast. "
                "This analysis independently uses matching hourly weather and air-quality forecasts. It is exploratory and does not prove that weather caused a pollution change."
            )

            if selected_hourly_weather.empty:
                st.info("Hourly weather data is unavailable for this city right now, so the relationship explorer cannot be calculated.")
                if hourly_weather_error:
                    with st.expander("Weather data details"):
                        st.code(hourly_weather_error)
            else:
                relationship_air = selected_hourly.drop_duplicates(subset=["time"], keep="last")
                relationship_weather = selected_hourly_weather.drop_duplicates(subset=["time"], keep="last")
                relationship_data = relationship_air.merge(
                    relationship_weather,
                    on="time",
                    how="inner",
                    validate="one_to_one",
                )
                relationship_data = relationship_data[
                    relationship_data["time"] >= local_now.floor("h")
                ].head(120).copy()

                pollutant_columns = ["us_aqi", "pm2_5", "pm10"]
                weather_columns = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
                relationship_data = relationship_data.dropna(subset=pollutant_columns + weather_columns, how="all")

                if len(relationship_data) < 12:
                    st.info("At least 12 matching hourly weather and air-quality forecast rows are needed for a useful comparison. More data may become available after refreshing.")
                else:
                    # Spearman's rank correlation: rank the values, then use
                    # pandas' built-in Pearson correlation on the ranks.
                    corr_matrix = pd.DataFrame(index=weather_columns, columns=pollutant_columns, dtype=float)
                    for weather_column in weather_columns:
                        for pollutant_column in pollutant_columns:
                            pair = relationship_data[[weather_column, pollutant_column]].dropna()
                            if len(pair) >= 12 and pair[weather_column].nunique() > 1 and pair[pollutant_column].nunique() > 1:
                                corr_matrix.loc[weather_column, pollutant_column] = pair[weather_column].rank().corr(pair[pollutant_column].rank())

                    def correlation_text(value: float | None) -> str:
                        if value is None or pd.isna(value):
                            return "Not enough variation"
                        magnitude = abs(float(value))
                        if magnitude < 0.20:
                            strength = "Very weak"
                        elif magnitude < 0.40:
                            strength = "Weak"
                        elif magnitude < 0.60:
                            strength = "Moderate"
                        elif magnitude < 0.80:
                            strength = "Strong"
                        else:
                            strength = "Very strong"
                        direction = "positive" if value > 0 else "negative"
                        return f"{strength} {direction} association"

                    primary_pairs = [
                        ("wind_speed_10m", "AQI vs wind", "Wind speed", "us_aqi"),
                        ("relative_humidity_2m", "AQI vs humidity", "Relative humidity", "us_aqi"),
                        ("temperature_2m", "AQI vs temperature", "Temperature", "us_aqi"),
                    ]
                    correlation_cols = st.columns(3)
                    for col, (weather_col, label, _, pollutant_col) in zip(correlation_cols, primary_pairs):
                        value = corr_matrix.loc[weather_col, pollutant_col]
                        with col:
                            st.metric(label, "—" if pd.isna(value) else f"{value:+.2f}")
                            st.caption(correlation_text(value))

                    weather_names = {
                        "temperature_2m": "Temperature",
                        "relative_humidity_2m": "Humidity",
                        "wind_speed_10m": "Wind speed",
                        "precipitation": "Precipitation",
                    }
                    pollutant_names = {
                        "us_aqi": "US AQI",
                        "pm2_5": "PM2.5",
                        "pm10": "PM10",
                    }
                    heatmap_values = corr_matrix.to_numpy(dtype=float)
                    heatmap_text = np.empty(heatmap_values.shape, dtype=object)
                    for row_index in range(heatmap_values.shape[0]):
                        for col_index in range(heatmap_values.shape[1]):
                            value = heatmap_values[row_index, col_index]
                            heatmap_text[row_index, col_index] = "—" if np.isnan(value) else f"{value:+.2f}"

                    relationship_fig = go.Figure(go.Heatmap(
                        z=heatmap_values,
                        x=[pollutant_names[column] for column in pollutant_columns],
                        y=[weather_names[column] for column in weather_columns],
                        text=heatmap_text,
                        texttemplate="%{text}",
                        textfont={"color": "#17251D", "size": 12},
                        zmin=-1,
                        zmax=1,
                        zmid=0,
                        colorscale=[
                            [0.0, "#D96B63"],
                            [0.5, "#F3F5F2"],
                            [1.0, "#39865D"],
                        ],
                        colorbar={
                            "title": {"text": "Spearman r", "font": {"color": "#17251D"}},
                            "tickfont": {"color": "#17251D"},
                        },
                        hovertemplate="%{y} vs %{x}<br>Spearman r: %{z:.2f}<extra></extra>",
                    ))
                    relationship_fig.update_layout(
                        title="Weather–pollutant rank correlations",
                        xaxis_title="Air-quality variable",
                        yaxis_title="Weather variable",
                    )
                    make_chart_layout(relationship_fig, height=330)
                    st.plotly_chart(relationship_fig, use_container_width=True, config={"displaylogo": False})

                    scatter_candidates = relationship_data[["wind_speed_10m", "us_aqi"]].dropna()
                    if len(scatter_candidates) >= 3:
                        scatter_fig = go.Figure(go.Scatter(
                            x=scatter_candidates["wind_speed_10m"],
                            y=scatter_candidates["us_aqi"],
                            mode="markers",
                            marker={"color": "#28734A", "size": 8, "opacity": 0.72, "line": {"color": "#FFFFFF", "width": 1}},
                            name="Hourly forecasts",
                            hovertemplate="Wind: %{x:.1f} km/h<br>US AQI: %{y:.0f}<extra></extra>",
                        ))
                        scatter_fig.update_layout(
                            title="Wind speed and modelled AQI",
                            xaxis_title="Wind speed (km/h)",
                            yaxis_title="US AQI",
                            showlegend=False,
                        )
                        make_chart_layout(scatter_fig, height=300)
                        st.plotly_chart(scatter_fig, use_container_width=True, config={"displaylogo": False})

                    st.caption(
                        f"Analysis uses {len(relationship_data)} matching hourly forecast rows from the available five-day forecast. "
                        "Spearman r describes how two variables move together in this forecast window; "
                        "it does not establish causation and should not be generalized beyond this city and period."
                    )

            # ============================================================
            # Research-ready forecast extract
            # Joins air quality and weather with transparent time features.
            # This is a forecast extract, not historical ground-truth data.
            # ============================================================
            st.markdown("---")
            st.markdown('<div class="air-section-label">RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
            st.markdown("### Build a combined research dataset")
            st.caption(
                "Combine hourly air-quality and weather forecasts into one structured CSV. "
                "The file includes city metadata, local calendar features, pollutant values, weather variables, "
                "and one-hour changes calculated only across consecutive timestamps."
            )
            research_dataset = build_research_dataset(
                future,
                selected_hourly_weather,
                city=selected_name,
                region=selected_region_for_data,
                country=selected_country_for_data,
                latitude=float(selected_place["latitude"]),
                longitude=float(selected_place["longitude"]),
                timezone_name=selected_timezone,
            )
            if research_dataset.empty:
                st.info("There are no forecast rows available to export for this location.")
            else:
                matched_weather_rows = int(research_dataset["temperature_2m"].notna().sum())
                research_cols = st.columns(3)
                with research_cols[0]:
                    st.metric("Research rows", len(research_dataset))
                with research_cols[1]:
                    st.metric("Available variables", len(research_dataset.columns))
                with research_cols[2]:
                    st.metric("Rows with weather data", f"{matched_weather_rows}/{len(research_dataset)}")
                st.dataframe(research_dataset.head(10), use_container_width=True, hide_index=True)
                research_slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in selected_name).strip("_")
                st.download_button(
                    "Download combined research dataset (CSV)",
                    data=research_dataset.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"airaware_{research_slug}_research_forecast.csv",
                    mime="text/csv",
                    key="download_research_dataset_day6",
                    use_container_width=False,
                )
                st.caption(
                    "Research note: this export contains model forecasts for future timestamps, not observed historical outcomes. "
                    "It is useful for exploring features and preparing a data pipeline, but it must not be used as ground-truth labels "
                    "to claim predictive model accuracy. A later modelling stage will need archived historical observations and a time-based evaluation split."
                )

            # ============================================================
            # DAYS 8–9: HISTORICAL MODELLED AIR-QUALITY EXPLORER
            # Explicitly loaded so the archive request does not run on every
            # unrelated Streamlit rerun. Results are tied to selected coords.
            # ============================================================
            st.markdown("---")
            st.markdown('<div class="air-section-label">HISTORICAL EXPLORER</div>', unsafe_allow_html=True)
            st.markdown("### Explore recent air-quality history")
            st.caption(
                "Load 7, 30, 60 or 90 days of archived air-quality model output. "
                "The archive is useful for pattern exploration, but it is not a set of independent ground-monitor observations."
            )

            history_period_key = (
                round(float(selected_place["latitude"]), 4),
                round(float(selected_place["longitude"]), 4),
                selected_timezone,
            )
            with st.form("historical_air_quality_form"):
                history_days_choice = st.selectbox(
                    "Historical period",
                    options=[7, 30, 60, 90],
                    index=1,
                    format_func=lambda value: f"Previous {value} days",
                    key="historical_days_choice_day8_9",
                )
                load_history_clicked = st.form_submit_button("Load historical analysis")

            if load_history_clicked:
                try:
                    with st.spinner(f"Loading the previous {history_days_choice} days of archived air-quality data..."):
                        fetched_history = fetch_historical_air_quality(
                            float(selected_place["latitude"]),
                            float(selected_place["longitude"]),
                            selected_timezone,
                            int(history_days_choice),
                        )
                    st.session_state["airaware_history_frame"] = fetched_history
                    st.session_state["airaware_history_key"] = (*history_period_key, int(history_days_choice))
                    st.session_state["airaware_history_error"] = None
                except (requests.RequestException, ValueError, KeyError, pd.errors.ParserError) as history_exc:
                    st.session_state["airaware_history_frame"] = None
                    st.session_state["airaware_history_key"] = (*history_period_key, int(history_days_choice))
                    st.session_state["airaware_history_error"] = str(history_exc)

            requested_history_key = (*history_period_key, int(history_days_choice))
            stored_history_key = st.session_state.get("airaware_history_key")
            if stored_history_key == requested_history_key:
                stored_history_error = st.session_state.get("airaware_history_error")
                stored_history_frame = st.session_state.get("airaware_history_frame")
                if stored_history_error:
                    st.warning("Historical data could not be loaded for this location and period.")
                    with st.expander("Historical data details"):
                        st.code(str(stored_history_error))
                elif isinstance(stored_history_frame, pd.DataFrame) and not stored_history_frame.empty:
                    history_frame = stored_history_frame.copy()
                    history_frame = history_frame.sort_values("time", kind="stable").drop_duplicates("time", keep="last")
                    history_daily = summarize_historical_air_quality(history_frame)
                    history_valid = history_frame.dropna(subset=["us_aqi"]).copy()
                    history_valid["us_aqi"] = pd.to_numeric(history_valid["us_aqi"], errors="coerce")
                    history_valid = history_valid.dropna(subset=["us_aqi"])

                    history_metric_cols = st.columns(4)
                    with history_metric_cols[0]:
                        st.metric("Archived records", f"{len(history_frame):,}")
                    with history_metric_cols[1]:
                        st.metric("Mean archived AQI", format_value(history_valid["us_aqi"].mean(), decimals=0) if not history_valid.empty else "—")
                    with history_metric_cols[2]:
                        st.metric("Highest archived AQI", format_value(history_valid["us_aqi"].max(), decimals=0) if not history_valid.empty else "—")
                    with history_metric_cols[3]:
                        aqi_coverage_history = (history_frame["us_aqi"].notna().mean() * 100) if len(history_frame) else 0.0
                        st.metric("AQI availability", f"{aqi_coverage_history:.0f}%")

                    history_start = pd.Timestamp(history_frame["time"].min())
                    history_end = pd.Timestamp(history_frame["time"].max())
                    st.caption(
                        f"Returned local-time range: {history_start.strftime('%d %b %Y %H:%M')} to "
                        f"{history_end.strftime('%d %b %Y %H:%M')} · typical sample interval: {infer_sample_interval(history_frame)}."
                    )

                    if not history_daily.empty and history_daily["mean_us_aqi"].notna().any():
                        trend_left, trend_right = st.columns(2, gap="large")
                        with trend_left:
                            st.markdown("##### Daily average AQI and 7-day trend")
                            historical_trend_fig = go.Figure()
                            historical_trend_fig.add_trace(go.Scatter(
                                x=history_daily["date"], y=history_daily["mean_us_aqi"],
                                mode="lines+markers", name="Daily mean AQI",
                                line={"color": "#6498C9", "width": 1.8},
                                marker={"size": 5}, connectgaps=False,
                            ))
                            historical_trend_fig.add_trace(go.Scatter(
                                x=history_daily["date"], y=history_daily["mean_7d_aqi"],
                                mode="lines", name="7-day moving average",
                                line={"color": "#28734A", "width": 3}, connectgaps=False,
                            ))
                            historical_trend_fig.update_layout(
                                yaxis_title="Modelled US AQI",
                                legend={"orientation": "h", "y": 1.12, "x": 0},
                            )
                            make_chart_layout(historical_trend_fig, height=330)
                            st.plotly_chart(historical_trend_fig, use_container_width=True, config={"displaylogo": False})

                        with trend_right:
                            st.markdown("##### Daily particulate matter")
                            historical_pm_fig = go.Figure()
                            historical_pm_fig.add_trace(go.Scatter(
                                x=history_daily["date"], y=history_daily["mean_pm2_5"],
                                mode="lines+markers", name="PM2.5",
                                line={"color": "#28734A", "width": 2.4}, connectgaps=False,
                            ))
                            historical_pm_fig.add_trace(go.Scatter(
                                x=history_daily["date"], y=history_daily["mean_pm10"],
                                mode="lines+markers", name="PM10",
                                line={"color": "#6498C9", "width": 2.4}, connectgaps=False,
                            ))
                            historical_pm_fig.update_layout(
                                yaxis_title="Concentration (µg/m³)",
                                legend={"orientation": "h", "y": 1.12, "x": 0},
                            )
                            make_chart_layout(historical_pm_fig, height=330)
                            st.plotly_chart(historical_pm_fig, use_container_width=True, config={"displaylogo": False})

                        # Day 9: within-period patterns and robust high-day flags.
                        history_valid["hour"] = pd.to_datetime(history_valid["time"]).dt.hour
                        history_valid["weekday"] = pd.to_datetime(history_valid["time"]).dt.dayofweek
                        hour_profile = history_valid.groupby("hour", as_index=False)["us_aqi"].mean()
                        weekday_profile = history_valid.groupby("weekday", as_index=False)["us_aqi"].mean()
                        pattern_left, pattern_right = st.columns(2, gap="large")
                        with pattern_left:
                            st.markdown("##### Typical AQI by local hour")
                            hour_fig = go.Figure(go.Bar(
                                x=hour_profile["hour"], y=hour_profile["us_aqi"],
                                marker_color="#28734A",
                                hovertemplate="Hour %{x}:00<br>Mean AQI: %{y:.1f}<extra></extra>",
                            ))
                            hour_fig.update_layout(xaxis_title="Local hour", yaxis_title="Mean modelled US AQI", showlegend=False)
                            make_chart_layout(hour_fig, height=285)
                            st.plotly_chart(hour_fig, use_container_width=True, config={"displaylogo": False})
                        with pattern_right:
                            st.markdown("##### Typical AQI by weekday")
                            weekday_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
                            weekday_profile["weekday_name"] = weekday_profile["weekday"].map(weekday_names)
                            weekday_fig = go.Figure(go.Bar(
                                x=weekday_profile["weekday_name"], y=weekday_profile["us_aqi"],
                                marker_color="#6498C9",
                                hovertemplate="%{x}<br>Mean AQI: %{y:.1f}<extra></extra>",
                            ))
                            weekday_fig.update_layout(xaxis_title="Local weekday", yaxis_title="Mean modelled US AQI", showlegend=False)
                            make_chart_layout(weekday_fig, height=285)
                            st.plotly_chart(weekday_fig, use_container_width=True, config={"displaylogo": False})

                        valid_daily_history = history_daily.dropna(subset=["mean_us_aqi"]).copy()
                        if len(valid_daily_history) >= 14:
                            q1 = float(valid_daily_history["mean_us_aqi"].quantile(0.25))
                            q3 = float(valid_daily_history["mean_us_aqi"].quantile(0.75))
                            iqr = q3 - q1
                            upper_fence = q3 + 1.5 * iqr
                            unusually_high = valid_daily_history[valid_daily_history["mean_us_aqi"] > upper_fence]
                            st.markdown("##### Unusually high days within this period")
                            if not unusually_high.empty:
                                st.dataframe(
                                    unusually_high[["date", "mean_us_aqi", "max_us_aqi"]].rename(columns={
                                        "date": "Local date", "mean_us_aqi": "Daily mean AQI", "max_us_aqi": "Daily maximum AQI"
                                    }),
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Daily mean AQI": st.column_config.NumberColumn(format="%.0f"),
                                        "Daily maximum AQI": st.column_config.NumberColumn(format="%.0f"),
                                    },
                                )
                                st.caption(
                                    f"Flagged using the 1.5×IQR upper fence ({upper_fence:.1f}) within this selected period. "
                                    "These are statistical outliers relative to this dataset, not official health-alert days."
                                )
                            else:
                                st.info("No daily mean exceeded the selected period's 1.5×IQR upper fence.")
                        else:
                            st.caption("Outlier screening needs at least 14 days with valid daily AQI averages; choose 30, 60 or 90 days for a more useful comparison.")

                        # Compare two actual seven-calendar-day windows. The older
                        # implementation compared the latest 7 valid rows, which
                        # could span more than 7 calendar days when data was missing.
                        if not valid_daily_history.empty:
                            latest_date = pd.Timestamp(valid_daily_history["date"].max()).date()
                            latest_start = latest_date - pd.Timedelta(days=6)
                            previous_end = latest_start - pd.Timedelta(days=1)
                            previous_start = previous_end - pd.Timedelta(days=6)
                            latest_window = valid_daily_history[
                                (valid_daily_history["date"] >= latest_start)
                                & (valid_daily_history["date"] <= latest_date)
                            ]
                            previous_window = valid_daily_history[
                                (valid_daily_history["date"] >= previous_start)
                                & (valid_daily_history["date"] <= previous_end)
                            ]
                            if len(latest_window) >= 4 and len(previous_window) >= 4:
                                last_seven = float(latest_window["mean_us_aqi"].mean())
                                previous_seven = float(previous_window["mean_us_aqi"].mean())
                                delta = float(last_seven - previous_seven)
                                trend_cols = st.columns(3)
                                with trend_cols[0]:
                                    st.metric("Latest 7-calendar-day mean", format_value(last_seven, decimals=0), help=f"Based on {len(latest_window)} days with valid AQI values in {latest_start} to {latest_date}.")
                                with trend_cols[1]:
                                    st.metric("Previous 7-calendar-day mean", format_value(previous_seven, decimals=0), help=f"Based on {len(previous_window)} days with valid AQI values in {previous_start} to {previous_end}.")
                                with trend_cols[2]:
                                    st.metric("Change between periods", f"{delta:+.1f}", help="Positive means the latest calendar window had a higher modelled AQI average. This is descriptive, not a causal explanation.")
                            else:
                                st.caption("The 7-day comparison needs at least four valid daily AQI averages in each of the two calendar windows.")

                        # Day 10: identify consecutive elevated-AQI sample runs.
                        st.markdown("##### Elevated AQI episode detector")
                        st.caption(
                            "Find uninterrupted runs of archived model samples above a chosen threshold. "
                            "The detector adapts to the archive's typical sample interval and never bridges missing timestamps."
                        )
                        episode_controls = st.columns(2, gap="large")
                        with episode_controls[0]:
                            episode_threshold = st.slider(
                                "Modelled US AQI threshold", min_value=51, max_value=301,
                                value=101, step=10, key="episode_threshold_day10",
                                help="A statistical threshold for exploring this archive, not an official alert setting."
                            )
                        with episode_controls[1]:
                            episode_min_samples = st.slider(
                                "Minimum consecutive samples", min_value=2, max_value=12,
                                value=3, step=1, key="episode_min_samples_day10",
                                help="Uses the typical timestamp interval reported for this archived dataset."
                            )

                        episode_frame, episode_cadence = detect_elevated_aqi_episodes(
                            history_frame, threshold=float(episode_threshold),
                            min_samples=int(episode_min_samples),
                        )
                        episode_metric_cols = st.columns(3)
                        with episode_metric_cols[0]:
                            st.metric("Detected runs", len(episode_frame))
                        with episode_metric_cols[1]:
                            st.metric("Typical sample interval", episode_cadence)
                        with episode_metric_cols[2]:
                            peak_episode = (
                                format_value(episode_frame["peak_us_aqi"].max(), decimals=0)
                                if not episode_frame.empty else "—"
                            )
                            st.metric("Highest run peak", peak_episode)

                        if episode_frame.empty:
                            st.info("No uninterrupted runs met the current threshold and minimum-sample settings. This can also happen when the archive has sparse or missing timestamps.")
                        else:
                            episode_display = episode_frame.rename(columns={
                                "start_time": "Start (local)",
                                "end_time": "End (local)",
                                "samples": "Samples",
                                "sample_interval": "Sample interval",
                                "observed_span_hours": "Span between samples (hours)",
                                "mean_us_aqi": "Mean AQI",
                                "peak_us_aqi": "Peak AQI",
                                "peak_pm2_5": "Peak PM2.5 (µg/m³)",
                                "peak_pm10": "Peak PM10 (µg/m³)",
                            })
                            st.dataframe(
                                episode_display, use_container_width=True, hide_index=True,
                                column_config={
                                    "Mean AQI": st.column_config.NumberColumn(format="%.1f"),
                                    "Peak AQI": st.column_config.NumberColumn(format="%.0f"),
                                    "Peak PM2.5 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                                    "Peak PM10 (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                                    "Span between samples (hours)": st.column_config.NumberColumn(format="%.1f"),
                                },
                            )
                            st.download_button(
                                "Download elevated-AQI runs (CSV)",
                                data=episode_frame.to_csv(index=False).encode("utf-8-sig"),
                                file_name=f"airaware_{city_slug}_elevated_aqi_runs.csv",
                                mime="text/csv",
                                key="download_elevated_aqi_runs_day10",
                            )
                        st.caption(
                            f"Method: US AQI ≥ {episode_threshold}, at least {episode_min_samples} consecutive samples, "
                            f"with the archive's modal interval ({episode_cadence}). These are exploratory archived-model runs, "
                            "not official health alerts, verified exposure periods, or medical guidance."
                        )

                        analysis_note = build_historical_analysis_note(
                            city=selected_name,
                            region=selected_region_for_data,
                            country=selected_country_for_data,
                            period_days=int(history_days_choice),
                            frame=history_frame,
                            episodes=episode_frame,
                            threshold=float(episode_threshold),
                            minimum_samples=int(episode_min_samples),
                            cadence_label=episode_cadence,
                        )
                        st.download_button(
                            "Download historical analysis note (Markdown)",
                            data=analysis_note.encode("utf-8"),
                            file_name=f"airaware_{city_slug}_{history_days_choice}d_analysis_note.md",
                            mime="text/markdown",
                            key="download_historical_analysis_note_day10",
                        )

                        history_export = history_frame.copy()
                        history_export["aqi_category"] = history_export["us_aqi"].apply(aqi_category)
                        history_export = history_export.rename(columns={
                            "time": "local_timestamp",
                            "us_aqi": "us_aqi_archived_model_output",
                            "pm2_5": "pm2_5_ug_m3_model_output",
                            "pm10": "pm10_ug_m3_model_output",
                        })
                        st.download_button(
                            "Download historical archive (CSV)",
                            data=history_export.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"airaware_{city_slug}_{history_days_choice}d_archived_model_output.csv",
                            mime="text/csv",
                            key="download_historical_archive_day8_9",
                        )
                        st.caption(
                            "Research limitation: this is archived atmospheric-model output supplied by the Air Quality API, "
                            "not independent ground-monitor observations. Use it for exploratory model-output patterns, "
                            "not as verified ground truth for evaluating a new AQI prediction model."
                        )
                    else:
                        st.info("Historical records were returned, but no valid AQI values were available for the selected period.")
                else:
                    st.info("No historical archive has been loaded for this location and period yet. Choose a period and select Load historical analysis.")
            else:
                st.info("Choose a historical period and select Load historical analysis to retrieve archived data. The request is made only when you press the button.")

        else:
            st.info("The city was selected, but hourly air-quality forecast data is temporarily unavailable.")
            if hourly_error:
                with st.expander("Technical details"):
                    st.code(hourly_error)
        if weather_error:
            st.caption("Current weather data is temporarily unavailable for this location.")

st.markdown("---")
st.markdown('<div class="air-section-label">WORLD CITY COMPARISON</div>', unsafe_allow_html=True)
st.subheader("Air quality in major cities")
st.caption("A weather-app-style comparison of selected major cities across continents. Values are modelled estimates and the list is a representative selection, not every city worldwide.")

with st.spinner("Loading comparison city forecasts..."):
    comparison_points = [(lat, lon) for _, _, lat, lon in FEATURED_CITIES]
    comparison_frame, comparison_failures = collect_samples(comparison_points, chunk_size=36)

if not comparison_frame.empty:
    city_lookup = {
        (round(float(lat), 4), round(float(lon), 4)): (name, country)
        for name, country, lat, lon in FEATURED_CITIES
    }
    comparison_frame = comparison_frame.copy()
    comparison_frame["_lookup"] = comparison_frame.apply(
        lambda row: city_lookup.get((
            round(float(row["requested_latitude"]), 4),
            round(float(row["requested_longitude"]), 4),
        ), ("Unknown location", "Unknown country")),
        axis=1,
    )
    comparison_frame["city"] = comparison_frame["_lookup"].apply(lambda item: item[0])
    comparison_frame["country"] = comparison_frame["_lookup"].apply(lambda item: item[1])
    comparison_frame = comparison_frame.drop(columns=["_lookup"])
    comparison_view = comparison_frame[["city", "country", "us_aqi", "pm2_5", "pm10"]].copy()
    render_comparison_table(comparison_view)
    if comparison_failures:
        st.warning(f"{comparison_failures} comparison-data request batch(es) failed. Missing values are shown as unavailable.")

    # Benchmark the selected location against this fixed representative sample.
    # Exclude the same city/country pair to avoid comparing a value with itself.
    if selected_aqi_for_benchmark is not None:
        benchmark = comparison_view.dropna(subset=["us_aqi"]).copy()
        if selected_city_for_benchmark and selected_country_for_benchmark:
            same_location = (
                benchmark["city"].str.casefold().eq(selected_city_for_benchmark.casefold())
                & benchmark["country"].str.casefold().eq(selected_country_for_benchmark.casefold())
            )
            benchmark = benchmark.loc[~same_location]
        if len(benchmark) >= 3:
            values = pd.to_numeric(benchmark["us_aqi"], errors="coerce").dropna()
            worse_count = int((values > selected_aqi_for_benchmark).sum())
            rank = worse_count + 1
            sample_size = len(values) + 1
            lower_share = float((values < selected_aqi_for_benchmark).mean() * 100.0)
            st.markdown("#### Selected city vs worldwide sample")
            bench_a, bench_b, bench_c = st.columns(3)
            with bench_a:
                st.metric("AQI rank in sample", f"{rank} / {sample_size}", help="Ranked from highest modelled AQI to lowest; higher AQI is worse on the US AQI scale.")
            with bench_b:
                st.metric("Higher than", f"{lower_share:.0f}%", help="Percentage of available comparison cities with a lower modelled US AQI.")
            with bench_c:
                st.metric("Cities compared", len(values))
            st.caption("This benchmark compares modelled AQI values from representative city coordinates. It is not a ranking of every city worldwide or an official public-health comparison.")
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
