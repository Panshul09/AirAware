
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------
# AIR AWARE - DAY 1
# Environmental intelligence for Pakistani cities
# --------------------------------------------------

st.set_page_config(
    page_title="AirAware | Environmental Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Coordinates are used to request model data for each city.
CITIES = {
    "Hyderabad, Sindh": {"latitude": 25.3960, "longitude": 68.3578},
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011},
    "Lahore": {"latitude": 31.5204, "longitude": 74.3587},
    "Islamabad": {"latitude": 33.6844, "longitude": 73.0479},
    "Peshawar": {"latitude": 34.0151, "longitude": 71.5249},
    "Quetta": {"latitude": 30.1798, "longitude": 66.9750},
    "Multan": {"latitude": 30.1575, "longitude": 71.5249},
    "Faisalabad": {"latitude": 31.4504, "longitude": 73.1350},
}

TIMEZONE = "Asia/Karachi"


# --------------------------------------------------
# MODERN UI STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 80% 0%,
            rgba(25, 100, 105, 0.20), transparent 35%),
            #080E18;
        color: #EAF0F8;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1725, #090F19);
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    [data-testid="stMetric"] {
        background: rgba(18, 29, 45, 0.85);
        border: 1px solid rgba(150, 175, 205, 0.14);
        border-radius: 16px;
        padding: 19px 20px;
        min-height: 125px;
    }

    [data-testid="stMetricLabel"] {
        color: #A8B9CF;
    }

    [data-testid="stMetricValue"] {
        color: #F4F8FF;
        font-size: 1.8rem;
    }

    [data-testid="stMetricDelta"] {
        color: #9AAFC5;
    }

    h1, h2, h3 {
        color: #F0F5FC;
        letter-spacing: -0.5px;
    }

    p, li, label {
        color: #CBD5E4;
    }

    .air-hero {
        background:
            linear-gradient(120deg,
            rgba(23, 79, 82, 0.76),
            rgba(17, 28, 45, 0.96) 65%);
        border: 1px solid rgba(87, 214, 188, 0.22);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 22px;
    }

    .air-eyebrow {
        color: #72E3C6;
        font-size: 12px;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .air-title {
        font-size: 43px;
        line-height: 1.15;
        font-weight: 800;
        color: #F7FAFF;
        margin-bottom: 10px;
    }

    .air-subtitle {
        font-size: 15px;
        color: #C5D4E4;
        max-width: 700px;
        line-height: 1.7;
    }

    .air-status {
        background: rgba(17, 29, 44, 0.90);
        border: 1px solid rgba(150, 175, 205, 0.17);
        border-left: 4px solid #65D9B9;
        border-radius: 16px;
        padding: 21px 23px;
        margin: 8px 0 22px 0;
    }

    .air-status-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        color: #9FB4CA;
    }

    .air-status-title {
        font-size: 25px;
        font-weight: 750;
        margin: 6px 0;
    }

    .air-muted {
        font-size: 13px;
        color: #9EADC2;
        line-height: 1.7;
    }

    .section-kicker {
        color: #6ADDC0;
        letter-spacing: 1.8px;
        font-size: 11px;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .info-panel {
        border-radius: 16px;
        border: 1px solid rgba(150, 175, 205, 0.14);
        background: rgba(16, 27, 42, 0.86);
        padding: 20px;
        min-height: 126px;
    }

    div.stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(87, 214, 188, 0.45);
        background: rgba(37, 106, 99, 0.25);
        color: #E8FFF8;
        min-height: 43px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border-color: #72E3C6;
        color: #FFFFFF;
        background: rgba(37, 106, 99, 0.48);
    }

    hr {
        border-color: rgba(150, 175, 205, 0.15);
    }

    a {
        color: #72E3C6 !important;
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(150, 175, 205, 0.12);
        border-radius: 16px;
        overflow: hidden;
        background: rgba(15, 25, 39, 0.58);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# API FUNCTIONS
# --------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_air_quality(latitude, longitude):
    """Fetch five days of modelled air-quality forecasts."""

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm2_5,pm10,us_aqi,european_aqi,"
            "ozone,nitrogen_dioxide"
        ),
        "forecast_days": 5,
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload or "time" not in payload["hourly"]:
        raise ValueError("The air-quality API returned unexpected data.")

    data = pd.DataFrame(payload["hourly"])
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).reset_index(drop=True)

    for column in [
        "pm2_5",
        "pm10",
        "us_aqi",
        "european_aqi",
        "ozone",
        "nitrogen_dioxide",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.empty:
        raise ValueError("No hourly air-quality data was returned.")

    return data


@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(latitude, longitude):
    """Fetch current modelled weather conditions."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,"
            "apparent_temperature,wind_speed_10m,precipitation"
        ),
        "timezone": TIMEZONE,
    }

    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()

    if "current" not in payload:
        raise ValueError("The weather API returned unexpected data.")

    return payload["current"]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def number_or_none(value):
    """Convert a value to a number, safely handling missing data."""

    if value is None or pd.isna(value):
        return None

    return float(value)


def format_measurement(value, suffix="", decimals=1):
    """Format a measurement or show an unavailable-data marker."""

    if value is None:
        return "—"

    return f"{value:.{decimals}f}{suffix}"


def aqi_information(value):
    """Return a category and general information using the US AQI scale."""

    if value is None:
        return (
            "Unavailable",
            "#94A3B8",
            "An AQI estimate is not available for this time.",
        )

    if value <= 50:
        return (
            "Good",
            "#22C55E",
            "The estimate is in the Good band of the US AQI scale.",
        )

    if value <= 100:
        return (
            "Moderate",
            "#EAB308",
            "Air quality is in the Moderate band. Sensitive people "
            "may prefer to monitor conditions.",
        )

    if value <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#FB923C",
            "People sensitive to air pollution may wish to reduce "
            "prolonged or strenuous outdoor activity.",
        )

    if value <= 200:
        return (
            "Unhealthy",
            "#F87171",
            "Check trusted local public-health guidance and follow "
            "updates as conditions change.",
        )

    if value <= 300:
        return (
            "Very Unhealthy",
            "#C084FC",
            "The estimate indicates very high pollution. Follow "
            "trusted public-health guidance.",
        )

    return (
        "Hazardous",
        "#F43F5E",
        "The estimate is in the highest US AQI category. "
        "Consult trusted local public-health guidance.",
    )


def format_time(timestamp):
    """Format a timestamp for the local Pakistan time zone."""

    if pd.isna(timestamp):
        return "Time unavailable"

    return timestamp.strftime("%d %b %Y · %I:%M %p")


def style_figure(fig, height=340):
    """Apply consistent styling to all Plotly charts."""

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#C9D5E5",
            family="Inter, Segoe UI, sans-serif",
            size=12,
        ),
        margin=dict(l=12, r=15, t=48, b=12),
        height=height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
        title_text="",
        tickformat="%d %b · %H:%M",
    )

    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
    )

    return fig


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size: 31px;">🌿</div>
        <div style="font-size: 23px; font-weight: 800; color: #F4F8FF;">
            AirAware
        </div>
        <div style="font-size: 12px; color: #93A7BE; margin-top: 5px;">
            Environmental intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Your location")

city = st.sidebar.selectbox(
    "Select a Pakistani city",
    options=list(CITIES.keys()),
    index=0,
)

coordinates = CITIES[city]

st.sidebar.caption(
    f"Latitude: {coordinates['latitude']} · "
    f"Longitude: {coordinates['longitude']}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Forecast settings")

chart_window = st.sidebar.radio(
    "Chart time range",
    options=["Next 24 hours", "Next 48 hours", "Next 5 days"],
    index=1,
)

if st.sidebar.button(
    "↻ Refresh data",
    use_container_width=True,
    type="primary",
):
    fetch_air_quality.clear()
    fetch_weather.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About AirAware")
st.sidebar.caption(
    "A public-interest environmental data project. "
    "This first version uses openly accessible model forecasts."
)

st.sidebar.markdown(
    """
    <div style="font-size: 11px; color: #8295AC; line-height: 1.7;">
        DATA SOURCES<br>
        Open-Meteo Air Quality API<br>
        Open-Meteo Weather API
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("Fetching environmental data..."):
    try:
        air_df = fetch_air_quality(
            coordinates["latitude"],
            coordinates["longitude"],
        )

        weather = fetch_weather(
            coordinates["latitude"],
            coordinates["longitude"],
        )

    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(
            "We couldn't retrieve the environmental data. "
            "Check your internet connection and try refreshing."
        )

        with st.expander("Technical details"):
            st.code(str(error))

        st.stop()


# Find the modelled hour closest to the current local time.
now_local = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

valid_aqi = air_df.dropna(subset=["us_aqi"])

if not valid_aqi.empty:
    nearest_index = (
        valid_aqi["time"] - now_local
    ).abs().idxmin()

    current_row = valid_aqi.loc[nearest_index]
else:
    nearest_index = (
        air_df["time"] - now_local
    ).abs().idxmin()

    current_row = air_df.loc[nearest_index]

current_aqi = number_or_none(current_row.get("us_aqi"))
current_pm25 = number_or_none(current_row.get("pm2_5"))
current_pm10 = number_or_none(current_row.get("pm10"))

category, category_color, category_message = aqi_information(current_aqi)

temperature = number_or_none(weather.get("temperature_2m"))
humidity = number_or_none(weather.get("relative_humidity_2m"))
wind_speed = number_or_none(weather.get("wind_speed_10m"))

# Prepare the future portion of the model forecast for the charts.
future_df = air_df[
    air_df["time"] >= now_local.floor("h")
].copy()

if future_df.empty:
    future_df = air_df.copy()

hour_limits = {
    "Next 24 hours": 24,
    "Next 48 hours": 48,
    "Next 5 days": 120,
}

chart_df = future_df.head(hour_limits[chart_window]).copy()


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------

st.markdown(
    f"""
    <div class="air-hero">
        <div class="air-eyebrow">
            ENVIRONMENTAL INTELLIGENCE · PAKISTAN
        </div>
        <div class="air-title">
            Breathe smarter.
        </div>
        <div class="air-subtitle">
            Understand air quality, explore pollution trends, and
            see how environmental conditions may change over time.
            <br><br>
            <strong style="color:#72E3C6;">⌖ {city}</strong>
            &nbsp; · &nbsp; Modelled forecast data
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# AIR QUALITY STATUS
# --------------------------------------------------

st.markdown(
    '<div class="section-kicker">CURRENT OUTLOOK</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="air-status"
         style="border-left-color: {category_color};">
        <div class="air-status-label">
            MODELLED US AIR QUALITY INDEX
        </div>
        <div class="air-status-title"
             style="color: {category_color};">
            {category}
        </div>
        <div class="air-muted">
            {category_message}
        </div>
        <div class="air-muted" style="margin-top:10px;">
            Nearest model hour: {format_time(current_row["time"])}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------

st.markdown(
    '<div class="section-kicker">ENVIRONMENTAL SNAPSHOT</div>',
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "US AQI · MODELLED",
        format_measurement(current_aqi, decimals=0),
        help=(
            "An air-quality estimate on the US AQI scale. "
            "This is not a direct monitoring-station reading."
        ),
    )

with metric2:
    st.metric(
        "PM2.5",
        format_measurement(current_pm25, " µg/m³"),
        help=(
            "Modelled fine particulate matter concentration "
            "in micrograms per cubic metre."
        ),
    )

with metric3:
    st.metric(
        "PM10",
        format_measurement(current_pm10, " µg/m³"),
        help=(
            "Modelled particulate matter concentration "
            "in micrograms per cubic metre."
        ),
    )

with metric4:
    st.metric(
        "Temperature",
        format_measurement(temperature, " °C"),
        help="Current modelled weather temperature.",
    )


# --------------------------------------------------
# WEATHER CONTEXT
# --------------------------------------------------

st.markdown("---")
st.markdown("## Weather context")
st.caption(
    "Weather provides additional context for studying "
    "environmental conditions; it does not explain every pollution change."
)

weather1, weather2, weather3 = st.columns(3)

with weather1:
    st.metric(
        "Relative humidity",
        format_measurement(humidity, " %"),
    )

with weather2:
    st.metric(
        "Wind speed",
        format_measurement(wind_speed, " km/h"),
    )

with weather3:
    apparent_temperature = number_or_none(
        weather.get("apparent_temperature")
    )

    st.metric(
        "Feels-like temperature",
        format_measurement(apparent_temperature, " °C"),
    )


# --------------------------------------------------
# FORECAST CHARTS
# --------------------------------------------------

st.markdown("---")
st.markdown("## Explore the forecast")
st.caption(
    f"Showing {chart_window.lower()} for {city}. "
    "These values are model estimates, not confirmed measurements."
)

left_chart, right_chart = st.columns(2, gap="large")

with left_chart:
    st.markdown("### US AQI outlook")

    aqi_fig = go.Figure()

    aqi_fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["us_aqi"],
            mode="lines+markers",
            name="US AQI",
            line=dict(color="#65D9B9", width=3),
            marker=dict(size=5, color="#65D9B9"),
            fill="tozeroy",
            fillcolor="rgba(101, 217, 185, 0.08)",
            connectgaps=False,
            hovertemplate=(
                "%{x|%d %b, %H:%M}<br>"
                "US AQI: %{y:.0f}<extra></extra>"
            ),
        )
    )

    for threshold in [50, 100, 150, 200, 300]:
        aqi_fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_width=1,
            line_color="rgba(200, 215, 235, 0.20)",
        )

    aqi_fig.update_layout(
        title="Hourly US AQI estimate",
        yaxis_title="US AQI",
    )

    style_figure(aqi_fig)

    st.plotly_chart(aqi_fig, use_container_width=True)

    st.caption(
        "AQI category bands follow the US AQI scale. "
        "Local reporting standards may differ."
    )


with right_chart:
    st.markdown("### Particulate matter")

    pm_fig = go.Figure()

    pm_fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["pm2_5"],
            mode="lines",
            name="PM2.5",
            line=dict(color="#65D9B9", width=3),
            connectgaps=False,
            hovertemplate=(
                "%{x|%d %b, %H:%M}<br>"
                "PM2.5: %{y:.1f} µg/m³<extra></extra>"
            ),
        )
    )

    pm_fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["pm10"],
            mode="lines",
            name="PM10",
            line=dict(color="#8BB8FF", width=3),
            connectgaps=False,
            hovertemplate=(
                "%{x|%d %b, %H:%M}<br>"
                "PM10: %{y:.1f} µg/m³<extra></extra>"
            ),
        )
    )

    pm_fig.update_layout(
        title="Hourly pollutant concentration",
        yaxis_title="Concentration (µg/m³)",
    )

    style_figure(pm_fig)

    st.plotly_chart(pm_fig, use_container_width=True)


# --------------------------------------------------
# DAILY SUMMARY
# --------------------------------------------------

st.markdown("---")
st.markdown("## Five-day overview")
st.caption(
    "Compare the highest modelled US AQI estimate for each "
    "forecast day. These daily maxima may include incomplete days."
)

daily = air_df.copy()
daily["day"] = daily["time"].dt.strftime("%a, %d %b")

daily_summary = (
    daily.groupby("day", sort=False, as_index=False)
    .agg(max_aqi=("us_aqi", "max"))
    .dropna(subset=["max_aqi"])
)

if daily_summary.empty:
    st.info("Daily AQI data is unavailable for this location.")
else:
    daily_fig = go.Figure()

    daily_fig.add_trace(
        go.Bar(
            x=daily_summary["day"],
            y=daily_summary["max_aqi"],
            name="Daily maximum US AQI",
            marker=dict(
                color="#54CBB0",
                line=dict(color="#7AE6CD", width=1),
            ),
            hovertemplate=(
                "%{x}<br>Maximum modelled AQI: "
                "%{y:.0f}<extra></extra>"
            ),
        )
    )

    daily_fig.update_layout(
        title="Daily maximum AQI estimate",
        xaxis_title="Forecast day",
        yaxis_title="US AQI",
        hovermode="x",
        height=310,
    )

    style_figure(daily_fig, height=310)

    st.plotly_chart(daily_fig, use_container_width=True)


# --------------------------------------------------
# PROJECT INFORMATION AND ATTRIBUTION
# --------------------------------------------------

st.markdown("---")
st.markdown("## Understanding the data")

info1, info2 = st.columns(2, gap="large")

with info1:
    st.markdown(
        """
        <div class="info-panel">
            <div style="font-weight:700; color:#EAF3FF;
                        margin-bottom:9px;">
                What is PM2.5?
            </div>
            <div class="air-muted">
                PM2.5 describes fine airborne particles with
                diameters of 2.5 micrometres or less. Its
                concentration is expressed in µg/m³.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info2:
    st.markdown(
        """
        <div class="info-panel">
            <div style="font-weight:700; color:#EAF3FF;
                        margin-bottom:9px;">
                What is AQI?
            </div>
            <div class="air-muted">
                The US Air Quality Index is a communication
                scale used to describe air pollution levels.
                Higher values indicate greater air-quality
                concern on that scale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="air-muted" style="margin-top:20px;">
        <strong>Data limitations:</strong>
        This application displays modelled air-quality forecasts,
        not readings from verified local monitoring stations.
        Coverage, model uncertainty, weather, and local conditions
        can affect accuracy. Do not treat the forecast as a medical
        assessment or an official local air-quality warning.
        <br><br>
        Air-quality forecasts:
        <a href="https://open-meteo.com/" target="_blank">
            Open-Meteo
        </a>
        · Air-quality modelling based on:
        <a href="https://atmosphere.copernicus.eu/"
           target="_blank">
            Copernicus Atmosphere Monitoring Service
        </a>
        <br>
        Weather data:
        <a href="https://open-meteo.com/"
           target="_blank">
            Open-Meteo
        </a>
        · Open-Meteo data attribution:
        <a href="https://creativecommons.org/licenses/by/4.0/"
           target="_blank">
            CC BY 4.0
        </a>
        <br><br>
        AirAware · Day 1 prototype · Built with Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
