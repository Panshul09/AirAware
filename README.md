# AirAware — Global Environmental Intelligence

AirAware is an educational environmental-data application for exploring modelled air-quality forecasts, archived atmospheric-model output, weather context, and city-level comparisons around the world.

> **Important:** AirAware displays atmospheric-model estimates, not verified readings from local ground sensors. Its interpolated heat layer visualises sparse sample locations; it does not represent a direct measurement at every point on the map. AirAware is not an official public-health warning service or a medical assessment.

## Features

- **Global location selection:** choose a country, first-level region, and city.
- **Interactive global map:** OpenStreetMap basemap with an interpolated air-quality layer where valid model samples are available.
- **Hourly and daily forecasts:** explore modelled AQI and particulate-matter trends.
- **Global city comparison:** compare selected locations with a curated set of representative cities.
- **Weather context:** view temperature, humidity, wind, and precipitation alongside air-quality information.
- **Historical explorer:** inspect archived atmospheric-model output over supported periods.
- **Environmental analysis:** explore descriptive patterns, correlations, elevated-AQI periods, and forecast data quality.
- **Research exports:** download forecast and analysis tables as CSV.

## Technology

Python · Streamlit · Pandas · NumPy · Plotly · Folium · Requests

## Run locally

Python 3.10 or newer is recommended. An internet connection is needed for the data services, global city directory, and map tiles.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and start the app:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application opens at the local URL printed by Streamlit, usually `http://localhost:8501`.

## Automated checks

Run the included helper-function smoke tests from the repository root:

```bash
python qa_smoke_test.py
```

These checks use synthetic data and do not replace live API, browser, mobile-layout, or deployment testing.

## Data sources and attribution

- **Air quality and weather:** [Open-Meteo](https://open-meteo.com/). Air-quality modelling uses datasets from the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).
- **City directory:** [DataHub world-cities](https://github.com/datasets/world-cities), sourced from [GeoNames](https://www.geonames.org/).
- **Basemap:** [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright).

See [ATTRIBUTION.md](ATTRIBUTION.md) for source attribution and third-party usage notes. Review the providers' current terms and usage policies before public deployment or any non-personal use. In particular, check Open-Meteo's access terms and OpenStreetMap's public tile usage policy.

## Limitations

1. **Model output is not a sensor observation.** Forecasts and archived values represent atmospheric-model output and can differ from ground conditions.
2. **The map is interpolated.** The heat layer is generated from sparse sample points. Its smooth appearance must not be interpreted as a high-resolution measurement surface.
3. **City comparisons are limited.** Featured cities represent values at selected coordinates, not citywide averages.
4. **Correlation is not causation.** Associations between weather and pollution do not establish causal relationships.
5. **No validated machine-learning claim.** Predictive accuracy should not be claimed until a model is evaluated against independent observations using a time-based holdout.
6. **Not for health decisions.** This application is not an official warning service, medical assessment, or substitute for local public-health guidance.

## Project status

AirAware is a portfolio and research prototype prepared for final quality assurance and deployment testing. Live API behaviour, browser rendering, mobile layout, provider limits, and the hosted deployment should be verified before announcing a production release.

## Licence

The original AirAware source code in this repository is licensed under the [MIT License](LICENSE). This licence applies to the project's own code and documentation; it does **not** relicense third-party datasets, API content, map tiles, or dependencies. Those materials remain subject to their respective licences, attribution requirements, and service terms. See [ATTRIBUTION.md](ATTRIBUTION.md) before reusing or redistributing them.
