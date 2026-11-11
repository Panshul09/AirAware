# AirAware — Release Candidate QA Build

AirAware is a Streamlit application for exploring global modelled air-quality forecasts, weather context, archived atmospheric-model output, and city comparisons.

This update is a **feature-freeze and reliability pass**. It does not add a new user-facing feature. It focuses on reducing avoidable API failures, safer data alignment, input validation, and a repeatable local smoke-test suite before public beta preparation.

## What's hardened

- Bounded retries for transient HTTP/network failures (timeouts, connection errors, 408/425/429/5xx responses).
- API-level JSON errors and invalid JSON are converted into readable exceptions.
- City-directory CSV download now has an explicit timeout and uses the same retry handler.
- Multi-coordinate API batches are rejected if the returned item count does not match the requested coordinate count, rather than risking city/value mismatches.
- Invalid or non-finite interpolation samples and invalid raster bounds are rejected safely.
- Row-aligned hourly changes use row positions, preventing calculations from attaching to the wrong record when indexes are duplicated or unordered.
- A single timestamp no longer reports 100% interval continuity when there is no interval to assess.
- Selected API timezones are validated with a UTC fallback.
- Refresh clears the archived air-quality cache as well as current forecast/weather caches.

## Requirements

- Python 3.10 or newer recommended.
- Internet access for the city directory, geocoding, Open-Meteo data and OpenStreetMap tiles.
- No personal mapping API key is required for the OpenStreetMap basemap.

## Install on Windows with VS Code

1. Stop the running Streamlit app with `Ctrl+C`.
2. Copy `app.py`, `requirements.txt`, `qa_smoke_test.py`, and the `.streamlit` folder into your existing `AirAware` project folder.
3. Keep the existing `.git` and `.venv` folders. Do not replace either one.
4. Open a new terminal in the AirAware folder and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Run the local QA smoke tests

With the AirAware virtual environment active (or using its Python executable), run:

```powershell
.\.venv\Scripts\python.exe qa_smoke_test.py
```

The script tests pure helper functions extracted from `app.py` without importing the Streamlit page (which otherwise starts the app and makes network requests). It checks numeric handling, AQI boundaries, daily aggregation, timestamp alignment, data-quality auditing, contiguous windows, elevated-AQI episodes, research joins, raster generation, map HTML generation, timezone fallback, and request retry behavior.

These tests do **not** replace manual browser testing or live API testing.

## Data sources and interpretation

- Air quality and weather: Open-Meteo APIs; air-quality modelling is based on Copernicus Atmosphere Monitoring Service datasets.
- City directory: DataHub `world-cities` dataset based on GeoNames.
- Basemap tiles: OpenStreetMap contributors.

AirAware's forecasts and archived values are atmospheric-model output, not independent ground-monitor observations. The map heat layer interpolates sparse representative city samples and is a visualization surface, not a measurement at every pixel. Correlations are exploratory associations, not causal findings. Do not use AirAware as an official public-health warning or medical assessment.

Check the upstream providers' current attribution, usage and licensing terms before public deployment. In particular, keep visible OpenStreetMap attribution and observe its tile usage policy.

## Release status

This is a local release-candidate QA build. It is not yet a public production release. Before deployment, manually verify the app in a browser, test the external APIs from the deployment environment, inspect mobile layout, confirm attribution, and ensure no secrets or local environment folders are committed.
