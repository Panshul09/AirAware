# AirAware — Public Beta Readiness Checklist

## Local functionality

- [ ] App starts from a clean terminal using the documented command.
- [ ] Global map loads with OpenStreetMap attribution.
- [ ] Global heat layer appears when usable samples are returned.
- [ ] Country → region → city selectors work; initial location is blank.
- [ ] Selecting another city updates the map and selected-location panels.
- [ ] Hourly and daily forecast sections both render.
- [ ] Current weather and hourly weather data failures are shown gracefully.
- [ ] Forecast data-quality audit handles missing and duplicate timestamps.
- [ ] Forecast CSV and combined research CSV download and open correctly.
- [ ] Historical Explorer loads the selected archive period.
- [ ] Elevated-AQI detector does not bridge missing time intervals.
- [ ] Episode CSV, historical CSV and Markdown analysis note download correctly.
- [ ] Environmental relationship plots handle insufficient data without crashing.
- [ ] Global city comparison retains correct city/country labels after partial batch failures.
- [ ] No clipped labels, dark-on-dark controls, broken map frame, or unusable small-screen layout.

## Data and scientific interpretation

- [ ] Forecasts and archived model output are clearly labelled as modelled estimates.
- [ ] Map interpolation is not presented as sensor-level or pixel-level observation.
- [ ] Correlations are described as associations, not causes.
- [ ] No machine-learning accuracy claim is made without independent observations and time-based validation.
- [ ] Missing or failed API data is shown as unavailable, never silently converted to zero.

## Security and repository hygiene

- [ ] `.venv/` is not committed.
- [ ] `.streamlit/secrets.toml`, `.env`, and credentials are not committed.
- [ ] `git status` contains only intended source, config, documentation and test files.
- [ ] Dependency versions are reviewed and reproducible for deployment.
- [ ] No private location, user data or credentials are stored in the repository.

## Public deployment

- [ ] README has setup instructions, screenshots, features, data sources and limitations.
- [ ] OpenStreetMap and data-provider attribution is visible.
- [ ] Provider usage limits and terms are reviewed.
- [ ] Deployment succeeds from a clean GitHub checkout.
- [ ] Public URL is tested in a private/incognito browser session.
- [ ] At least a few external testers complete the key user flows and provide feedback.
- [ ] Critical bugs are fixed before announcing version 1.0.
