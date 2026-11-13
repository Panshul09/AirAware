# AirAware — Data, map and third-party attribution

This file records the upstream sources used by AirAware. Keep the relevant attribution visible in the app and in the repository. Re-check the linked terms before public launch because provider policies can change.

## Air quality and weather data

**Source:** [Open-Meteo](https://open-meteo.com/), including the [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) and [Weather API](https://open-meteo.com/en/docs).

Air-quality modelling is based on datasets from the [Copernicus Atmosphere Monitoring Service (CAMS)](https://atmosphere.copernicus.eu/).

Suggested attribution:

> Air-quality and weather data by [Open-Meteo](https://open-meteo.com/). Air-quality modelling based on [Copernicus Atmosphere Monitoring Service](https://atmosphere.copernicus.eu/) data.

Open-Meteo states that free API access is for **non-commercial use**, with request limits and attribution conditions. Review the [current terms](https://open-meteo.com/en/terms) before launch. If the project becomes commercial or exceeds the applicable limits, obtain the appropriate access or use another compliant provider.

Open-Meteo data is offered under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), subject to the provider's terms and relevant source-data conditions. Indicate that data may have been transformed or aggregated by AirAware.

## City directory

**Source:** [DataHub world-cities](https://github.com/datasets/world-cities), sourced from [GeoNames](https://www.geonames.org/).

Suggested attribution:

> City directory: DataHub world-cities dataset, sourced from GeoNames. Data is used to populate the location selector. Data is licensed under CC BY 4.0; credit GeoNames and the DataHub dataset.

The dataset repository identifies CC BY 4.0 as its data licence. Preserve attribution and review the upstream repository for any changes.

## Map tiles

**Source:** [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright).

The interactive map uses OpenStreetMap basemap tiles. Keep the attribution visible on the map. The public tile servers are a community service with best-effort availability, not an unlimited free tile API. Follow the [OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/), including the correct HTTPS tile URL, attribution, caching behaviour, and no bulk tile downloading or prefetching.

## Software dependencies

AirAware uses third-party open-source packages listed in `requirements.txt`, including Streamlit, Requests, pandas, NumPy, Plotly, pycountry, and Folium. Each package retains its own licence. See the package metadata and upstream project documentation for licence details.

## Project-code licence is separate

The original AirAware source code in this repository is licensed under the [MIT License](LICENSE). The MIT License applies only to the project's own code and documentation. It does not grant rights to third-party data, map tiles, API content, or software dependencies; their respective licences and service terms continue to apply.
