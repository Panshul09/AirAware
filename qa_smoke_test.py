from __future__ import annotations

import ast
import math
from pathlib import Path
import unittest
from typing import Any
from unittest.mock import Mock, patch
import requests

import folium
import numpy as np
import pandas as pd
from branca.element import Element
from folium.raster_layers import ImageOverlay

APP_PATH = Path(__file__).with_name("app.py")

FUNCTIONS = {
    "safe_number", "aqi_category", "format_value", "safe_timezone_name",
    "aggregate_daily_forecast", "summarize_historical_air_quality",
    "infer_sample_interval", "detect_elevated_aqi_episodes",
    "audit_forecast_data", "consecutive_hour_changes",
    "find_best_contiguous_window", "build_research_dataset",
    "interpolate_raster", "build_folium_map", "request_with_retry", "request_json", "fetch_current_batch",
}


def load_app_helpers():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(APP_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if any(name in {"HEAT_STOPS", "HEAT_RGB", "TIMEZONE_DEFAULT", "AQI_COLORS", "RETRYABLE_HTTP_STATUS"} for name in names):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            node.decorator_list = []  # Avoid requiring Streamlit just to test pure helpers.
            selected.append(node)
    namespace = {
        "pd": pd, "np": np, "math": math, "Any": Any,
        "folium": folium, "Element": Element, "ImageOverlay": ImageOverlay,
        "requests": requests, "time": __import__("time"),
        "AIR_URL": "https://example.invalid/air-quality",
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


H = load_app_helpers()


class AirAwareQualityChecks(unittest.TestCase):
    def test_finite_number_handling(self):
        self.assertEqual(H["safe_number"]("12.5"), 12.5)
        self.assertIsNone(H["safe_number"](float("nan")))
        self.assertIsNone(H["safe_number"](float("inf")))
        self.assertIsNone(H["safe_number"](None))

    def test_aqi_category_boundaries(self):
        self.assertEqual(H["aqi_category"](50), "Good")
        self.assertEqual(H["aqi_category"](51), "Moderate")
        self.assertEqual(H["aqi_category"](101), "Unhealthy for sensitive groups")
        self.assertEqual(H["aqi_category"](301), "Hazardous")
        self.assertEqual(H["aqi_category"](None), "No data")

    def test_daily_forecast_deduplicates_and_groups(self):
        times = list(pd.date_range("2026-11-01 00:00", periods=26, freq="h"))
        frame = pd.DataFrame({
            "time": times + [times[0]],
            "us_aqi": list(range(26)) + [1000],
            "pm2_5": list(range(26)) + [999],
            "pm10": list(range(26)) + [999],
        })
        result = H["aggregate_daily_forecast"](frame, today=pd.Timestamp("2026-11-01").date())
        self.assertEqual(len(result), 2)
        self.assertEqual(int(result.iloc[0]["available_hours"]), 24)
        self.assertAlmostEqual(float(result.iloc[0]["mean_us_aqi"]), 53.1666666667, places=6)

    def test_consecutive_changes_are_row_aligned_with_duplicate_index(self):
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-11-01 02:00", "2026-11-01 00:00", "2026-11-01 01:00"]),
                "aqi": [30, 10, 20],
            },
            index=[7, 7, 3],
        )
        result = H["consecutive_hour_changes"](frame, "aqi")
        self.assertEqual(len(result), 3)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertEqual(float(result.iloc[2]), 10.0)
        self.assertEqual(float(result.iloc[0]), 10.0)

    def test_changes_do_not_bridge_missing_hours(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-11-01 00:00", "2026-11-01 01:00", "2026-11-01 04:00"]),
            "aqi": [10, 20, 70],
        })
        result = H["consecutive_hour_changes"](frame, "aqi")
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertEqual(float(result.iloc[1]), 10.0)
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_best_window_does_not_bridge_gaps(self):
        frame = pd.DataFrame({
            "time": list(pd.date_range("2026-11-01 00:00", periods=4, freq="h"))
                    + list(pd.date_range("2026-11-01 08:00", periods=4, freq="h")),
            "us_aqi": [80, 70, 60, 50, 40, 30, 20, 10],
        })
        result = H["find_best_contiguous_window"](frame, window_size=4)
        self.assertIsNotNone(result)
        self.assertEqual(result["mean_aqi"], 25.0)
        self.assertEqual(result["start"], pd.Timestamp("2026-11-01 08:00"))

    def test_elevated_episodes_split_at_time_gap(self):
        frame = pd.DataFrame({
            "time": list(pd.date_range("2026-11-01 00:00", periods=3, freq="h"))
                    + list(pd.date_range("2026-11-01 06:00", periods=3, freq="h")),
            "us_aqi": [120, 130, 125, 140, 150, 145],
            "pm2_5": [10, 12, 11, 13, 14, 13],
            "pm10": [20, 22, 21, 23, 24, 23],
        })
        episodes, cadence = H["detect_elevated_aqi_episodes"](frame, threshold=101, min_samples=3)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes.iloc[0]["samples"], 3)
        self.assertIn("hour", cadence)

    def test_data_audit_counts_duplicates_gaps_and_invalid_values(self):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-11-01 00:00", "2026-11-01 00:00", "2026-11-01 02:00"]),
            "us_aqi": [25, float("inf"), 520],
            "pm2_5": [10, -1, 12],
            "pm10": [20, 22, 24],
        })
        result = H["audit_forecast_data"](frame)
        self.assertEqual(result["duplicate_timestamps"], 1)
        self.assertEqual(result["missing_hour_intervals"], 1)
        self.assertGreaterEqual(result["invalid_values"], 3)

    def test_single_timestamp_has_unknown_continuity_not_100_percent(self):
        frame = pd.DataFrame({"time": [pd.Timestamp("2026-11-01")], "us_aqi": [25]})
        result = H["audit_forecast_data"](frame)
        self.assertEqual(result["continuity_pct"], 0.0)

    def test_research_dataset_preserves_alignment_and_metadata(self):
        times = pd.date_range("2026-11-01 00:00", periods=3, freq="h")
        air = pd.DataFrame({"time": times, "us_aqi": [10, 20, 30], "pm2_5": [2, 3, 4], "pm10": [4, 5, 6]})
        weather = pd.DataFrame({"time": times, "temperature_2m": [20, 21, 22], "wind_speed_10m": [5, 6, 7]})
        result = H["build_research_dataset"](
            air, weather, city="Test City", region="Region", country="Country",
            latitude=1.0, longitude=2.0, timezone_name="UTC",
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[1, "aqi_change_1h"], 10.0)
        self.assertEqual(result.loc[1, "temperature_2m"], 21.0)
        self.assertTrue((result["city"] == "Test City").all())

    def test_empty_and_invalid_raster_inputs_fail_safely(self):
        self.assertIsNone(H["interpolate_raster"](pd.DataFrame(), (-10, -10, 10, 10), 20, 10, 20))
        samples = pd.DataFrame({"latitude": [0, float("inf")], "longitude": [0, 0], "us_aqi": [55, 80]})
        raster = H["interpolate_raster"](samples, (-2, -2, 2, 2), 24, 12, 20)
        self.assertEqual(raster.shape, (12, 24, 4))
        self.assertEqual(raster.dtype, np.uint8)
        self.assertTrue((raster[:, :, 3] > 0).any())
        self.assertIsNone(H["interpolate_raster"](samples, (2, 2, -2, -2), 24, 12, 20))

    def test_osm_map_is_generated_even_without_heat_samples(self):
        fmap = H["build_folium_map"](
            pd.DataFrame(), center=(20, 0), zoom=2,
            bounds=(-180, -60, 180, 80), smoothing_km=1000,
            raster_width=120, raster_height=60, height=400,
        )
        html_doc = fmap.get_root().render()
        self.assertIn("openstreetmap.org", html_doc.lower())
        self.assertIn("leaflet", html_doc.lower())

    def test_request_helper_retries_transient_http_status(self):
        first = Mock()
        first.status_code = 503
        first.headers = {}
        second = Mock()
        second.status_code = 200
        second.headers = {}
        second.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[first, second]) as mocked_get, \
             patch("time.sleep", return_value=None):
            response = H["request_with_retry"]("https://example.invalid", timeout=2, attempts=2)
        self.assertIs(response, second)
        self.assertEqual(mocked_get.call_count, 2)

    def test_request_helper_does_not_retry_permanent_client_error(self):
        failed = Mock()
        failed.status_code = 400
        failed.headers = {}
        failed.raise_for_status.side_effect = requests.HTTPError("400 client error")
        with patch("requests.get", return_value=failed) as mocked_get:
            with self.assertRaises(requests.HTTPError):
                H["request_with_retry"]("https://example.invalid", timeout=2, attempts=3)
        self.assertEqual(mocked_get.call_count, 1)

    def test_multi_coordinate_response_length_mismatch_is_rejected(self):
        original = H["request_json"]
        H["request_json"] = lambda *args, **kwargs: [{"latitude": 1, "longitude": 2, "current": {}}]
        try:
            with self.assertRaisesRegex(ValueError, "one result per requested coordinate"):
                H["fetch_current_batch"](((1.0, 2.0), (3.0, 4.0)))
        finally:
            H["request_json"] = original

    def test_timezone_fallback(self):
        self.assertEqual(H["safe_timezone_name"]("Not/A_Real_Zone"), "UTC")
        self.assertEqual(H["safe_timezone_name"]("UTC"), "UTC")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AirAwareQualityChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
