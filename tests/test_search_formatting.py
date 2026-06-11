import unittest

from app.tools.search import (
    format_news_results,
    format_search_results,
    format_weather_result,
)


class SearchFormattingTests(unittest.TestCase):
    def test_news_metadata_is_preserved_for_grounding(self):
        output = format_search_results(
            [
                {
                    "title": "Current story",
                    "url": "https://example.com/story",
                    "snippet": "A current report.",
                    "published_at": "2026-06-12T08:00:00+00:00",
                    "source": "Example News",
                }
            ]
        )
        self.assertIn("Published: 2026-06-12", output)
        self.assertIn("Publisher: Example News", output)

    def test_weather_formatter_uses_structured_live_data(self):
        output = format_weather_result(
            {
                "location": "Delhi, India",
                "timezone": "Asia/Kolkata",
                "current": {
                    "time": "2026-06-12T14:30",
                    "condition": "Partly cloudy",
                    "temperature_2m": 35,
                    "apparent_temperature": 39,
                    "relative_humidity_2m": 45,
                    "wind_speed_10m": 12,
                    "precipitation": 0,
                    "units": {
                        "temperature_2m": "C",
                        "relative_humidity_2m": "%",
                        "wind_speed_10m": "km/h",
                        "precipitation": "mm",
                    },
                },
                "forecast": [],
            }
        )
        self.assertIn("Current weather in Delhi, India", output)
        self.assertIn("35 C", output)
        self.assertIn("Partly cloudy", output)

    def test_news_formatter_always_displays_publication_date(self):
        output = format_news_results(
            [
                {
                    "title": "Current story",
                    "url": "https://example.com/story",
                    "snippet": "A current report.",
                    "published_at": "2026-06-12T08:00:00+00:00",
                    "source": "Example News",
                }
            ],
            "2026-06-12",
        )
        self.assertIn("Retrieved on 2026-06-12", output)
        self.assertIn("Published:** 2026-06-12T08:00:00+00:00", output)
        self.assertIn("[Read source](https://example.com/story)", output)


if __name__ == "__main__":
    unittest.main()
