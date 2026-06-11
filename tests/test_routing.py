import unittest

from app.agents.routing import choose_route, detect_priority_route


class RoutingTests(unittest.TestCase):
    def test_current_news_routes_to_web(self):
        self.assertEqual(
            detect_priority_route("What is the latest AI news today?"),
            "web",
        )

    def test_research_news_does_not_route_to_rag(self):
        self.assertEqual(
            detect_priority_route("Latest news about AI research models"),
            "web",
        )

    def test_weather_routes_to_web(self):
        self.assertEqual(
            detect_priority_route("Will it rain in Mumbai tomorrow?"),
            "web",
        )

    def test_stock_quote_routes_to_finance(self):
        self.assertEqual(
            detect_priority_route("What is the stock price of TSLA?"),
            "finance",
        )

    def test_live_cricket_routes_to_finance(self):
        self.assertEqual(
            detect_priority_route("Live cricket score today"),
            "finance",
        )

    def test_other_live_sports_routes_to_web(self):
        self.assertEqual(
            detect_priority_route("What is the live NBA score?"),
            "web",
        )

    def test_implicit_current_office_holder_routes_to_web(self):
        self.assertEqual(
            detect_priority_route("Who is the president of the United States?"),
            "web",
        )

    def test_exchange_rate_routes_to_web(self):
        self.assertEqual(
            detect_priority_route("What is the USD to INR exchange rate?"),
            "web",
        )

    def test_uploaded_file_does_not_hijack_weather(self):
        self.assertEqual(
            detect_priority_route(
                "Weather in Delhi",
                has_files=True,
            ),
            "web",
        )

    def test_explicit_uploaded_document_routes_to_rag(self):
        self.assertEqual(
            detect_priority_route(
                "Summarize this uploaded PDF",
                has_files=True,
            ),
            "rag",
        )

    def test_general_question_stays_for_llm_classification(self):
        self.assertIsNone(
            detect_priority_route(
                "Explain recursion with a simple example",
                has_files=True,
            )
        )

    def test_recipe_question_goes_directly_to_general(self):
        self.assertEqual(
            choose_route("How to make chole bhature?"),
            "general",
        )

    def test_weather_followup_keeps_web_route(self):
        self.assertEqual(
            detect_priority_route(
                "What about Mumbai?",
                previous_query="What is the weather in Delhi?",
            ),
            "web",
        )

    def test_news_followup_keeps_web_route(self):
        self.assertEqual(
            detect_priority_route(
                "What about technology?",
                previous_query="Show me today's top news",
            ),
            "web",
        )


if __name__ == "__main__":
    unittest.main()
