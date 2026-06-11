import unittest

from app.config import Settings


class ConfigTests(unittest.TestCase):
    def test_release_environment_value_disables_debug(self):
        settings = Settings(debug="release")
        self.assertFalse(settings.debug)

    def test_development_environment_value_enables_debug(self):
        settings = Settings(debug="development")
        self.assertTrue(settings.debug)


if __name__ == "__main__":
    unittest.main()
