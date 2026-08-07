from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


class ApplicationCompositionTests(unittest.TestCase):
    def test_all_public_routes_are_registered(self):
        routes = {
            (route.path, method)
            for route in app.routes
            for method in getattr(route, "methods", set())
        }
        expected = {
            ("/ollama/models", "GET"),
            ("/persona-preparations", "POST"),
            ("/persona-preparations/{preparation_id}", "GET"),
            ("/initiate-request", "POST"),
            ("/requests/{request_id}/text", "GET"),
            ("/requests/{request_id}/interrupt", "POST"),
            ("/requests/{request_id}/audio", "GET"),
            ("/pseudo-stream/initiate-request", "POST"),
            ("/pseudo-stream/requests/{request_id}/text", "GET"),
            ("/pseudo-stream/requests/{request_id}/interrupt", "POST"),
            ("/pseudo-stream/requests/{request_id}/audio", "GET"),
        }
        self.assertTrue(expected.issubset(routes))
        self.assertTrue(any(route.path == "/speaker/v1" for route in app.routes))


if __name__ == "__main__":
    unittest.main()
