import logging
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logging_config import configure_persistent_logging


class PersistentLoggingTests(unittest.TestCase):
    def test_debug_messages_are_written_to_rotating_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speaker.log"
            logger = logging.getLogger(f"speaker-log-test-{id(self)}")
            logger.propagate = False

            configured_path = configure_persistent_logging(
                logger,
                backup_count=2,
                level_name="DEBUG",
                max_bytes=4_096,
                path=str(path),
            )
            logger.debug("vad diagnostic")
            for handler in logger.handlers:
                handler.flush()

            self.assertEqual(configured_path, path.resolve())
            self.assertIn("DEBUG", path.read_text(encoding="utf-8"))
            self.assertIn("vad diagnostic", path.read_text(encoding="utf-8"))
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
