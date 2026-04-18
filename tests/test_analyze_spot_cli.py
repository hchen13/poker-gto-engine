import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class AnalyzeSpotCLITest(unittest.TestCase):
    def test_generic_cli_accepts_json_input_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "kuhn_root.json"
            input_path.write_text(
                json.dumps(
                    {
                        "game": "kuhn",
                        "hero_card": "Q",
                        "history": "",
                        "iterations": 20000,
                    }
                ),
                encoding="utf-8",
            )
            output = subprocess.check_output(
                ["python3", "-m", "python.analyze_spot", "--input-file", str(input_path), "--format", "json"],
                text=True,
            )

        result = json.loads(output)
        self.assertEqual(result["game"], "kuhn")
        self.assertEqual(result["hero_card"], "Q")
        self.assertEqual(result["recommended_action"], "check")
        self.assertLess(result["strategy"]["bet"], 0.1)


if __name__ == "__main__":
    unittest.main()
