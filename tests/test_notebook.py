"""Fast repository checks; scientific execution tests live in the notebook."""
import ast
import csv
import json
import math
from pathlib import Path
import unittest

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class NotebookRepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / "OptiEnergy_AI.ipynb", as_version=4)
        cls.code = [c for c in cls.notebook.cells if c.cell_type == "code"]

    def test_notebook_schema(self):
        nbformat.validate(self.notebook)

    def test_every_python_cell_compiles(self):
        for i, cell in enumerate(self.code):
            with self.subTest(cell=i):
                ast.parse(cell.source)

    def test_no_missing_project_imports(self):
        for cell in self.code:
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.module == "config" or (node.module or "").startswith("src."))

    def test_no_opaque_project_or_data_blobs(self):
        for cell in self.code:
            self.assertNotIn("STREAMLIT_SOURCE_FILES", cell.source)
            self.assertNotIn("HOURLY_DATA_B64", cell.source)

    def test_saved_outputs_have_no_errors(self):
        errors = [o for cell in self.code for o in cell.get("outputs", []) if o.output_type == "error"]
        self.assertFalse(errors)

    def test_required_support_files_exist(self):
        for name in ["README.md", "requirements.txt", "LICENSE", ".gitignore", ".github/workflows/verify.yml", "tools/run_notebook.py"]:
            self.assertTrue((ROOT/name).is_file(), name)

    def test_committed_evidence_uses_portfolio_profile(self):
        path = ROOT / "results/temporal/protocol.json"
        if not path.exists():
            self.skipTest("Measured evidence has not been exported yet")
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["profile"], "portfolio")
        self.assertEqual(protocol["protocol"]["folds"], 3)
        self.assertEqual(protocol["protocol"]["seeds"], [7, 21, 42])

    def test_holdout_scores_recompute_from_prediction_table(self):
        folder = ROOT / "results/temporal"
        if not (folder / "holdout_predictions.csv").exists():
            self.skipTest("Measured predictions have not been exported yet")
        with (folder / "holdout_predictions.csv").open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        actual = [float(row["actual_kwh"]) for row in predictions]
        mean = sum(actual)/len(actual)
        total_squares = sum((value-mean)**2 for value in actual)
        with (folder / "holdout_runs.csv").open(encoding="utf-8", newline="") as handle:
            scores = list(csv.DictReader(handle))
        for row in scores:
            column = f"{row['family']} / seed {int(row['seed'])}"
            predicted = [float(record[column]) for record in predictions]
            squared = sum((a-p)**2 for a, p in zip(actual, predicted))
            rmse = math.sqrt(squared/len(actual))
            mae = sum(abs(a-p) for a, p in zip(actual, predicted))/len(actual)
            r2 = 1-squared/total_squares
            with self.subTest(family=row["family"], seed=row["seed"]):
                self.assertAlmostEqual(rmse, float(row["test_rmse"]), places=10)
                self.assertAlmostEqual(mae, float(row["test_mae"]), places=10)
                self.assertAlmostEqual(r2, float(row["test_r2"]), places=10)

    def test_leaderboard_agrees_with_frozen_selection(self):
        folder = ROOT / "results/temporal"
        if not (folder / "frozen_selection.json").exists():
            self.skipTest("Selection has not been exported yet")
        frozen = json.loads((folder / "frozen_selection.json").read_text(encoding="utf-8"))
        with (folder / "forecasting_leaderboard.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["family"], frozen["champion_family"])
        for row in rows:
            self.assertEqual(row["candidate_id"], frozen["candidate_ids"][row["family"]])
        self.assertEqual(frozen["deployment_seed"], 7)


if __name__ == "__main__":
    unittest.main()
