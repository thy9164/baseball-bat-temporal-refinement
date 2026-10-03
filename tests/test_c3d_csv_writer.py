"""CSV-writer regression checks without optional C3D or preview dependencies."""

import ast
import csv
import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


GENERATOR = (Path(__file__).resolve().parents[1] / "refinement"
             / "synthetic_data_by_c3d" / "generate_bat_2d_coordinates.py")


class C3DCSVWriterTests(unittest.TestCase):
    def setUp(self):
        # Execute the actual writer definitions, isolating only process_file:
        # this test exercises serialization, not C3D projection or video rendering.
        tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
        names = {"safe_name", "write_single_csv", "write_split_by_folder"}
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({node.name for node in functions}, names)
        self.process = Mock(side_effect=[
            [{"sequence_id": "a", "frame_idx": 0, "head_u": 12.5}],
            [{"sequence_id": "b", "frame_idx": 1, "head_u": 25.0}],
        ])
        self.namespace = {"csv": csv, "Path": Path, "re": re,
                          "process_file": self.process}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(GENERATOR), "exec"),
             self.namespace)

    def test_default_single_csv_writes_all_rows_to_out(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "all.csv"
            args = SimpleNamespace(split_by_folder=False, out=str(output), append=False)
            files = [Path("first/a.c3d"), Path("second/b.c3d")]
            rng = object()
            with redirect_stdout(io.StringIO()):
                self.namespace["write_single_csv"](files, args, rng)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [
                {"sequence_id": "a", "frame_idx": "0", "head_u": "12.5"},
                {"sequence_id": "b", "frame_idx": "1", "head_u": "25.0"},
            ])
            self.assertEqual(self.process.call_args_list,
                             [unittest.mock.call(path, args, rng) for path in files])

    def test_split_by_folder_keeps_separate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(split_by_folder=True, out_dir=directory, append=False)
            with redirect_stdout(io.StringIO()):
                self.namespace["write_split_by_folder"](
                    [Path("first/a.c3d"), Path("second/b.c3d")], args, object())
            for folder, sequence in [("first", "a"), ("second", "b")]:
                with (Path(directory) / f"{folder}.csv").open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["sequence_id"], sequence)


if __name__ == "__main__":
    unittest.main()
