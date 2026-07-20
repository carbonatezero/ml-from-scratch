import ast
import json
from pathlib import Path


def test_demo_notebooks_are_parseable_json():
    for notebook in Path("demos").glob("*_repo.ipynb"):
        data = json.loads(notebook.read_text())
        assert data.get("nbformat") == 4
        assert data.get("cells")


def test_demo_notebook_code_cells_are_parseable_python_when_plain_code():
    for notebook in Path("demos").glob("*_repo.ipynb"):
        data = json.loads(notebook.read_text())
        for index, cell in enumerate(data.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            stripped = source.lstrip()
            if stripped.startswith("%%") or stripped.startswith("!") or "%" in source:
                continue
            ast.parse(source, filename=f"{notebook}:cell-{index}")
