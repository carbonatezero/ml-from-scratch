import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from PyPDF2 import PdfMerger
except ImportError:
    try:
        from pypdf import PdfWriter as PdfMerger
    except ImportError:
        PdfMerger = None


def notebook_pdf_name(notebook):
    return Path(notebook).with_suffix(".pdf")


def renumber_markdown_headings(source, notebook_number, section_number):
    lines = source.splitlines(keepends=True)
    renumbered = []

    for line in lines:
        newline = "\n" if line.endswith("\n") else ""
        text = line[:-1] if newline else line

        h1_match = re.match(r"^(#)\s+(?!\d+\.\s)(.+)$", text)
        if h1_match:
            renumbered.append(f"# {notebook_number}. {h1_match.group(2)}{newline}")
            continue

        h2_match = re.match(r"^(##)\s+(?:\d+\.\s+)?(.+)$", text)
        if h2_match:
            section_number += 1
            title = h2_match.group(2)
            renumbered.append(f"## {notebook_number}.{section_number} {title}{newline}")
            continue

        renumbered.append(line)

    return renumbered, section_number


def prepare_notebook_for_pdf(notebook, output_dir, notebook_number):
    notebook = Path(notebook)
    with notebook.open() as f:
        data = json.load(f)

    section_number = 0
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        cell["source"], section_number = renumber_markdown_headings(
            source, notebook_number, section_number
        )

    data.setdefault("cells", []).insert(
        0,
        {
            "cell_type": "raw",
            "metadata": {},
            "source": ["\\setcounter{secnumdepth}{0}\n"],
        },
    )

    output_path = output_dir / notebook.name
    with output_path.open("w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    return output_path


def convert_notebooks(notebooks, output_dir):
    generated_pdfs = []
    for number, notebook in enumerate(notebooks, start=1):
        prepared_notebook = prepare_notebook_for_pdf(notebook, output_dir, number)
        pdf = output_dir / notebook_pdf_name(prepared_notebook).name
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--log-level",
                "CRITICAL",
                "--to",
                "pdf",
                "--output-dir",
                str(output_dir),
                str(prepared_notebook),
            ],
            check=True,
        )
        generated_pdfs.append(pdf)
        print(f"Created PDF {pdf}.")
    return generated_pdfs


def merge_pdfs(pdfs, output_name):
    if PdfMerger is not None:
        merger = PdfMerger()
        for pdf in pdfs:
            merger.append(str(pdf))
        merger.write(output_name)
        merger.close()
        return

    if shutil.which("pdfunite"):
        subprocess.run(["pdfunite", *map(str, pdfs), str(output_name)], check=True)
        return

    if shutil.which("qpdf"):
        subprocess.run(
            ["qpdf", "--empty", "--pages", *map(str, pdfs), "--", str(output_name)],
            check=True,
        )
        return

    print("Could not find PyPDF2, pypdf, pdfunite, or qpdf. Leaving PDFs unmerged.")


def main(notebooks, pdf_name, keep_parts):
    with tempfile.TemporaryDirectory(prefix="demo-pdfs-") as tmp:
        output_dir = Path(tmp)
        pdfs = convert_notebooks(notebooks, output_dir)

        if keep_parts:
            for pdf in pdfs:
                shutil.copy2(pdf, Path(pdf.name))

        merge_pdfs(pdfs, pdf_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    parser.add_argument("--pdf_filename", type=str, required=True)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    main(args.notebooks, args.pdf_filename, args.keep_parts)
