import argparse
import json
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


def convert_notebooks(notebooks, output_dir):
    generated_pdfs = []
    for notebook in notebooks:
        notebook = Path(notebook)
        pdf = output_dir / notebook.with_suffix(".pdf").name
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
                str(notebook),
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
    with tempfile.TemporaryDirectory(prefix="assignment2-demo-pdfs-") as tmp:
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
