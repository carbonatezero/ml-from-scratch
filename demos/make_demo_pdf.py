import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    from PyPDF2 import PdfMerger

    MERGE = True
except ImportError:
    print("Could not find PyPDF2. Leaving generated PDF files unmerged.")
    MERGE = False


def notebook_pdf_name(notebook):
    return Path(notebook).with_suffix(".pdf")


def convert_notebooks(notebooks, output_dir):
    generated_pdfs = []
    for notebook in notebooks:
        pdf = output_dir / notebook_pdf_name(notebook).name
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
                notebook,
            ],
            check=True,
        )
        generated_pdfs.append(pdf)
        print(f"Created PDF {pdf}.")
    return generated_pdfs


def merge_pdfs(pdfs, output_name):
    if not MERGE:
        return

    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(str(pdf))
    merger.write(output_name)
    merger.close()


def main(notebooks, pdf_name, keep_parts):
    with tempfile.TemporaryDirectory(prefix="demo-pdfs-") as tmp:
        output_dir = Path(tmp)
        pdfs = convert_notebooks(notebooks, output_dir)

        if keep_parts or not MERGE:
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
