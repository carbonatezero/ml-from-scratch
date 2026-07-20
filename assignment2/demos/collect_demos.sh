#!/usr/bin/env bash
# Collect the repo demo notebooks and source code into a zip and combined PDF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NOTEBOOKS=(
  "01_batch_normalization_repo.ipynb"
  "02_dropout_repo.ipynb"
  "03_convolutional_neural_networks_repo.ipynb"
  "04_pytorch_on_cifar10_repo.ipynb"
  "05_image_captioning_vanilla_rnns_repo.ipynb"
)

ZIP_FILENAME="demos_code_bundle.zip"
PDF_FILENAME="demos_combined.pdf"
: "${PYTHON:=python}"

C_R="\033[31m"
C_G="\033[32m"
C_E="\033[0m"

cd "${SCRIPT_DIR}"

for notebook in "${NOTEBOOKS[@]}"; do
  if [ ! -f "${notebook}" ]; then
    echo -e "${C_R}Required notebook ${notebook} not found. Exiting.${C_E}"
    exit 1
  fi
done

if [ ! -d "${REPO_ROOT}/src" ]; then
  echo -e "${C_R}Required source directory ${REPO_ROOT}/src not found. Exiting.${C_E}"
  exit 1
fi

echo "### Zipping demos and source ###"
rm -f "${ZIP_FILENAME}"
(
  cd "${REPO_ROOT}"
  zip -q "demos/${ZIP_FILENAME}" \
    -r src demos/README.md \
    $(printf "demos/%s " "${NOTEBOOKS[@]}") \
    -x "*/__pycache__/*" "*.pyc" "*/.DS_Store" "demos/.ipynb_checkpoints/*"
)

echo "### Creating combined PDF ###"
"${PYTHON}" make_demo_pdf.py --notebooks "${NOTEBOOKS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "${C_G}### Done. Created ${ZIP_FILENAME} and ${PDF_FILENAME}. ###${C_E}"
