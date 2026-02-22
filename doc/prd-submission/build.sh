#!/bin/bash
# Build PRD manuscript PDF
# Requires: pdflatex (from MacTeX, TeX Live, or MiKTeX)

set -e
cd "$(dirname "$0")"

echo "Building main.pdf..."
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex  # second run for references

echo "Building cover_letter.pdf..."
pdflatex -interaction=nonstopmode cover_letter.tex

echo "Done. Output: main.pdf, cover_letter.pdf"
