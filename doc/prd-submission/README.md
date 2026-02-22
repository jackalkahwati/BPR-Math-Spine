# Physical Review D Submission

Manuscript adapted from the PNAS submission for *Physical Review D*.

## Build PDF

### Option A: Overleaf (no LaTeX install)

1. Go to [overleaf.com](https://www.overleaf.com) and create a free account.
2. **New Project** → **Upload Project** → upload a zip of this folder (or create a blank project and upload `main.tex` and `cover_letter.tex`).
3. Set `main.tex` as the main document (Menu → Main document).
4. Click **Recompile**. Download the PDF from the preview pane.

### Option B: Local LaTeX

Requires a LaTeX distribution with `revtex4-2` (MacTeX, TeX Live, or MiKTeX).

```bash
cd doc/prd-submission
./build.sh
```

Or manually:
```bash
pdflatex main.tex
pdflatex main.tex   # second run for references
pdflatex cover_letter.tex
```

## Files

- `main.tex` — Main manuscript (PRD format, revtex4-2)
- `cover_letter.tex` — Cover letter for submission
- `README.md` — This file

## Differences from PNAS version

- Uses `revtex4-2` document class (PRD standard)
- Removed significance statement, author contributions (PRD does not use these)
- Added PACS numbers
- Added **fine structure constant derivation** (§III.D) — new result: α⁻¹ = [ln p]² + z/2 + γ − 1/(2π), 32 ppm deviation
- Condensed abstract for PRD length
- Bibliography in PRD citation style
- Methods integrated into main text

## Submission

1. Build `main.tex` to produce `main.pdf`
2. Build `cover_letter.tex` to produce `cover_letter.pdf`
3. Submit at https://authors.aps.org/
4. Upload main PDF; cover letter as supplementary file or in submission form

## Supplementary Material

The PNAS SI Appendix (derivations of κ, ξ, δ) can be submitted as supplementary material if the referee requests it. The main text is self-contained for initial review.
