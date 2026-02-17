# PNAS Submission Fixes (Tracking #2026-04895)

This folder contains the fixes for the two PNAS editorial issues.

---

## Issue 1: Supporting Information Missing

**Fix:** Compile the SI Appendix to PDF and upload it to PNAS Central.

### Option A: Local compilation (if you have LaTeX/TeX Live)

```bash
cd doc/pnas-submission
make si
```

This produces `SI_Appendix.pdf`. Upload it as the "Supporting Information" / "SI Appendix" file in PNAS Central.

### Option B: Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload `si_appendix_standalone.tex` (uses standard `article` class, no PNAS template needed)
3. Click "Recompile" to generate the PDF
4. Download the PDF and upload to PNAS Central

### Option C: Online LaTeX compiler

Use [LaTeX.Online](https://latexonline.cc) or similar: paste the contents of `si_appendix_standalone.tex` and compile.

---

## Issue 2: Abstract Mismatch

**Fix:** Ensure the abstract in the PNAS online form matches the manuscript exactly.

1. Log in to PNAS Central: https://www.pnascentral.org/cgi-bin/main.plex?el=A6B6EyqL7A5OSWX2F4A9ftdpAfXwKDPmp4L8iH9g2lowZ
2. Open your manuscript and go to the abstract field
3. Copy the exact text from `abstract_for_online_form.txt` (between the === markers)
4. Paste into the online form, replacing any existing text
5. Save

The manuscript abstract (in `main.tex` lines 25–26) is the authoritative version. The online form must match it verbatim.

---

## Checklist Before Resubmitting

- [ ] SI Appendix PDF uploaded to PNAS Central
- [ ] Abstract in online form matches manuscript exactly
- [ ] Click "Submit Manuscript" (not "Approve Manuscript" until you've verified all changes)
