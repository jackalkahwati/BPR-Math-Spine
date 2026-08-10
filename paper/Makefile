# Build the BPR papers and textbook from source.
#   make            -> all three papers + textbook
#   make paper      -> bpr_paper.pdf        (arXiv/website "Paper")
#   make textbook   -> textbook/main.pdf    (website "Textbook")
#   make deploy-pdfs-> copy built PDFs to the website staging dir if present
#
# Requires pdflatex (TeX Live / MacTeX). Missing packages install with:
#   tlmgr --usermode install <pkg>

PDFLATEX = pdflatex -interaction=nonstopmode -halt-on-error

all: paper expanded mega textbook

paper: bpr_paper.pdf
expanded: bpr_paper_expanded.pdf
mega: bpr_mega_paper.pdf

%.pdf: %.tex
	$(PDFLATEX) $< >/dev/null
	$(PDFLATEX) $< >/dev/null

textbook: textbook/main.pdf

textbook/main.pdf: textbook/main.tex
	cd textbook && $(PDFLATEX) main.tex >/dev/null && $(PDFLATEX) main.tex >/dev/null

WEB_STAGE ?= /tmp/bpr-site
deploy-pdfs: paper textbook
	@if [ -d "$(WEB_STAGE)" ]; then \
	  cp bpr_paper.pdf "$(WEB_STAGE)/bpr-paper.pdf"; \
	  cp textbook/main.pdf "$(WEB_STAGE)/bpr-textbook.pdf"; \
	  echo "PDFs staged to $(WEB_STAGE) — deploy with: cd $(WEB_STAGE) && vercel deploy --prod"; \
	else echo "$(WEB_STAGE) not found; set WEB_STAGE=<site dir>"; fi

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *Notes.bib
	rm -f textbook/*.aux textbook/*.log textbook/*.out textbook/*.toc

.PHONY: all paper expanded mega textbook deploy-pdfs clean
