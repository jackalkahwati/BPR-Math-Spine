# PNAS submission build
# Usage: make si    - compile SI Appendix PDF
#        make clean - remove generated PDFs

.PHONY: si clean

si: SI_Appendix.pdf

SI_Appendix.pdf: si_appendix_standalone.tex
	pdflatex -interaction=nonstopmode si_appendix_standalone.tex
	@echo ""
	@echo "Done: SI_Appendix.pdf ready for PNAS upload"

clean:
	rm -f SI_Appendix.pdf si_appendix_standalone.aux si_appendix_standalone.log si_appendix_standalone.out
