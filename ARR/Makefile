# ARR submission build. TEXMFHOME is isolated so the local ~/texmf l3kernel
# does not clash with the system pdflatex format. Path contains spaces: keep quotes.
export TEXMFHOME := $(CURDIR)/.empty_texmf
LATEX  := /usr/bin/pdflatex -interaction=nonstopmode   # system TeX; the conda pdflatex lacks article.cls
BIBTEX := /usr/bin/bibtex

.PHONY: all main clean cleanpdf
all: main

$(TEXMFHOME):
	@mkdir -p "$(TEXMFHOME)"

main: $(TEXMFHOME)
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.fls *.fdb_latexmk

cleanpdf: clean
	rm -f main.pdf
