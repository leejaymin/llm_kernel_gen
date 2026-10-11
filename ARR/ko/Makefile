# ARR submission build. TEXMFHOME is isolated so the local ~/texmf l3kernel
# does not clash with the system pdflatex format. Path contains spaces: keep quotes.
export TEXMFHOME := $(CURDIR)/.empty_texmf
LATEX  := /usr/bin/pdflatex -interaction=nonstopmode   # system TeX; the conda pdflatex lacks article.cls
BIBTEX := /usr/bin/bibtex

.PHONY: all main_ko clean cleanpdf
all: main_ko

$(TEXMFHOME):
	@mkdir -p "$(TEXMFHOME)"

main_ko: $(TEXMFHOME)
	$(LATEX) main_ko.tex
	$(BIBTEX) main_ko
	$(LATEX) main_ko.tex
	$(LATEX) main_ko.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.fls *.fdb_latexmk

cleanpdf: clean
	rm -f main_ko.pdf
