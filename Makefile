all: main.tex preamble.sty references.bib
	pdflatex main.tex 1>/dev/null

bib: main.tex preamble.sty references.bib
	pdflatex main.tex 1>/dev/null
	biber main 1>/dev/null
	pdflatex main.tex 1>/dev/null

intro-slides: intro-slides.tex slides-preamble.sty
	pdflatex intro-slides.tex 1>/dev/null
	pdflatex intro-slides.tex 1>/dev/null
