all: main.tex
	pdflatex main.tex 1>/dev/null

intro-slides: intro-slides.tex
	pdflatex intro-slides.tex 1>/dev/null
	pdflatex intro-slides.tex 1>/dev/null
