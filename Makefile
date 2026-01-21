all: main.tex preamble.sty references.bib
	pdflatex -output-directory=latex_output main.tex 1>/dev/null
	cp ./latex_output/main.pdf main.pdf

bib: main.tex preamble.sty references.bib
	pdflatex -output-directory=latex_output main.tex 1>/dev/null
	biber -output-directory=latex_output main 1>/dev/null
	pdflatex -output-directory=latex_output main.tex 1>/dev/null
	cp ./latex_output/main.pdf main.pdf

intro-slides: intro-slides.tex slides-preamble.sty
	pdflatex -output-directory=latex_output intro-slides.tex 1>/dev/null
	pdflatex -output-directory=latex_output intro-slides.tex 1>/dev/null
	cp latex_output/intro-slides.pdf intro-slides.pdf

final-slides: final-slides.tex slides-preamble.sty
	pdflatex -output-directory=latex_output final-slides.tex 1>/dev/null
	pdflatex -output-directory=latex_output final-slides.tex 1>/dev/null
	cp latex_output/final-slides.pdf final-slides.pdf

report-slides: report-slides.tex slides-preamble.sty
	pdflatex -output-directory=latex_output report-slides.tex 1>/dev/null
	pdflatex -output-directory=latex_output report-slides.tex 1>/dev/null
	cp latex_output/report-slides.pdf report-slides.pdf