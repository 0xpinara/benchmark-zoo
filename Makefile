# One command per stage.  `make all` reproduces every table and figure in the
# paper from the raw data.  Each stage writes into data/ and paper/, and every
# later stage reads only what an earlier one wrote, so stages can be re-run
# individually without re-running the whole pipeline.

PY := ./.venv/bin/python
DROPBOX_HTAP := "https://www.dropbox.com/scl/fo/qd3aw9x8eje18tiy7gqm2/h?rlkey=6h07i9npnf852x92uqe6nebgn&dl=1"

.PHONY: all env fetch data sanity calibrate validate reevaluate sweep sweep-arxiv \
        mlanalysis observen robust tables figures paper overleaf test clean help

help:
	@echo "make env         create the virtual environment and install the package"
	@echo "make fetch       download the 1.8 GB mined strategy archive (once)"
	@echo "make data        convert every source to the parquet cache"
	@echo "make sanity      partition the mined populations, run the three sanity checks"
	@echo "make calibrate   estimate the empirical null on the known-null population"
	@echo "make validate    test the known-null assumption"
	@echo "make reevaluate  thresholds, and the 212 published predictors"
	@echo "make sweep       machine learning sweep on the three small graphs"
	@echo "make sweep-arxiv machine learning sweep on ogbn-arxiv (slow)"
	@echo "make mlanalysis  sigma_Delta, deflation, leaderboard, saturation"
	@echo "make robust      the robustness suite"
	@echo "make tables      emit every .tex table and macro"
	@echo "make figures     emit every figure"
	@echo "make observen    observable trial counts from leaderboard and citations"
	@echo "make overleaf    self-contained, validated folder and zip for Overleaf"
	@echo "make all         everything above, in order"

env:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,ml]"

fetch:
	mkdir -p data/raw/htap
	test -f data/raw/htap/TickerSignalsLongShort.csv.gzip || ( \
	  curl -sSL --retry 3 -o data/raw/htap/htap.zip $(DROPBOX_HTAP) && \
	  unzip -o -j data/raw/htap/htap.zip -d data/raw/htap && \
	  rm data/raw/htap/htap.zip )

data: fetch
	$(PY) scripts/01_download_data.py

sanity: data
	$(PY) scripts/02_partition_and_sanity.py

calibrate: sanity
	$(PY) scripts/03_empirical_null.py

validate: calibrate
	$(PY) scripts/04_known_null_validation.py

reevaluate: validate
	$(PY) scripts/05_reevaluate_published.py

sweep:
	$(PY) scripts/06_ml_sweep.py --datasets cora citeseer pubmed --workers 4 --threads 2

sweep-arxiv:
	$(PY) scripts/06_ml_sweep.py --datasets ogbn-arxiv --workers 4 --threads 2

mlanalysis: sweep
	$(PY) scripts/07_ml_analysis.py

observen:
	$(PY) scripts/10_observable_n.py

robust: mlanalysis reevaluate
	$(PY) scripts/08_robustness.py

# The alpha mechanism: the dose-response across factor models, the exposure
# deciles, the placebo and the second no-content population.  Depends only on
# the finance data, so it does not wait on the ML sweep.
mechanism: validate
	$(PY) scripts/12_alpha_mechanism.py

tables figures: robust observen mechanism
	$(PY) scripts/09_tables_and_figures.py
	$(PY) scripts/13_alpha_tables_and_figures.py

paper: tables
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
	  bibtex main && pdflatex -interaction=nonstopmode main.tex && \
	  pdflatex -interaction=nonstopmode main.tex

# Self-contained folder plus zip for Overleaf, statically validated.  Use this
# when there is no local LaTeX installation.
overleaf: tables
	$(PY) scripts/11_package_paper.py

all: reevaluate mlanalysis observen robust mechanism tables overleaf

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/cache data/interim data/results paper/tables/*.tex \
	  paper/figures/* overleaf benchmark-zoo-paper.zip
