# One-command reproduction:
#   make setup   - create .venv and install pinned deps (+ NeMo)
#   make data    - download the fixed public eval subsets + build the synthetic sets
#   make eval    - Nemotron 3.5 ASR (320 ms + 1.12 s) and Whisper over every set x condition, + NeMo cross-check
#   make report  - score everything -> results/RESULTS.md
#   make test    - unit tests (codec, normalization, metrics, fine-tune aug, Pipecat)
#   make all     - data + eval + report
PY := .venv/bin/python
SETS ?= fleurs_es419 ciempiess_mx openslr_ar openslr_cl openslr_co openslr_pe openslr_pr openslr_ve spanish_synth_control codeswitch_synth
LIMIT ?=
LIMIT_ARG := $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: setup data eval eval-nemotron eval-nemotron-1120 eval-whisper eval-extra eval-nemo-crosscheck eval-deepgram report test test-fast all clean-results

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -r requirements-nemo.txt
	$(PY) -m pip install -e .

data:
	$(PY) -m esasr.datasets --sets $(SETS)

REAL_SETS := fleurs_es419 ciempiess_mx openslr_ar openslr_cl openslr_co openslr_pe openslr_pr openslr_ve

eval: eval-nemotron eval-nemotron-1120 eval-whisper eval-extra eval-nemo-crosscheck

eval-nemotron:
	$(PY) -m esasr.run_eval --system nemotron --sets $(SETS) $(LIMIT_ARG)

# Nemotron at 1.12 s chunks (most accurate streaming setting) on the real-speech sets
eval-nemotron-1120:
	$(PY) -m esasr.run_eval --system nemotron:es-US:13 --sets $(REAL_SETS) $(LIMIT_ARG)

eval-whisper:
	$(PY) -m esasr.run_eval --system whisper --sets $(SETS) $(LIMIT_ARG)

# Nemotron with automatic language ID on the synthetic sets (code-switching question)
eval-extra:
	$(PY) -m esasr.run_eval --system nemotron:auto --sets spanish_synth_control codeswitch_synth $(LIMIT_ARG)

# NeMo reference runtime on CPU, one set, to cross-check the transformers port
eval-nemo-crosscheck:
	$(PY) -m esasr.run_eval --system nemotron-nemo --sets fleurs_es419 --conditions clean g711 $(LIMIT_ARG)

# Paid API: refuses to run without DEEPGRAM_API_KEY. Not part of `make eval`.
eval-deepgram:
	@test -n "$$DEEPGRAM_API_KEY" || (echo "DEEPGRAM_API_KEY not set - skipping (paid API)"; exit 1)
	$(PY) -m esasr.run_eval --system deepgram:es --sets $(SETS) $(LIMIT_ARG)

report:
	$(PY) -m esasr.report

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow"

all: data eval report

clean-results:
	rm -rf results/hyps results/*.csv results/RESULTS.md
