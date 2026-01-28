SHELL := bash

PYTHON ?= python3
VENV ?= venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV_PY) -m pip

# Pip / wheel indexes (override if you need a local mirror)
PIP_INDEX ?= https://pypi.org/simple
# Default to CUDA 12.4 wheels; override with a CPU index if needed.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu124

# Local model directory
MODEL_PATH ?= $(abspath models)

# Common runtime knobs
INPUT_PATH ?=
OUTPUT_PATH ?=
PROMPT ?=

# PyTorch versions
TORCH_VER ?= 2.6.0
TORCHVISION_VER ?= 0.21.0
TORCHAUDIO_VER ?= 2.6.0
VLLM_VER ?= 0.8.5

.PHONY: help venv install-base install-vllm install-hf run-image run-pdf run-eval api clean-venv

help:
	@echo "Targets:"
	@echo "  make venv                 Create $(VENV)"
	@echo "  make install-vllm         Install deps for vLLM scripts"
	@echo "  make run-image            Run vLLM image demo (set INPUT_PATH/OUTPUT_PATH/PROMPT)"
	@echo "  make run-pdf              Run vLLM pdf demo (set INPUT_PATH/OUTPUT_PATH/PROMPT)"
	@echo "  make run-eval             Run vLLM batch eval (set INPUT_PATH/OUTPUT_PATH/PROMPT)"
	@echo "  make api                  Start OpenAI-compatible API server (HOST/PORT optional)"
	@echo "  make install-hf           Install deps for Transformers HF script"
	@echo ""
	@echo "Examples:"
	@echo "  make install-vllm"
	@echo "  make run-image INPUT_PATH=/path/to/images OUTPUT_PATH=/tmp/ocr_out"
	@echo "  make api HOST=0.0.0.0 PORT=8000"

venv:
	@set -euo pipefail; \
	if ! test -x "$(VENV_PY)"; then \
		$(PYTHON) -m venv "$(VENV)"; \
	fi; \
	# Some minimal venvs may be created without pip; bootstrap it so other targets work.
	if ! "$(VENV_PY)" -c "import pip" >/dev/null 2>&1; then \
		"$(VENV_PY)" -m ensurepip --upgrade; \
	fi; \
	# Best-effort upgrades (offline environments should still work with ensurepip's wheels).
	"$(VENV_PY)" -m pip install --upgrade pip setuptools wheel || \
		echo "WARNING: pip upgrade failed (offline?). Continuing with bundled pip."

install-base: venv
	$(VENV_PIP) config set global.index-url $(PIP_INDEX)
	$(VENV_PIP) install torch==$(TORCH_VER) torchvision==$(TORCHVISION_VER) torchaudio==$(TORCHAUDIO_VER) \
		--index-url $(TORCH_INDEX)

install-vllm: install-base
	$(VENV_PIP) install vllm==$(VLLM_VER)
	$(VENV_PIP) install flash-attn --no-build-isolation
	$(VENV_PIP) install fastapi "uvicorn[standard]" PyMuPDF img2pdf einops easydict addict Pillow numpy

run-image: install-vllm
	@test -n "$(INPUT_PATH)" || (echo "Set INPUT_PATH=/path/to/image_or_dir"; exit 1)
	@test -n "$(OUTPUT_PATH)" || (echo "Set OUTPUT_PATH=/path/to/output_dir"; exit 1)
	cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm && \
		MODEL_PATH="$(MODEL_PATH)" INPUT_PATH="$(INPUT_PATH)" OUTPUT_PATH="$(OUTPUT_PATH)" PROMPT="$(PROMPT)" \
		"$(abspath $(VENV_PY))" run_dpsk_ocr2_image.py

run-pdf: install-vllm
	@test -n "$(INPUT_PATH)" || (echo "Set INPUT_PATH=/path/to/pdf_or_dir"; exit 1)
	@test -n "$(OUTPUT_PATH)" || (echo "Set OUTPUT_PATH=/path/to/output_dir"; exit 1)
	cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm && \
		MODEL_PATH="$(MODEL_PATH)" INPUT_PATH="$(INPUT_PATH)" OUTPUT_PATH="$(OUTPUT_PATH)" PROMPT="$(PROMPT)" \
		"$(abspath $(VENV_PY))" run_dpsk_ocr2_pdf.py

run-eval: install-vllm
	@test -n "$(INPUT_PATH)" || (echo "Set INPUT_PATH=/path/to/images_dir"; exit 1)
	@test -n "$(OUTPUT_PATH)" || (echo "Set OUTPUT_PATH=/path/to/output_dir"; exit 1)
	cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm && \
		MODEL_PATH="$(MODEL_PATH)" INPUT_PATH="$(INPUT_PATH)" OUTPUT_PATH="$(OUTPUT_PATH)" PROMPT="$(PROMPT)" \
		"$(abspath $(VENV_PY))" run_dpsk_ocr2_eval_batch.py

HOST ?= 127.0.0.1
PORT ?= 8000
api: install-vllm
	cd DeepSeek-OCR2-master/DeepSeek-OCR2-vllm && \
		MODEL_PATH="$(MODEL_PATH)" "$(abspath $(VENV_PY))" openai_api_server.py --host "$(HOST)" --port "$(PORT)" --model "$(MODEL_PATH)"

install-hf: install-base
	$(VENV_PIP) install -r requirements.txt
	@echo "HF script installed. Run:"
	@echo "  cd DeepSeek-OCR2-master/DeepSeek-OCR2-hf && \\"
	@echo "    MODEL_PATH=\"$(MODEL_PATH)\" IMAGE_FILE=/path/to/image OUTPUT_PATH=/tmp/ocr_out \\"
	@echo "    $(VENV_PY) run_dpsk_ocr2.py"

clean-venv:
	rm -rf "$(VENV)"
