SHELL := bash

PYTHON ?= python3
VENV ?= venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Pip mirrors
PIP_INDEX ?= https://pypi.internal-mirrors.ucloud.cn/simple
TORCH_INDEX ?= https://mirrors.nju.edu.cn/pytorch/whl/cu124

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
	@if test -x "$(VENV_PY)"; then \
		echo "$(VENV) already exists"; \
	else \
		$(PYTHON) -m venv "$(VENV)"; \
	fi

install-base: venv
	$(VENV_PIP) config set global.index-url $(PIP_INDEX)
	$(VENV_PIP) install --upgrade pip setuptools
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
