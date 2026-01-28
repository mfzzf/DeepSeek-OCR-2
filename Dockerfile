FROM uhub.service.ucloud.cn/clientfzzf/vllm-openai:v0.8.5

# Configure pip mirror
RUN pip3 config set global.index-url https://pypi.internal-mirrors.ucloud.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_USE_V1=0

WORKDIR /app

# Install flash-attn from local wheel
COPY flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl /tmp/
RUN pip3 install /tmp/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && rm /tmp/*.whl

# Install additional dependencies
RUN pip3 install \
    PyMuPDF \
    img2pdf \
    einops \
    easydict \
    addict \
    matplotlib


COPY models /app/models

ENV MODEL_PATH=/app/models

COPY DeepSeek-OCR2-master /app/DeepSeek-OCR2-master
WORKDIR /app/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm
EXPOSE 8000
ENTRYPOINT []
CMD ["python3", "openai_api_server.py", "--host", "0.0.0.0", "--port", "8000"]
