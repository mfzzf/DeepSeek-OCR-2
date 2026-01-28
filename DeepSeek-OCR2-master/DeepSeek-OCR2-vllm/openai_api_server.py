# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import json
import os
import time
import uuid
import urllib.request
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageOps

from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.image_process import DeepseekOCR2Processor
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry


def _now_ts() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _safe_len(obj: Any) -> Optional[int]:
    if obj is None:
        return None
    try:
        return len(obj)  # type: ignore[arg-type]
    except TypeError:
        return None


def _prompt_tokens_from_mm_data(mm_data: Any) -> Optional[int]:
    # DeepseekOCR2Processor.tokenize_with_images returns:
    # [[input_ids, pixel_values, images_crop, images_seq_mask, images_spatial_crop, num_image_tokens, image_shapes]]
    try:
        input_ids = mm_data[0][0]
    except Exception:
        return None

    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        try:
            return int(shape[-1])
        except Exception:
            pass

    # Fallback for list-like tensors.
    try:
        return len(input_ids[0])
    except Exception:
        return None


def _is_data_url(url: str) -> bool:
    return url.startswith("data:")


def _load_image_from_data_url(data_url: str) -> Image.Image:
    # Format: data:<mime>;base64,<payload>
    try:
        header, b64_data = data_url.split(",", 1)
    except ValueError as e:
        raise ValueError("Invalid data URL format") from e

    if ";base64" not in header:
        raise ValueError("Only base64-encoded data URLs are supported")

    raw = base64.b64decode(b64_data, validate=True)
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _load_image_from_url(url: str, *, timeout_s: float = 10.0) -> Image.Image:
    if _is_data_url(url):
        return _load_image_from_data_url(url)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DeepSeek-OCR-2 OpenAI-Compatible Server",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _coerce_messages_to_prompt_and_images(messages: Any) -> tuple[str, list[Image.Image]]:
    if not isinstance(messages, list):
        raise ValueError("`messages` must be a list")

    system_parts: list[str] = []
    user_text_parts: list[str] = []
    images: list[Image.Image] = []

    last_user: Optional[dict[str, Any]] = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user = msg
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())

    if last_user is None:
        raise ValueError("No `role=user` message found")

    content = last_user.get("content")
    if isinstance(content, str):
        user_text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text")
                if isinstance(text, str):
                    user_text_parts.append(text)
            elif ptype == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url:
                        images.append(_load_image_from_url(url))
    else:
        raise ValueError("Unsupported `content` type in user message")

    system_text = "\n".join(system_parts).strip()
    user_text = "".join(user_text_parts).strip()

    text = user_text
    if system_text:
        text = f"{system_text}\n{text}".strip()

    if images:
        num_placeholders = text.count("<image>")
        if num_placeholders == 0:
            text = ("<image>\n" * len(images)) + text
        elif num_placeholders != len(images):
            raise ValueError(
                f"Prompt has {num_placeholders} '<image>' placeholders, "
                f"but received {len(images)} image(s)."
            )

    if not text:
        raise ValueError("Empty prompt")

    return text, images


def _get_int(d: dict[str, Any], key: str, default: int) -> int:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _get_float(d: dict[str, Any], key: str, default: float) -> float:
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get_bool(d: dict[str, Any], key: str, default: bool) -> bool:
    v = d.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def _get_stop(d: dict[str, Any]) -> Optional[list[str]]:
    stop = d.get("stop")
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        out: list[str] = []
        for s in stop:
            if isinstance(s, str):
                out.append(s)
        return out or None
    return None


def build_app(
    *,
    model_path: str,
    served_model_name: str,
    engine: AsyncLLMEngine,
    default_crop_mode: bool,
) -> FastAPI:
    app = FastAPI(title="DeepSeek-OCR-2 OpenAI-Compatible API")
    processor = DeepseekOCR2Processor()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        stream = _get_bool(body, "stream", False)
        stream_options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        include_usage = _get_bool(stream_options, "include_usage", False)
        n = _get_int(body, "n", 1)
        if n != 1:
            raise HTTPException(status_code=400, detail="Only `n=1` is supported")

        try:
            prompt, images = _coerce_messages_to_prompt_and_images(body.get("messages"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        max_tokens = _get_int(body, "max_tokens", 8192)
        temperature = _get_float(body, "temperature", 0.0)
        top_p = _get_float(body, "top_p", 1.0)
        stop = _get_stop(body)

        vllm_xargs = body.get("vllm_xargs") if isinstance(body.get("vllm_xargs"), dict) else {}
        ngram_size = _get_int(vllm_xargs, "ngram_size", 20)
        window_size = _get_int(vllm_xargs, "window_size", 90)
        whitelist_token_ids = vllm_xargs.get("whitelist_token_ids")
        if isinstance(whitelist_token_ids, list):
            whitelist = {int(x) for x in whitelist_token_ids if isinstance(x, (int, float, str))}
        else:
            whitelist = {128821, 128822}  # <td>, </td>

        logits_processors = [
            NoRepeatNGramLogitsProcessor(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids=whitelist,
            )
        ]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            logits_processors=logits_processors,
            skip_special_tokens=False,
        )

        req_id = _new_id("chatcmpl")
        created = _now_ts()

        prompt_tokens_fallback: Optional[int] = None
        if images:
            image_features = processor.tokenize_with_images(
                images=images,
                prompt=prompt,
                bos=True,
                eos=True,
                cropping=default_crop_mode,
            )
            prompt_tokens_fallback = _prompt_tokens_from_mm_data(image_features)
            model_input: dict[str, Any] = {
                "prompt": prompt,
                "multi_modal_data": {"image": image_features},
            }
        else:
            model_input = {"prompt": prompt}
            try:
                prompt_tokens_fallback = len(processor.encode(prompt, bos=True, eos=False))
            except Exception:
                prompt_tokens_fallback = None

        if not stream:
            final_text = ""
            finish_reason: Optional[str] = None
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            async for request_output in engine.generate(model_input, sampling_params, req_id):
                if prompt_tokens is None:
                    prompt_tokens = _safe_len(getattr(request_output, "prompt_token_ids", None))
                if request_output.outputs:
                    out0 = request_output.outputs[0]
                    final_text = getattr(out0, "text", final_text)
                    finish_reason = getattr(out0, "finish_reason", finish_reason)
                    completion_tokens = _safe_len(getattr(out0, "token_ids", None)) or completion_tokens

            if prompt_tokens is None:
                prompt_tokens = prompt_tokens_fallback or 0
            if completion_tokens is None:
                try:
                    completion_tokens = len(processor.tokenizer.encode(final_text, add_special_tokens=False))
                except Exception:
                    completion_tokens = 0

            return JSONResponse(
                {
                    "id": req_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": served_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": final_text},
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )

        async def event_stream():
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": served_model_name,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            printed_len = 0
            finish_reason: Optional[str] = None
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            final_text = ""

            async for request_output in engine.generate(model_input, sampling_params, req_id):
                if prompt_tokens is None:
                    prompt_tokens = _safe_len(getattr(request_output, "prompt_token_ids", None))
                if not request_output.outputs:
                    continue
                out0 = request_output.outputs[0]
                text = getattr(out0, "text", "")
                final_text = text
                finish_reason = getattr(out0, "finish_reason", finish_reason)
                completion_tokens = _safe_len(getattr(out0, "token_ids", None)) or completion_tokens
                if len(text) <= printed_len:
                    continue
                delta = text[printed_len:]
                printed_len = len(text)

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": served_model_name,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

            if prompt_tokens is None:
                prompt_tokens = prompt_tokens_fallback or 0
            if completion_tokens is None:
                try:
                    completion_tokens = len(processor.tokenizer.encode(final_text, add_special_tokens=False))
                except Exception:
                    completion_tokens = 0

            final_payload: dict[str, Any] = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": served_model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason or "stop"}],
            }
            if include_usage:
                final_payload["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            yield (
                "data: "
                + json.dumps(
                    final_payload,
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 OpenAI-Compatible API Server (vLLM)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", dest="model_path", default=os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-OCR-2"))
    parser.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--crop-mode", action="store_true", default=True)
    parser.add_argument("--no-crop-mode", action="store_false", dest="crop_mode")
    args = parser.parse_args()

    served_model_name = args.served_model_name or args.model_path

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app = build_app(
        model_path=args.model_path,
        served_model_name=served_model_name,
        engine=engine,
        default_crop_mode=args.crop_mode,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
