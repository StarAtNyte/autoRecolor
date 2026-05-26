from __future__ import annotations
import json
import uuid
import io
import base64
import copy
from pathlib import Path
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
import numpy as np

from autoRecolor.analyze import analyze_image
from autoRecolor.palette import Palette
from autoRecolor.recolor import recolor_image
from autoRecolor.utils import hex_to_rgb
from autoRecolor.agent import TOOLS, MODEL, OLLAMA_BASE, SYSTEM_PROMPT

HERE = Path(__file__).parent
STATIC = HERE / "static"
TEMP = HERE / ".uploads"
TEMP.mkdir(exist_ok=True)

app = FastAPI(title="autoRecolor")

sessions: dict[str, dict] = {}


def _make_preview(img: Image.Image, original: Palette, modified: Palette, label_map: np.ndarray) -> str:
    buf = io.BytesIO()
    recolor_image(img, original, modified, label_map).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _make_palette_strip(palette: Palette, width: int = 600) -> str:
    h = 40
    canvas = Image.new("RGB", (width, h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    n = len(palette.palette)
    if n == 0:
        return ""
    seg = max(1, width // n)
    for i, c in enumerate(palette.palette):
        draw.rectangle([i * seg, 0, (i + 1) * seg, h], fill=tuple(hex_to_rgb(c.hex)))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    sid = uuid.uuid4().hex
    ext = Path(file.filename).suffix or ".png"
    path = TEMP / f"{sid}{ext}"
    with open(path, "wb") as f:
        f.write(await file.read())

    palette, img, label_map, _ = analyze_image(str(path), n_colors=6)
    original = copy.deepcopy(palette)

    preview_url = _make_preview(img, original, palette, label_map)
    palette_url = _make_palette_strip(palette, img.size[0])

    sessions[sid] = {
        "path": path,
        "img": img,
        "original_palette": original,
        "palette": palette,
        "label_map": label_map,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(palette_json=palette.to_json())},
        ],
    }

    return {
        "session_id": sid,
        "palette": palette.to_dict(),
        "preview_url": preview_url,
        "palette_strip_url": palette_url,
    }


@app.get("/stream/{sid}")
async def stream_chat(sid: str, prompt: str):
    session = sessions.get(sid)
    if not session:
        return StreamingResponse(
            _sse_error("Session not found"),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _agent_loop(session, prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx/proxy buffering
        },
    )


async def _agent_loop(session: dict, prompt: str) -> AsyncGenerator[str]:
    palette: Palette = session["palette"]
    messages: list[dict] = session["messages"]
    original = session["original_palette"]
    img = session["img"]
    label_map = session["label_map"]

    messages.append({"role": "user", "content": prompt})

    max_turns = 15
    for turn in range(max_turns):
        content = ""
        tool_calls = None

        # Stream thinking tokens live; collect content + tool_calls for processing
        async for event, *payload in _call_llm(messages):
            if event == "thinking":
                yield f"event: thinking\ndata: {json.dumps({'chunk': payload[0]})}\n\n"
            elif event == "done":
                content, tool_calls = payload

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            yield f"event: message\ndata: {json.dumps({'text': content})}\n\n"
            break

        history_tool_calls = []
        for tc in tool_calls:
            raw_args = tc["function"]["arguments"]
            dict_args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
            history_tool_calls.append({
                "id": tc.get("id", ""),
                "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": dict_args},
            })

        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": history_tool_calls,
        })

        finalize_result = None
        for tc, htc in zip(tool_calls, history_tool_calls):
            func_name = htc["function"]["name"]
            args = htc["function"]["arguments"]
            result = _execute_tool(palette, func_name, args)

            yield f"event: tool_call\ndata: {json.dumps({'name': func_name, 'args': args})}\n\n"

            messages.append({"role": "tool", "content": json.dumps(result)})

            if func_name == "finalize":
                finalize_result = result.get("reasoning", "Done.")
                yield f"event: finalize\ndata: {json.dumps({'reasoning': finalize_result})}\n\n"

            if func_name in ("update_color", "adjust_hsl", "relabel_color"):
                _sync_prompt(messages, palette)
                preview = _make_preview(img, original, palette, label_map)
                palette_strip = _make_palette_strip(palette, img.size[0])
                yield f"event: preview\ndata: {json.dumps({'preview_url': preview, 'palette_strip_url': palette_strip, 'palette': palette.to_dict()})}\n\n"

        if finalize_result is not None:
            yield f"event: done\ndata: {json.dumps({'reasoning': finalize_result})}\n\n"
            return

        if turn == 0:
            messages.append({
                "role": "user",
                "content": "Good. If you're done with all changes, call finalize now.",
            })

    yield "event: done\ndata: {}\n\n"


async def _call_llm(messages: list[dict]):
    """
    Async generator.
    Yields: ("thinking", chunk_str)  — one per streamed thinking token
    Final:  ("done", content, tool_calls)
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "think": True,                    # enable Qwen3 reasoning tokens
        "options": {"temperature": 0.3, "num_ctx": 32768},
    }
    content = ""
    tool_calls_acc: dict[int, dict] = {}

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", f"{OLLAMA_BASE}/api/chat", json=payload) as resp:
            if not resp.is_success:
                body = await resp.aread()
                raise RuntimeError(f"Ollama {resp.status_code}: {body[:400]}")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})

                if "thinking" in msg and msg["thinking"]:
                    yield "thinking", msg["thinking"]

                if "content" in msg:
                    content += msg["content"]

                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        idx = tc.get("index", len(tool_calls_acc))
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        acc = tool_calls_acc[idx]
                        if "id" in tc:
                            acc["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if "name" in fn:
                            acc["function"]["name"] += fn["name"]
                        if "arguments" in fn:
                            raw = fn["arguments"]
                            if isinstance(raw, dict):
                                acc["function"]["arguments"] = json.dumps(raw)
                            else:
                                acc["function"]["arguments"] += raw

                if chunk.get("done"):
                    break

    tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None
    yield "done", content, tool_calls


def _execute_tool(palette: Palette, name: str, args: dict) -> dict:
    try:
        match name:
            case "get_palette":
                return palette.to_dict()
            case "update_color":
                palette.update_hex(args["color_id"], args["hex"])
                return {"success": True, "palette": palette.to_dict()}
            case "adjust_hsl":
                palette.adjust_hsl(args["color_id"], args["h_shift"], args["s_shift"], args["l_shift"])
                return {"success": True, "palette": palette.to_dict()}
            case "relabel_color":
                palette.relabel(args["color_id"], args["label"])
                return {"success": True, "palette": palette.to_dict()}
            case "finalize":
                return {"success": True, "reasoning": args.get("reasoning", "")}
            case _:
                return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}


def _sync_prompt(messages: list[dict], palette: Palette) -> None:
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = SYSTEM_PROMPT.format(palette_json=palette.to_json())


async def _sse_error(msg: str) -> AsyncGenerator[str]:
    yield f"event: error\ndata: {json.dumps({'error': msg})}\n\n"


@app.get("/model-status")
async def model_status():
    """Proxy Ollama /api/ps so the browser doesn't need direct Ollama access."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE}/api/ps")
            data = r.json()
            models = data.get("models", [])
            loaded = any(MODEL in m.get("name", "") for m in models)
            return {"loaded": loaded, "model": MODEL, "models": [m["name"] for m in models]}
    except Exception as e:
        return {"loaded": False, "model": MODEL, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def index():
    html = STATIC / "index.html"
    if html.exists():
        return HTMLResponse(html.read_text())
    return HTMLResponse("<h1>autoRecolor</h1><p>Frontend not found.</p>")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn
    uvicorn.run("autoRecolor.server:app", host="::", port=8010, reload=True)


if __name__ == "__main__":
    main()
