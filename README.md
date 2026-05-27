# autoRecolor

**autoRecolor** is a locally-run AI image recoloring tool that lets you change the mood, palette, and feel of any image through plain-English conversation. Upload a photo or illustration, describe what you want — *"make it warmer and more contrasty"*, *"go full vintage desaturated"*, *"shift the sky to a deep teal"* — and a local LLM figures out exactly which colors to change and by how much. Everything runs on your machine via [Ollama](https://ollama.com): no cloud APIs, no data leaving your device. Under the hood the model works in OKLAB color space, a perceptually uniform representation that makes operations like "a little warmer" or "more vivid" map cleanly to real numeric adjustments. The result is pixel-accurate recoloring that respects the tonal structure and detail of the original image, delivered in seconds through a streaming chat interface with live before/after comparison and a full version history of every edit.

---

## How it works

1. **Analyze** — the image is clustered into dominant colors using HyAB K-Means in OKLAB space (perceptually uniform, so clusters are more meaningful than plain RGB). Exact-color images (pixel art, flat illustrations) skip clustering entirely.
2. **Edit** — the local LLM receives the palette and your prompt, then calls perceptual tools (`adjust_color`, `adjust_palette`, `set_lightness`, `set_chroma`, etc.) to mutate colors using named axes: *warmth, chroma, hue, exposure, purity, mute, contrast* and more.
3. **Recolor** — every pixel is re-mapped from its nearest original cluster to the new target color using smooth OKLAB interpolation, preserving fine detail while applying the palette change globally.

---

## Requirements

- Python ≥ 3.11
- [Ollama](https://ollama.com) running locally with a tool-calling model
  Default: `qwen3.6:27b` — requires ~18 GB VRAM (runs well on an RTX 3090/4090)

---

## Installation

```bash
git clone https://github.com/StarAtNyte/autoRecolor.git
cd autoRecolor
pip install -e .
```

---

## Usage

### Web UI

```bash
autorecolor-server
```

Opens at `http://localhost:8010`. Upload an image, then chat to recolor it. The model streams its reasoning and tool calls live; each edit round shows a thumbnail in the chat and adds a version to the history strip — click any to jump back to it.

### CLI

```bash
autorecolor path/to/image.png
```

Interactive terminal session with the same LLM agent and tool set.

---

## LLM tools

| Tool | What it does |
|------|-------------|
| `get_palette` | View palette: hex, OKLAB, chroma, hue angle, coverage % |
| `update_color` | Set a color to an exact hex value |
| `adjust_color` | Shift one color on named OKLAB axes (warmth, chroma, hue, exposure, purity, mute…) |
| `adjust_palette` | Apply axis shifts to all colors at once; supports `contrast` axis |
| `set_lightness` | Pin a color's absolute lightness (0=black, 1=white) |
| `set_chroma` | Pin a color's absolute colorfulness (0=gray, ~0.4=vivid) |
| `match_lightness` | Copy lightness from one color onto another |
| `rate_palette` | Compute Ou et al. perceptual harmony score |
| `relabel_color` | Rename a color's semantic label |
| `finalize` | Commit all changes and return a summary |

---

## Project structure

```
src/autoRecolor/
├── agent.py      # Ollama LLM agent, tool definitions, system prompt
├── analyze.py    # HyAB K-Means palette extraction in OKLAB space
├── cli.py        # Interactive CLI
├── palette.py    # Palette data model with full OKLAB mutation API
├── recolor.py    # Pixel remapping engine
├── server.py     # FastAPI web server + SSE streaming
├── static/       # Browser UI (single-file HTML/CSS/JS)
└── utils.py      # Color space conversions, harmony scoring
```

---

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Ollama model | `agent.py` → `MODEL` | `qwen3.6:27b` |
| Ollama base URL | `agent.py` → `OLLAMA_BASE` | `http://localhost:11434` |
| Palette colors | `server.py` → `analyze_image(..., n_colors=6)` | `6` |
| Server port | `server.py` → `main()` | `8010` |

---

## License

MIT
