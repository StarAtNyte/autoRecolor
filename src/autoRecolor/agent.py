from __future__ import annotations
import json
import math
import sys
import requests
from autoRecolor.palette import Palette


OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3.6:27b"

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    # ── View ──────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "get_palette",
            "description": (
                "View the current palette. Each color shows: id, label, hex, "
                "OKLAB coords (L=lightness 0-1, a=green↔red, b=blue↔yellow), "
                "chroma (colorfulness, 0=gray ~0.4=vivid), hue_angle (degrees in ab-plane), "
                "and pixel_percent. Also shows the overall harmony score."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ── Exact color set ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "update_color",
            "description": "Set a palette color to an exact hex value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer", "description": "ID of the color to update"},
                    "hex": {"type": "string", "description": "New hex color e.g. #FF6B35"},
                },
                "required": ["color_id", "hex"],
            },
        },
    },

    # ── Per-color OKLAB axis adjust ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "adjust_color",
            "description": (
                "Adjust a single color using perceptual OKLCH axes. "
                "All axes are optional — include only what you want to change. "
                "Axes and useful ranges:\n"
                "  lightness   (-0.5 to +0.5)  — additive L shift; +0.1 = noticeably brighter\n"
                "  warmth      (-0.2 to +0.2)  — along blackbody locus; + = warmer amber/orange\n"
                "  chroma      (-0.3 to +0.3)  — radial ab push, hue angle unchanged; + = more vivid\n"
                "  hue         (-180 to +180)  — rotate hue in ab-plane in degrees; +30 = 30° shift\n"
                "  exposure    (-2.0 to +2.0)  — multiply L by 2^delta (photo stops); +1 = 2× brighter\n"
                "  saturation  (-1.0 to +1.0)  — proportional chroma scale; +0.5 = 50% more saturated\n"
                "  purity      (-0.5 to +0.5)  — jewel-tone: + = darker+more chromatic; - = pastel\n"
                "  mute        (-0.5 to +0.5)  — + = pull toward mid gray (neutral/corporate)\n"
                "  green_red   (-0.2 to +0.2)  — direct a-shift; + = redder, - = greener\n"
                "  blue_yellow (-0.2 to +0.2)  — direct b-shift; + = yellower, - = bluer\n"
                "Multiple axes can be combined in one call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer", "description": "ID of the color to adjust"},
                    "axes": {
                        "type": "object",
                        "description": "Dict of axis → delta value. Only include axes you want to change.",
                        "properties": {
                            "lightness":   {"type": "number"},
                            "warmth":      {"type": "number"},
                            "chroma":      {"type": "number"},
                            "hue":         {"type": "number"},
                            "exposure":    {"type": "number"},
                            "saturation":  {"type": "number"},
                            "purity":      {"type": "number"},
                            "mute":        {"type": "number"},
                            "green_red":   {"type": "number"},
                            "blue_yellow": {"type": "number"},
                        },
                    },
                },
                "required": ["color_id", "axes"],
            },
        },
    },

    # ── Palette-wide OKLAB axis adjust ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "adjust_palette",
            "description": (
                "Apply OKLCH axis shifts to ALL colors at once (except any in exclude_ids). "
                "Same axes as adjust_color, PLUS:\n"
                "  contrast (-0.5 to +1.0) — expand/compress all L values around palette mean; "
                "+ = more contrast between dark and light colors\n"
                "Use this for global mood changes: overall warmth, contrast boost, desaturation, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "axes": {
                        "type": "object",
                        "description": "Dict of axis → delta value.",
                        "properties": {
                            "lightness":   {"type": "number"},
                            "warmth":      {"type": "number"},
                            "chroma":      {"type": "number"},
                            "hue":         {"type": "number"},
                            "exposure":    {"type": "number"},
                            "saturation":  {"type": "number"},
                            "purity":      {"type": "number"},
                            "mute":        {"type": "number"},
                            "green_red":   {"type": "number"},
                            "blue_yellow": {"type": "number"},
                            "contrast":    {"type": "number"},
                        },
                    },
                    "exclude_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Color IDs to skip (e.g. a locked base color). Can be empty.",
                    },
                },
                "required": ["axes"],
            },
        },
    },

    # ── Absolute lightness set ─────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_lightness",
            "description": (
                "Set the absolute lightness (L) of a color, keeping its hue and chroma. "
                "L = 0.0 is black, L = 1.0 is white. "
                "Useful when you want an exact tone, e.g. 'make this color mid-tone (L=0.5)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer"},
                    "L": {"type": "number", "description": "Target lightness 0.0 (black) to 1.0 (white)"},
                },
                "required": ["color_id", "L"],
            },
        },
    },

    # ── Absolute chroma set ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "set_chroma",
            "description": (
                "Set the absolute chroma (colorfulness) of a color, keeping lightness and hue. "
                "C = 0.0 is a pure achromatic gray. C ≈ 0.1 is muted, C ≈ 0.2 is moderate, "
                "C ≈ 0.3–0.4 is very vivid/saturated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer"},
                    "C": {"type": "number", "description": "Target chroma 0.0 (gray) to ~0.4 (vivid)"},
                },
                "required": ["color_id", "C"],
            },
        },
    },

    # ── Lightness matching ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "match_lightness",
            "description": (
                "Copy the lightness (L) of one color onto another color. "
                "The target color keeps its hue and chroma — only its brightness changes. "
                "Useful for tonal alignment, e.g. 'make the shadow the same brightness as the highlight'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "integer", "description": "Color to copy lightness FROM"},
                    "target_id": {"type": "integer", "description": "Color to apply lightness TO"},
                },
                "required": ["source_id", "target_id"],
            },
        },
    },

    # ── Harmony score ──────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "rate_palette",
            "description": (
                "Compute the Ou et al. perceptual harmony score for the current palette. "
                "Returns a scalar; higher is more harmonious (typical range -1 to +1). "
                "Use this to verify that edits improved the palette, or to compare options."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },

    # ── Label ──────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "relabel_color",
            "description": "Rename a color's label to better describe what it represents in the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer"},
                    "label":    {"type": "string", "description": "New descriptive label"},
                },
                "required": ["color_id", "label"],
            },
        },
    },

    # ── Finalize ───────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Call when all palette modifications are complete and ready to apply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief summary of all changes made and why.",
                    },
                },
                "required": ["reasoning"],
            },
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert colorist AI. You edit image palettes using perceptual color science tools.

━━ OKLCH color space ━━
  L  = lightness    (0=black … 1=white)
  C  = chroma       — colorfulness (0=gray, ~0.4=vivid); √(a²+b²) in ab-plane
  h  = hue_angle    — hue direction in degrees (atan2(b,a)); 0°=red, 90°=yellow, 180°=cyan, 270°=blue
  Cartesian axes (OKLAB):
    a = green↔red    (−=greener, +=redder)
    b = blue↔yellow  (−=bluer,  +=yellower)
  The `hue` tool axis rotates h in degrees. The `chroma` axis shifts C radially.
  All pixel transforms happen in OKLCH — hue rotations are exact and artifact-free.

━━ Hue angle targets (degrees) ━━
  red=15  orange=45  yellow=75  lime=110  green=150
  cyan=185  blue=230  indigo=260  purple=290  magenta=320  pink=345

  HOW TO SHIFT HUE (do this every time):
  1. Read the hue_angle of the colors in the palette below.
  2. Pick the target angle from the table above.
  3. raw_Δ = target − current_hue_angle.
  4. Wrap to shortest path: if raw_Δ > 180 subtract 360; if raw_Δ < −180 add 360.
  5. Call adjust_palette with that wrapped Δhue value.

  Examples (shortest-path wrapping):
    green(150) → blue(230):     230−150 =  +80            → {{"hue":  80}}
    orange(45) → purple(290):   290−45  = +245 → 245−360 = −115 → {{"hue": −115}}
    blue(230)  → red(15):       15−230  = −215 → −215+360 = +145 → {{"hue": +145}}
    cyan(185)  → magenta(320):  320−185 = +135            → {{"hue": 135}}
    pink(345)  → lime(110):     110−345 = −235 → −235+360 = +125 → {{"hue": 125}}
    yellow(75) → indigo(260):   260−75  = +185 → 185−360 = −175 → {{"hue": −175}}

━━ Lightness / chroma quick reads ━━
  L < 0.25 = very dark   L 0.25–0.45 = dark   L 0.45–0.60 = mid
  L 0.60–0.75 = light    L > 0.75 = very light
  chroma < 0.08 = muted/gray   0.08–0.18 = moderate   > 0.18 = vivid

━━ Tool selection ━━
  adjust_palette  — global changes (hue family, mood, contrast, saturation)
  adjust_color    — single color tweak
  set_lightness   — exact tone (L value)
  set_chroma      — exact colorfulness (C value)
  match_lightness — copy brightness from one color to another
  update_color    — only when user gives an explicit hex code
  rate_palette    — check harmony score
  relabel_color   — rename after hue/character changes
  finalize        — call when done

━━ Mood recipes ━━
  warmer          → adjust_palette {{"warmth": 0.10, "blue_yellow": 0.05}}
  cooler          → adjust_palette {{"warmth": -0.10, "blue_yellow": -0.05}}
  vintage/retro   → adjust_palette {{"mute": 0.15, "warmth": 0.06, "contrast": -0.10}}
  dark/moody      → adjust_palette {{"exposure": -0.40, "contrast": 0.20}}
  bright/airy     → adjust_palette {{"exposure": 0.20, "saturation": 0.15}}
  vibrant/pop     → adjust_palette {{"chroma": 0.08, "contrast": 0.15}}
  pastel          → adjust_palette {{"purity": -0.20, "lightness": 0.10}}
  jewel tones     → adjust_palette {{"purity": 0.20, "contrast": 0.10}}
  desaturated     → adjust_palette {{"saturation": -0.40}}
  B&W             → adjust_palette {{"saturation": -1.0, "contrast": 0.50}}
  complementary   → adjust_palette {{"hue": 180}}

━━ Workflow (follow this order) ━━
1. Read the palette values below — note hue_angle, L, chroma for each color.
2. State your plan in one line (what you'll change and why).
3. Compute exact deltas from the actual palette values, not from memory.
4. Issue ALL tool calls in one turn.
5. After any hue/character change, relabel the affected colors.
6. Call rate_palette, then finalize.

━━ Current palette ━━
{palette_json}"""


# ── Tool response slimmer ─────────────────────────────────────────────────────

def _slim_tool_result(func_name: str, result: dict) -> dict:
    """
    Reduce the token cost of tool results that contain redundant palette dumps.

    adjust_palette returns the full palette dict on every call — after the
    system prompt is already synced, sending the whole thing back wastes ~400
    tokens per turn.  Instead we return a compact summary so the model still
    gets confirmation without ballooning the context.
    """
    if func_name == "adjust_palette" and "palette" in result:
        # Replace the full palette dump with a compact per-color summary
        colors = result["palette"].get("palette", [])
        summary = [
            {"id": c["id"], "label": c["label"], "hex": c["hex"],
             "L": round(c["oklab"][0], 3) if c.get("oklab") else None,
             "chroma": c.get("chroma")}
            for c in colors
        ]
        return {"success": True, "applied_to": len(colors), "colors": summary}

    if func_name == "get_palette" and "palette" in result:
        # get_palette is fine to return in full — it's an intentional read
        return result

    return result


# ── Constants ─────────────────────────────────────────────────────────────────

_PALETTE_MUTATING = {
    "update_color", "adjust_color", "adjust_palette",
    "set_lightness", "set_chroma", "match_lightness",
}

# Tool responses that include a full palette dump — trim after keeping for 2 turns
_HEAVY_TOOLS = {"adjust_palette", "get_palette"}

# Max number of non-system messages to keep in the window.
# Old *tool* result messages are condensed after this threshold to prevent OOM.
_MAX_HISTORY = 40


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    def __init__(self, palette: Palette):
        self.palette = palette
        self.messages: list[dict] = []
        self._init_messages()

    def _init_messages(self) -> None:
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(palette_json=self.palette.to_json())},
        ]

    def _sync_system_prompt(self) -> None:
        """Keep the system message's palette snapshot up to date after edits."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = SYSTEM_PROMPT.format(
                palette_json=self.palette.to_json()
            )

    def _trim_context(self) -> None:
        """
        Once the history grows past _MAX_HISTORY non-system messages, compact
        old *tool* result messages that contain heavy palette dumps to a short
        summary stub.  This keeps the context window under control without
        losing conversational coherence.
        """
        non_sys = [m for m in self.messages if m["role"] != "system"]
        if len(non_sys) <= _MAX_HISTORY:
            return

        cutoff = len(self.messages) - _MAX_HISTORY
        for i, msg in enumerate(self.messages[:cutoff]):
            if msg["role"] == "tool":
                try:
                    data = json.loads(msg["content"])
                    # Replace a heavy palette dump with a stub
                    if "palette" in data and isinstance(data["palette"], dict):
                        self.messages[i]["content"] = json.dumps({"trimmed": True, "success": True})
                except (json.JSONDecodeError, TypeError):
                    pass

    def chat_stream(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        return self._run_loop_stream()

    def _run_loop_stream(self) -> str:
        max_turns = 15
        for _turn in range(max_turns):
            self._trim_context()
            content, thinking, tool_calls = self._call_llm_stream()

            print()  # newline after streamed thinking

            if tool_calls:
                history_tool_calls = []
                for tc in tool_calls:
                    raw_args = tc["function"]["arguments"]
                    if isinstance(raw_args, str):
                        try:
                            dict_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            dict_args = {}
                    else:
                        dict_args = raw_args
                    history_tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": dict_args},
                    })

                self.messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": history_tool_calls,
                })

                finalize_result: str | None = None

                for tc, htc in zip(tool_calls, history_tool_calls):
                    func_name = htc["function"]["name"]
                    args = htc["function"]["arguments"]
                    print(f"  ▶ {func_name}({json.dumps(args)})")
                    result = self._execute_tool(func_name, args)

                    # Trim heavy palette responses to keep tool messages lean
                    tool_content = _slim_tool_result(func_name, result)
                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps(tool_content),
                    })

                    if func_name == "finalize":
                        finalize_result = result.get("reasoning", "Edits complete.")
                        print(f"  ✓ {finalize_result}")

                    if func_name in _PALETTE_MUTATING:
                        self._sync_system_prompt()

                if finalize_result is not None:
                    return finalize_result

                # Escalating finalize pressure after the first turn
                nudge = (
                    "Good. If you're satisfied with all changes, call finalize now."
                    if _turn == 0
                    else "All requested changes should now be applied. Call finalize to complete."
                )
                self.messages.append({"role": "user", "content": nudge})

            else:
                self.messages.append({"role": "assistant", "content": content})
                if content:
                    print(f"  {content}")
                return content

        return "Reached maximum edit turns."

    def _call_llm_stream(self) -> tuple[str, str, list | None]:
        payload = {
            "model": MODEL,
            "messages": self.messages,
            "tools": TOOLS,
            "stream": True,
            "think": True,                      # enable Qwen3 reasoning tokens
            "options": {"temperature": 0.4, "num_ctx": 8192},
        }
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=300,
        )
        if not resp.ok:
            raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:400]}")

        content = ""
        thinking = ""
        tool_calls_acc: dict[int, dict] = {}

        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})

            if "thinking" in msg:
                t = msg["thinking"]
                thinking += t
                sys.stdout.write(t)
                sys.stdout.flush()

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
        return content, thinking, tool_calls

    def _execute_tool(self, name: str, args: dict) -> dict:
        try:
            match name:
                case "get_palette":
                    return self.palette.to_dict()

                case "update_color":
                    self.palette.update_hex(args["color_id"], args["hex"])
                    return {"success": True, "entry": self.palette.get_entry(args["color_id"]).to_info_dict()}

                case "adjust_color":
                    self.palette.adjust_oklab_axes(args["color_id"], args["axes"])
                    entry = self.palette.get_entry(args["color_id"])
                    return {"success": True, "entry": entry.to_info_dict()}

                case "adjust_palette":
                    self.palette.adjust_palette_axes(
                        args["axes"],
                        exclude_ids=args.get("exclude_ids", []),
                    )
                    return {"success": True, "palette": self.palette.to_dict()}

                case "set_lightness":
                    self.palette.set_lightness(args["color_id"], float(args["L"]))
                    entry = self.palette.get_entry(args["color_id"])
                    return {"success": True, "entry": entry.to_info_dict()}

                case "set_chroma":
                    self.palette.set_chroma(args["color_id"], float(args["C"]))
                    entry = self.palette.get_entry(args["color_id"])
                    return {"success": True, "entry": entry.to_info_dict()}

                case "match_lightness":
                    self.palette.match_lightness(args["source_id"], args["target_id"])
                    entry = self.palette.get_entry(args["target_id"])
                    return {"success": True, "entry": entry.to_info_dict()}

                case "rate_palette":
                    score = self.palette.harmony_score()
                    return {"harmony_score": score, "interpretation":
                            "good" if score > 0.2 else "neutral" if score > -0.2 else "low"}

                case "relabel_color":
                    self.palette.relabel(args["color_id"], args["label"])
                    return {"success": True}

                case "finalize":
                    return {"success": True, "reasoning": args.get("reasoning", "")}

                case _:
                    return {"error": f"Unknown tool: {name}"}

        except Exception as e:
            return {"error": str(e)}
