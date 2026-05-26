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
                "Adjust a single color using perceptual OKLAB axes. "
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
                "Apply OKLAB axis shifts to ALL colors at once (except any in exclude_ids). "
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
You are an expert colorist AI. Edit the image palette using the available tools.

Color space reference (OKLAB):
  L = lightness  (0 = black, 1 = white)
  a = green↔red  (negative = greener, positive = redder)
  b = blue↔yellow (negative = bluer, positive = yellower)
  chroma = √(a²+b²)  (0 = achromatic gray, ~0.4 = very vivid)
  hue_angle = direction in ab-plane in degrees

Rules:
- Make BOLD, VISIBLE changes the user can actually see.
- Prefer adjust_color / adjust_palette over update_color — perceptual axes give more natural results.
- For global mood changes (warmer, cooler, more contrast, desaturated) use adjust_palette.
- For individual color edits use adjust_color with the most relevant axes.
- Use set_lightness when you want an exact tone; use match_lightness to align brightness between colors.
- Call rate_palette after making edits to confirm harmony improved.
- Batch multiple tool calls in a single response when possible.
- Keep thinking VERY BRIEF — state what you'll change, then call tools.
- When the user is satisfied, call finalize.

Current palette:
{palette_json}"""


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

    def chat_stream(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        return self._run_loop_stream()

    def _run_loop_stream(self) -> str:
        max_turns = 15
        for _turn in range(max_turns):
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
                palette_mutating = {
                    "update_color", "adjust_color", "adjust_palette",
                    "set_lightness", "set_chroma", "match_lightness",
                }

                for tc, htc in zip(tool_calls, history_tool_calls):
                    func_name = htc["function"]["name"]
                    args = htc["function"]["arguments"]
                    print(f"  ▶ {func_name}({json.dumps(args)})")
                    result = self._execute_tool(func_name, args)

                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                    })

                    if func_name == "finalize":
                        finalize_result = result.get("reasoning", "Edits complete.")
                        print(f"  ✓ {finalize_result}")

                    if func_name in palette_mutating:
                        self._sync_system_prompt()

                if finalize_result is not None:
                    return finalize_result

                if _turn == 0:
                    self.messages.append({
                        "role": "user",
                        "content": "Good. If you're done with all changes, call finalize now.",
                    })
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
            "options": {"temperature": 0.3},
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
