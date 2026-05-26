from __future__ import annotations
import json
import sys
import requests
from autoRecolor.palette import Palette


OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3.6:27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_palette",
            "description": "View the current color palette with all colors, labels, hex values, and HSL values",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_color",
            "description": "Set a palette color to an exact hex value",
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer", "description": "ID of the color to update"},
                    "hex": {"type": "string", "description": "New hex color (e.g. #FF6B35)"},
                },
                "required": ["color_id", "hex"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_hsl",
            "description": "Shift a color's hue, saturation, or lightness by relative amounts. Hue: -360 to 360, Saturation/Lightness: -100 to 100",
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer", "description": "ID of the color to adjust"},
                    "h_shift": {"type": "number", "description": "Hue shift in degrees (-360 to 360)"},
                    "s_shift": {"type": "number", "description": "Saturation shift (-100 to 100)"},
                    "l_shift": {"type": "number", "description": "Lightness shift (-100 to 100)"},
                },
                "required": ["color_id", "h_shift", "s_shift", "l_shift"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "relabel_color",
            "description": "Rename a color's label to better describe what it represents in the image",
            "parameters": {
                "type": "object",
                "properties": {
                    "color_id": {"type": "integer", "description": "ID of the color to relabel"},
                    "label": {"type": "string", "description": "New descriptive label"},
                },
                "required": ["color_id", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Call when all palette modifications are complete and you want to apply the recoloring",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Summary of all changes made"},
                },
                "required": ["reasoning"],
            },
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert colorist AI. Edit the image palette using the available tools.

Rules:
- Make BOLD, VISIBLE changes the user can actually see.
- Batch multiple tool calls in a single response when possible.
- Keep thinking VERY BRIEF — just state what you'll change, then call tools.
- When the user is satisfied, call finalize.

Current palette:
{palette_json}"""


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
                # Ollama requires `arguments` to be a dict object in message
                # history (not a JSON string like the OpenAI wire format).
                # Build a history-safe copy before storing.
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
                    args = htc["function"]["arguments"]   # already a dict
                    print(f"  ▶ {func_name}({json.dumps(args)})")
                    result = self._execute_tool(func_name, args)

                    # Ollama tool messages: role + content only (no tool_call_id)
                    self.messages.append({
                        "role": "tool",
                        "content": json.dumps(result),
                    })

                    if func_name == "finalize":
                        finalize_result = result.get("reasoning", "Edits complete.")
                        print(f"  ✓ {finalize_result}")

                    # Keep system prompt in sync after every palette mutation
                    if func_name in ("update_color", "adjust_hsl", "relabel_color"):
                        self._sync_system_prompt()

                if finalize_result is not None:
                    return finalize_result

                # Model made edits but forgot to finalize — nudge it once.
                # Inject a user-side reminder so the next turn wraps up.
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
            raise RuntimeError(
                f"Ollama {resp.status_code}: {resp.text[:400]}"
            )

        content = ""
        thinking = ""
        # Accumulate streaming tool-call deltas keyed by their index
        tool_calls_acc: dict[int, dict] = {}

        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})

            # Stream thinking tokens to stdout as they arrive
            if "thinking" in msg:
                t = msg["thinking"]
                thinking += t
                sys.stdout.write(t)
                sys.stdout.flush()

            if "content" in msg:
                content += msg["content"]

            # Merge streamed tool-call deltas
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
                        # Ollama may send arguments as a dict (complete) or str (delta)
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
                    return {"success": True, "palette": self.palette.to_dict()}
                case "adjust_hsl":
                    self.palette.adjust_hsl(
                        args["color_id"],
                        args["h_shift"],
                        args["s_shift"],
                        args["l_shift"],
                    )
                    return {"success": True, "palette": self.palette.to_dict()}
                case "relabel_color":
                    self.palette.relabel(args["color_id"], args["label"])
                    return {"success": True, "palette": self.palette.to_dict()}
                case "finalize":
                    return {"success": True, "reasoning": args.get("reasoning", "")}
                case _:
                    return {"error": f"Unknown tool: {name}"}
        except Exception as e:
            return {"error": str(e)}
