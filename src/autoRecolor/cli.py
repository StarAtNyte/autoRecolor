from __future__ import annotations
import sys
import os
import copy
import json
from autoRecolor.analyze import analyze_image
from autoRecolor.agent import Agent
from autoRecolor.recolor import recolor_image


def print_palette(palette):
    for c in palette.palette:
        blocks = "█" * max(1, int(c.pixel_percent / 5))
        print(f"  [{c.id}] {blocks} {c.hex}  {c.label:15s}  {c.pixel_percent}%")
    print()


def print_palette_json(palette):
    print(json.dumps(palette.to_dict(), indent=2))
    print()


def print_diff(original, modified):
    orig_by_id = {c.id: c for c in original.palette}
    changed = False
    for c in modified.palette:
        oc = orig_by_id.get(c.id)
        if oc and oc.hex != c.hex:
            print(f"  [{c.id}] {oc.hex} → {c.hex}  ({oc.label} → {c.label})")
            changed = True
    if not changed:
        print("  (no color changes made)")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cli.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        sys.exit(1)

    print(f"Analyzing {image_path}...")
    palette, img, label_map, _ = analyze_image(image_path, n_colors=6)
    original_palette = copy.deepcopy(palette)
    print_palette(palette)

    agent = Agent(palette)

    # History stores (palette_snapshot, messages_snapshot) pairs so undo
    # restores both palette state AND conversation context together.
    history: list[tuple] = [(copy.deepcopy(palette), copy.deepcopy(agent.messages))]

    print("Commands:")
    print("  <prompt>   describe the recolor you want")
    print("  json       show raw palette JSON")
    print("  preview    save a recolored preview")
    print("  undo       revert last change")
    print("  quit       exit and save final result")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            break

        if prompt.lower() == "undo":
            if len(history) > 1:
                history.pop()
                palette_snap, messages_snap = history[-1]
                agent.palette = copy.deepcopy(palette_snap)
                agent.messages = copy.deepcopy(messages_snap)
                print("Undone. Current palette:")
                print_palette(agent.palette)
            else:
                print("Nothing to undo.")
            continue

        if prompt.lower() == "json":
            print_palette_json(agent.palette)
            continue

        if prompt.lower() == "preview":
            _save_preview(img, original_palette, agent.palette, label_map, image_path)
            continue

        # Run agentic edit
        before = copy.deepcopy(agent.palette)
        result = agent.chat_stream(prompt)
        print()
        print_diff(before, agent.palette)

        # Save snapshot after each turn (palette + messages) for undo
        history.append((copy.deepcopy(agent.palette), copy.deepcopy(agent.messages)))

    # Final recolor on exit
    print("\nRecoloring image...")
    recolored = recolor_image(img, original_palette, agent.palette, label_map)
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_recolored{ext}"
    recolored.save(output_path)
    print(f"Saved: {output_path}")

    # Save palette strip preview
    from PIL import Image, ImageDraw
    from autoRecolor.utils import hex_to_rgb
    preview_h = 60
    preview = Image.new("RGB", (agent.palette.image_width, preview_h), (255, 255, 255))
    draw = ImageDraw.Draw(preview)
    n = len(agent.palette.palette)
    if n > 0:
        seg_w = agent.palette.image_width // n
        for i, c in enumerate(agent.palette.palette):
            draw.rectangle(
                [i * seg_w, 0, (i + 1) * seg_w, preview_h],
                fill=tuple(hex_to_rgb(c.hex)),
            )
    palette_preview_path = f"{base}_palette_preview{ext}"
    preview.save(palette_preview_path)
    print(f"Saved palette preview: {palette_preview_path}")


def _save_preview(img, original_palette, modified_palette, label_map, image_path):
    recolored = recolor_image(img, original_palette, modified_palette, label_map)
    base, ext = os.path.splitext(image_path)
    preview_path = f"{base}_preview{ext}"
    recolored.save(preview_path)
    print(f"  Preview saved: {preview_path}")


if __name__ == "__main__":
    main()
