#!/usr/bin/env python3
"""
Extract function-replacement ideas from an LLM response and produce
one code file per idea with the relevant function(s) swapped in.

Usage
-----
    python replace_functions.py --code base.py --ideas ideas.md [--out-dir variants]

The script parses the ideas file (Markdown / plain text) looking for
fenced Python code blocks.  Each block that contains a `def ...` is
treated as a replacement for the function of the same name in the base
code.  Blocks that belong to the same "Idea N" section are grouped
together so they end up in a single output file.

Output files are named  <original_stem>_idea_<N>.py  and placed in
--out-dir (default: ./variants).
"""

from __future__ import annotations

import argparse
import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ── dataclasses ──────────────────────────────────────────────────────

@dataclass
class FunctionSnippet:
    """A single replacement function extracted from the LLM response."""
    name: str
    source: str          # dedented source ready for insertion
    first_line: int = 0  # cosmetic, for debug messages


@dataclass
class Idea:
    """One logical idea that may touch one or more functions."""
    label: str                                 # e.g. "Idea 1"
    description: str = ""
    snippets: list[FunctionSnippet] = field(default_factory=list)


# ── parsing helpers ──────────────────────────────────────────────────

_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL
)

_IDEA_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?(?:\*\*)?Idea\s+(\d+)[:\s.\-–—]*([^\n]*)",
    re.IGNORECASE,
)


def _extract_functions(code: str) -> list[FunctionSnippet]:
    """Return every top-level function definition found in *code*."""
    snippets: list[FunctionSnippet] = []
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return snippets
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            snippets.append(FunctionSnippet(
                name=node.name,
                source=ast.get_source_segment(textwrap.dedent(code), node) or "",
                first_line=node.lineno,
            ))
    return snippets


def parse_ideas(text: str) -> list[Idea]:
    """
    Walk through the LLM response and group code blocks into Ideas.

    Strategy:
      1. Find every "Idea N" heading → record its position.
      2. Find every fenced python block → record its position.
      3. Assign each block to the nearest preceding Idea heading.
         Blocks before the first heading go into "Idea 0".
    """
    # Collect idea headings with their positions
    headings: list[tuple[int, str, str]] = []  # (pos, label, description)
    for m in _IDEA_HEADING_RE.finditer(text):
        label = f"Idea {m.group(1)}"
        desc = m.group(2).strip().rstrip("*")
        headings.append((m.start(), label, desc))

    # Collect fenced code blocks with their positions
    blocks: list[tuple[int, str]] = []  # (pos, code)
    for m in _FENCE_RE.finditer(text):
        blocks.append((m.start(), m.group(1)))

    if not blocks:
        return []

    # Build Idea objects
    ideas_map: dict[str, Idea] = {}
    idea_order: list[str] = []

    def _current_idea(pos: int) -> Idea:
        label = "Idea 0"
        for hpos, hlabel, hdesc in headings:
            if hpos <= pos:
                label = hlabel
                desc = hdesc
            else:
                break
        else:
            if headings and headings[-1][0] <= pos:
                label = headings[-1][1]
                desc = headings[-1][2]
        if label not in ideas_map:
            d = desc if label != "Idea 0" else ""
            ideas_map[label] = Idea(label=label, description=d)
            idea_order.append(label)
        return ideas_map[label]

    for pos, code in blocks:
        funcs = _extract_functions(code)
        if not funcs:
            continue
        idea = _current_idea(pos)
        idea.snippets.extend(funcs)

    return [ideas_map[k] for k in idea_order]


# ── code replacement ─────────────────────────────────────────────────

def _find_class_indent(source_lines: list[str]) -> int:
    """Detect the indentation (in spaces) used for methods inside the first class."""
    for line in source_lines:
        stripped = line.lstrip()
        if stripped.startswith("def "):
            return len(line) - len(stripped)
    return 4


def replace_function_in_source(
    source: str,
    func_name: str,
    new_body: str,
) -> str:
    """
    Replace the function *func_name* in *source* with *new_body*.

    Handles both module-level functions and methods inside a class by
    detecting the indentation of the original function.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    # Find the node (could be nested inside a class)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                target = node
                # don't break – we want the *last* definition if shadowed

    if target is None:
        raise KeyError(f"Function '{func_name}' not found in source")

    # Determine the range of lines to replace
    start = target.lineno - 1  # 0-based

    # Find the end: scan forward for the next node at the same or lower
    # indentation, or end-of-file.
    orig_indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() == "" or line.strip().startswith("#"):
            end += 1
            continue
        curr_indent = len(line) - len(line.lstrip())
        if curr_indent <= orig_indent:
            break
        end += 1

    # Strip trailing blank lines that belong to the gap between functions
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1

    # Re-indent new_body to match original indentation
    new_lines = textwrap.dedent(new_body).splitlines(keepends=True)
    # ensure each line is indented properly
    indented: list[str] = []
    for l in new_lines:
        if l.strip() == "":
            indented.append("\n")
        else:
            indented.append(" " * orig_indent + l.lstrip() if not l[0].isspace()
                            else " " * orig_indent + textwrap.dedent(l).lstrip())

    # Properly indent: keep the relative indentation of the new body
    dedented = textwrap.dedent(new_body)
    re_indented_lines: list[str] = []
    for l in dedented.splitlines(keepends=True):
        if l.strip():
            stripped = l.lstrip()
            extra = len(l) - len(stripped)
            re_indented_lines.append(" " * (orig_indent + extra) + stripped)
        else:
            re_indented_lines.append("\n")

    # Make sure it ends with a newline
    if re_indented_lines and not re_indented_lines[-1].endswith("\n"):
        re_indented_lines[-1] += "\n"

    result_lines = lines[:start] + re_indented_lines + lines[end:]
    return "".join(result_lines)


# ── main logic ───────────────────────────────────────────────────────

def generate_variants(
    base_source: str,
    ideas: list[Idea],
    base_name: str = "code",
    out_dir: Path = Path("variants"),
) -> list[Path]:
    """Apply each Idea's snippets to the base source and write files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for idea in ideas:
        if not idea.snippets:
            continue
        src = base_source
        applied: list[str] = []
        for snip in idea.snippets:
            try:
                src = replace_function_in_source(src, snip.name, snip.source)
                applied.append(snip.name)
            except KeyError as exc:
                print(f"  ⚠  {idea.label}: {exc} – skipped")

        if not applied:
            continue

        slug = idea.label.lower().replace(" ", "_")
        out_path = out_dir / f"{base_name}_{slug}.py"
        out_path.write_text(src)
        funcs = ", ".join(applied)
        print(f"  ✔  {out_path.name}  (replaced: {funcs})")
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate code variants from LLM-proposed function replacements.",
    )
    parser.add_argument("--code", required=True, help="Path to the base Python file")
    parser.add_argument("--ideas", required=True, help="Path to the LLM response (Markdown / txt)")
    parser.add_argument("--out-dir", default="variants", help="Output directory (default: variants)")
    args = parser.parse_args()

    base_path = Path(args.code)
    ideas_path = Path(args.ideas)
    out_dir = Path(args.out_dir)

    base_source = base_path.read_text()
    ideas_text = ideas_path.read_text()

    ideas = parse_ideas(ideas_text)
    if not ideas:
        print("No function-replacement ideas found in the LLM response.")
        return

    print(f"Found {len(ideas)} idea(s):")
    for idea in ideas:
        funcs = ", ".join(s.name for s in idea.snippets)
        print(f"  • {idea.label}: {idea.description[:60]}  →  [{funcs}]")
    print()

    written = generate_variants(base_source, ideas, base_path.stem, out_dir)
    print(f"\nDone – wrote {len(written)} variant(s) to {out_dir}/")


if __name__ == "__main__":
    main()