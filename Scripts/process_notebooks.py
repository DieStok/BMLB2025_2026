#!/usr/bin/env python3
"""
Build student notebooks from *_Answers.ipynb.

Supports explicit solution blocks inside code cells:

    # answer
    <solution code>
    # end answer

- Multiple blocks per cell are supported.
- Pre/post non-solution code is kept.
- Solution blocks become separate code cells tagged 'solution' in the answers file.
- In the student build, every 'solution' cell is replaced by a placeholder code cell:
      # Your answer here

Fallback legacy mode also works if only a single '# answer' (or '# solution') is present.

The script:
- Normalizes *_Answers.ipynb in-place (split + tag) — idempotent.
- Builds the student notebook (replace 'solution' with placeholders + clear outputs).
- Only overwrites student files when content actually changes.
- In pre-commit, if the answers nb is normalized, the hook exits nonzero to force re-stage.

Usage:
    Scripts/process_notebooks.py <paths...>
    Scripts/process_notebooks.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import nbformat

# ---- Config ----
PLACEHOLDER_TEXT = "# Your answer here"  # customize if you want

# ---- Marker definitions ----
# Explicit block markers (strongly preferred)
START_RE = re.compile(r"^\s*#+\s*answer\s*$", re.IGNORECASE)
END_RE = re.compile(r"^\s*#+\s*end\s*answer\s*$", re.IGNORECASE)

# Legacy single-line marker (fallback)
LEGACY_RE = re.compile(r"^\s*#+\s*(answer|solution)\b.*$", re.IGNORECASE)


def is_answers_notebook(p: Path) -> bool:
    return p.suffix == ".ipynb" and p.name.endswith("_Answers.ipynb")


def answers_to_student_path(p: Path) -> Path:
    return p.with_name(p.name.replace("_Answers.ipynb", ".ipynb"))


def ensure_tag(cell, tag: str):
    md = cell.get("metadata", {})
    tags = md.get("tags", [])
    if tag not in tags:
        tags.append(tag)
    md["tags"] = tags
    cell["metadata"] = md


def has_tag(cell, tag: str) -> bool:
    return tag in cell.get("metadata", {}).get("tags", [])


# ---- Splitting helpers ----


def split_code_cell_explicit_blocks(cell) -> List:
    """
    Split a code cell into alternating kept/solution/kept/... segments based on
    explicit '# answer' ... '# end answer' blocks. Returns [cell] if no blocks.

    Multiple non-overlapping blocks supported. Unmatched start/end:
    - stray '# answer' with no '# end answer' => legacy single-marker fallback
    - stray '# end answer' is ignored
    """
    if cell.get("cell_type") != "code":
        return [cell]

    src = cell.get("source") or ""
    lines = src.splitlines(keepends=False)

    # Quick presence check
    has_start = any(START_RE.match(ln) for ln in lines)
    if not has_start:
        return [cell]

    new_cells = []
    cursor = 0
    n = len(lines)

    while cursor < n:
        if START_RE.match(lines[cursor]):
            # emit kept segment before the start, if any
            kept = lines[:cursor]
            # find end
            j = cursor + 1
            end_idx = None
            while j < n:
                if END_RE.match(lines[j]):
                    end_idx = j
                    break
                j += 1
            if end_idx is None:
                # unmatched -> legacy fallback for whole cell
                return split_cell_on_legacy_marker(cell)

            # solution block is (cursor+1 .. end_idx-1)
            sol_block = lines[cursor + 1 : end_idx]

            if kept:
                kept_cell = deepcopy(cell)
                kept_cell["source"] = _join(kept)
                if has_tag(kept_cell, "solution"):
                    kept_cell["metadata"]["tags"].remove("solution")
                new_cells.append(kept_cell)

            sol_cell = deepcopy(cell)
            sol_cell["source"] = _join(sol_block)
            ensure_tag(sol_cell, "solution")
            new_cells.append(sol_cell)

            # continue on tail after end marker
            lines = lines[end_idx + 1 :]
            n = len(lines)
            cursor = 0
            continue
        else:
            cursor += 1

    # Emit remaining tail as kept
    if lines:
        kept_tail = deepcopy(cell)
        kept_tail["source"] = _join(lines)
        if has_tag(kept_tail, "solution"):
            kept_tail["metadata"]["tags"].remove("solution")
        new_cells.append(kept_tail)

    return new_cells if new_cells else [cell]


def split_cell_on_legacy_marker(cell) -> List:
    """
    Legacy single-marker split:
        <setup>
        # answer
        <solution>
    Produces setup (kept) + solution (tagged).
    If the first non-empty line is the marker, treat as pure solution cell.
    """
    if cell.get("cell_type") != "code":
        return [cell]
    src = cell.get("source") or ""
    lines = src.splitlines(keepends=False)

    # First non-empty line marker => pure solution
    for idx, line in enumerate(lines):
        if line.strip():
            if LEGACY_RE.match(line):
                sol = deepcopy(cell)
                sol["source"] = _join(lines[idx + 1 :])
                ensure_tag(sol, "solution")
                return [sol]
            break

    # Find first marker anywhere
    marker_idx = None
    for i, line in enumerate(lines):
        if LEGACY_RE.match(line):
            marker_idx = i
            break
    if marker_idx is None:
        return [cell]

    kept = lines[:marker_idx]
    sol_block = lines[marker_idx + 1 :]

    new_cells = []
    if kept:
        kept_cell = deepcopy(cell)
        kept_cell["source"] = _join(kept)
        if has_tag(kept_cell, "solution"):
            kept_cell["metadata"]["tags"].remove("solution")
        new_cells.append(kept_cell)

    sol_cell = deepcopy(cell)
    sol_cell["source"] = _join(sol_block)
    ensure_tag(sol_cell, "solution")
    new_cells.append(sol_cell)
    return new_cells


def _join(lines: List[str]) -> str:
    txt = "\n".join(lines).rstrip()
    return (txt + "\n") if txt else ""


def normalize_answers_notebook(nb) -> nbformat.NotebookNode:
    """
    For each cell:
    - If code cell contains explicit blocks, split into kept/solution/kept/etc.
    - Else, apply legacy single-marker split (if present).
    """
    new_cells = []
    for c in nb.cells:
        if c.get("cell_type") == "code":
            parts = split_code_cell_explicit_blocks(c)
            if parts == [c]:
                parts = split_cell_on_legacy_marker(c)
            new_cells.extend(parts)
        else:
            new_cells.append(c)
    nb.cells = new_cells
    return nb


# ---- Build helpers ----


def clear_outputs_and_counts(nb):
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb.metadata.pop("widgets", None)
    return nb


def _solution_to_placeholder(cell, text=PLACEHOLDER_TEXT):
    """Return a copy of a solution-tagged code cell turned into a placeholder."""
    new = deepcopy(cell)
    # Always a code cell here; we keep metadata except 'solution' tag
    md = new.get("metadata", {})
    tags = [t for t in md.get("tags", []) if t != "solution"]
    if tags:
        md["tags"] = tags
    else:
        md.pop("tags", None)
    new["metadata"] = md
    new["source"] = (text.rstrip() + "\n") if text else ""
    new["outputs"] = []
    new["execution_count"] = None
    return new


def studentize_with_placeholders(nb) -> nbformat.NotebookNode:
    """Replace each solution-tagged code cell with a placeholder; keep others; clear outputs."""
    nb2 = deepcopy(nb)
    replaced = []
    for c in nb2.cells:
        if c.get("cell_type") == "code" and "solution" in c.get("metadata", {}).get(
            "tags", []
        ):
            replaced.append(_solution_to_placeholder(c))
        else:
            # keep as-is (we'll clear outputs after)
            replaced.append(deepcopy(c))
    nb2.cells = replaced
    clear_outputs_and_counts(nb2)
    return nb2


def read_nb(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


def write_nb(path: Path, nb: nbformat.NotebookNode):
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)


def notebooks_equal(nb_a, nb_b) -> bool:
    a = deepcopy(nb_a)
    b = deepcopy(nb_b)
    for nb in (a, b):
        nb.metadata.pop("signature", None)
        nb.metadata.pop("language_info", None)
        nb.metadata.pop("orig_nbformat", None)
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(
        b, sort_keys=True, ensure_ascii=False
    )


def process_answers_notebook(ans_path: Path) -> Tuple[bool, bool, Path]:
    nb = read_nb(ans_path)
    nb_before = deepcopy(nb)

    # 1) Normalize answers: split/tag
    nb = normalize_answers_notebook(nb)
    answers_modified = not notebooks_equal(nb, nb_before)
    if answers_modified:
        write_nb(ans_path, nb)

    # 2) Student build: placeholders instead of solutions + clear outputs
    student_nb = studentize_with_placeholders(nb)

    student_path = answers_to_student_path(ans_path)
    if student_path.exists():
        existing_student = read_nb(student_path)
        existing_student_clean = studentize_with_placeholders(existing_student)
        student_modified = not notebooks_equal(student_nb, existing_student_clean)
    else:
        student_modified = True

    if student_modified:
        write_nb(student_path, student_nb)

    return answers_modified, student_modified, student_path


def find_all_answers_notebooks(root: Path):
    return [p for p in root.rglob("*_Answers.ipynb") if p.is_file()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Paths from pre-commit (or use --all)")
    ap.add_argument(
        "--all", action="store_true", help="Process all *_Answers.ipynb in repo"
    )
    args = ap.parse_args()

    if args.all:
        targets = find_all_answers_notebooks(Path(".").resolve())
    else:
        targets = [
            Path(p).resolve() for p in args.paths if is_answers_notebook(Path(p))
        ]

    if not targets:
        return 0

    exit_code = 0
    for ans in targets:
        try:
            a_mod, s_mod, s_path = process_answers_notebook(ans)
            msg = []
            if a_mod:
                msg.append(f"normalized answers (split/tag): {ans}")
            if s_mod:
                msg.append(f"updated student: {s_path}")
            if not msg:
                msg.append(f"no changes needed: {ans.name}")
            print(" | ".join(msg))
            if a_mod:
                exit_code = 1  # force re-stage if we changed *_Answers.ipynb
        except Exception as e:
            print(f"[ERROR] {ans}: {e}", file=sys.stderr)
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
