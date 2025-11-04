#!/usr/bin/env python3
"""
Build student notebooks from *_Answers.ipynb and write per-folder diff reports.

Features
- Explicit solution blocks inside code cells:
      # answer
      <solution code>
      # end answer
  (Multiple per cell supported)
- Legacy single-marker fallback (# answer / # solution).
- Answers nb is normalized in-place (split + tag 'solution') — idempotent.
- Student nb replaces each solution cell with a placeholder:
      # Your answer here
  and clears outputs/execution counts.
- Only rewrites student files if student-visible content actually changes.

NEW: Per-folder diff report (hidden Markdown)
- Path: <folder>/.practical_diff.md
- Updated if any paired notebooks in that folder changed OR the report doesn't exist.
- Compares normalized Answers vs Student:
    * Solution cells reported as "solution → placeholder"
    * Non-solution cells: unified text diff on sources
    * Outputs/metadata ignored

Pre-commit behavior:
- If we normalize *_Answers.ipynb or update the folder report, exit non-zero to force re-stage.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import nbformat

# ---- Config ----
PLACEHOLDER_TEXT = "# Your answer here"  # customize if you want

# Explicit block markers (preferred)
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


# ---------------- Splitting/normalizing ----------------


def split_code_cell_explicit_blocks(cell) -> List:
    """Split a code cell into kept/solution/kept/... by explicit '# answer' ... '# end answer'."""
    if cell.get("cell_type") != "code":
        return [cell]

    src = cell.get("source") or ""
    lines = src.splitlines(keepends=False)
    if not any(START_RE.match(ln) for ln in lines):
        return [cell]

    new_cells = []
    cursor = 0
    n = len(lines)

    while cursor < n:
        if START_RE.match(lines[cursor]):
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
                # unmatched -> legacy fallback
                return split_cell_on_legacy_marker(cell)

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

    if lines:
        kept_tail = deepcopy(cell)
        kept_tail["source"] = _join(lines)
        if has_tag(kept_tail, "solution"):
            kept_tail["metadata"]["tags"].remove("solution")
        new_cells.append(kept_tail)

    return new_cells if new_cells else [cell]


def split_cell_on_legacy_marker(cell) -> List:
    """Legacy single-marker split (<setup>, '# answer', <solution>)."""
    if cell.get("cell_type") != "code":
        return [cell]
    lines = (cell.get("source") or "").splitlines(keepends=False)

    # First non-empty line marker => pure solution
    for idx, line in enumerate(lines):
        if line.strip():
            if LEGACY_RE.match(line):
                sol = deepcopy(cell)
                sol["source"] = _join(lines[idx + 1 :])
                ensure_tag(sol, "solution")
                return [sol]
            break

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
    """Split explicit blocks and legacy markers; tag 'solution' cells."""
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


# ---------------- Build student with placeholders ----------------


def clear_outputs_and_counts(nb):
    for cell in nb.cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb.metadata.pop("widgets", None)
    return nb


def _solution_to_placeholder(cell, text=PLACEHOLDER_TEXT):
    new = deepcopy(cell)
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
    nb2 = deepcopy(nb)
    replaced = []
    for c in nb2.cells:
        if c.get("cell_type") == "code" and "solution" in c.get("metadata", {}).get(
            "tags", []
        ):
            replaced.append(_solution_to_placeholder(c))
        else:
            replaced.append(deepcopy(c))
    nb2.cells = replaced
    clear_outputs_and_counts(nb2)
    return nb2


# ---------------- IO / equality ----------------


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


# ---------------- Diff report ----------------


def _cell_label(c) -> str:
    t = c.get("cell_type")
    if t == "markdown":
        return "markdown"
    if t == "code":
        return "code"
    return t or "cell"


def _unified(src_a: str, src_b: str, fromfile: str, tofile: str) -> str:
    a = (src_a or "").splitlines()
    b = (src_b or "").splitlines()
    diff = difflib.unified_diff(a, b, fromfile=fromfile, tofile=tofile, lineterm="")
    return "\n".join(diff)


def generate_pair_diff_md(
    ans_nb: nbformat.NotebookNode, stu_nb: nbformat.NotebookNode, title: str
) -> str:
    """
    Produce a markdown diff between normalized Answers and Student:
    - Solution cells -> reported as 'solution → placeholder' (no code shown)
    - Others: unified diff on 'source' only
    """
    lines = []
    lines.append(f"### {title}")
    lines.append("")
    ca = ans_nb.cells
    cb = stu_nb.cells
    i = j = 0
    cell_id = 1

    while i < len(ca) and j < len(cb):
        a = ca[i]
        b = cb[j]
        a_is_sol = ("solution" in a.get("metadata", {}).get("tags", [])) and a.get(
            "cell_type"
        ) == "code"

        # Try to keep them aligned 1:1; student has placeholders for solutions, so length should match.
        label = _cell_label(a)
        lines.append(f"#### Cell {cell_id} ({label})")
        cell_id += 1

        if a_is_sol:
            # solution vs placeholder
            lines.append("- Answers: **solution code**")
            lines.append(f"- Student: **placeholder** (`{PLACEHOLDER_TEXT}`)")
            lines.append("")
        else:
            # Show diff only if sources differ
            src_a = a.get("source") or ""
            src_b = b.get("source") or ""
            if src_a != src_b:
                ud = _unified(src_a, src_b, "answers", "student")
                if ud.strip():
                    lines.append("```diff")
                    lines.append(ud)
                    lines.append("```")
                    lines.append("")
                else:
                    # No textual difference (shouldn't happen with !=, but safe)
                    lines.append("_No visible difference in source._")
                    lines.append("")
            else:
                lines.append("_No changes._")
                lines.append("")
        i += 1
        j += 1

    # If lengths differ (shouldn't with our pipeline), note extras.
    while i < len(ca):
        a = ca[i]
        label = _cell_label(a)
        lines.append(f"#### Cell {cell_id} ({label})")
        lines.append("_Present in Answers only._")
        lines.append("")
        i += 1
        cell_id += 1
    while j < len(cb):
        b = cb[j]
        label = _cell_label(b)
        lines.append(f"#### Cell {cell_id} ({label})")
        lines.append("_Present in Student only._")
        lines.append("")
        j += 1
        cell_id += 1

    return "\n".join(lines)


def write_folder_diff_report(
    folder: Path,
    normalized_answers: Dict[Path, nbformat.NotebookNode],
    built_students: Dict[Path, nbformat.NotebookNode],
) -> Tuple[bool, Path]:
    """
    Build/update <folder>/.practical_diff.md aggregating all pairs in this folder.
    Returns (report_modified, report_path).
    """
    report_path = folder / ".practical_diff.md"
    sections = []
    header = [
        "# Practical Diff Report",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
    ]
    sections.extend(header)

    pairs = []
    # Collect all *_Answers.ipynb in folder
    for ans_path in sorted(folder.glob("*_Answers.ipynb")):
        stu_path = answers_to_student_path(ans_path)
        if ans_path in normalized_answers and stu_path in built_students:
            ans_nb = normalized_answers[ans_path]
            stu_nb = built_students[stu_path]
        else:
            # Load from disk if not provided
            if not ans_path.exists() or not stu_path.exists():
                # skip incomplete pairs
                continue
            ans_nb = read_nb(ans_path)
            stu_nb = read_nb(stu_path)
        title = ans_path.name.replace("_Answers.ipynb", "")
        md = generate_pair_diff_md(ans_nb, stu_nb, title=title)
        sections.append(md)
        sections.append("")

    new_text = "\n".join(sections).rstrip() + "\n"

    old_text = None
    if report_path.exists():
        old_text = report_path.read_text(encoding="utf-8")

    modified = old_text != new_text
    if modified:
        report_path.write_text(new_text, encoding="utf-8")
    return modified, report_path


# ---------------- Main processing ----------------


def process_answers_notebook(
    ans_path: Path,
) -> Tuple[bool, bool, Path, nbformat.NotebookNode, nbformat.NotebookNode]:
    """
    Returns:
      answers_modified, student_modified, student_path, normalized_answers_nb, student_nb
    """
    nb = read_nb(ans_path)
    nb_before = deepcopy(nb)

    # 1) Normalize answers
    nb = normalize_answers_notebook(nb)
    answers_modified = not notebooks_equal(nb, nb_before)
    if answers_modified:
        write_nb(ans_path, nb)

    # 2) Build student
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

    return answers_modified, student_modified, student_path, nb, student_nb


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
    # Cache normalized/built in this run for report writing
    normalized_map: Dict[Path, nbformat.NotebookNode] = {}
    student_map: Dict[Path, nbformat.NotebookNode] = {}
    folders_touched: Dict[Path, bool] = {}

    results = []
    for ans in targets:
        try:
            a_mod, s_mod, s_path, ans_nb, stu_nb = process_answers_notebook(ans)
            normalized_map[ans] = ans_nb
            student_map[s_path] = stu_nb
            results.append((ans, a_mod, s_mod, s_path))
            folders_touched[ans.parent] = True

            msg = []
            if a_mod:
                msg.append(f"normalized answers (split/tag): {ans}")
            if s_mod:
                msg.append(f"updated student: {s_path}")
            if not msg:
                msg.append(f"no changes needed: {ans.name}")
            print(" | ".join(msg))

            if a_mod:
                exit_code = 1  # force re-stage if answers changed
        except Exception as e:
            print(f"[ERROR] {ans}: {e}", file=sys.stderr)
            exit_code = 2

    # Write folder-level diff reports for any touched folder, or if missing
    for folder in folders_touched.keys():
        # If report missing OR any pair in this folder had changes -> regenerate
        need_update = not (folder / ".practical_diff.md").exists()
        if not need_update:
            # check if any pair in this folder changed in this run
            for ans, a_mod, s_mod, _ in results:
                if ans.parent == folder and (a_mod or s_mod):
                    need_update = True
                    break
        if need_update:
            modified, path = write_folder_diff_report(
                folder, normalized_map, student_map
            )
            if modified:
                # Report is gitignored; just inform, do not fail the commit.
                print(f"updated folder diff report (gitignored): {path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
