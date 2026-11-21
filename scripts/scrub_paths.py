"""Pre-push scrubber: remove personal absolute paths from an EXPORT COPY of the repository.

Notebook outputs and run logs contain absolute paths of the training machine (project folder,
virtual environment, user home, Hugging Face cache). Run this on a copy of the files that will be
pushed (never on the working folder), then push the copy.

    python scripts/scrub_paths.py <export_dir>            # rewrite files in place
    python scripts/scrub_paths.py <export_dir> --check    # only report; exit 1 if anything is left

Replacements
    <venv>/site-packages, python   virtual-environment paths
    <repo>                         the 02_code folder (the repository root)
    <project>                      the parent project folder
    <home>                         the user's home folder (any <home> or <home>
Binary files (.png, .jpg, .pt, .npz, .zip) are skipped.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path, PurePath, PureWindowsPath

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pt", ".pth", ".npz", ".npy", ".zip", ".pdf"}
SKIP_DIRS = {".git", "data", "checkpoints", ".venv", "__pycache__"}


def _win(path: str) -> str:
    """Regex for a Windows path that matches single or JSON-escaped double backslashes, any case."""
    parts = [re.escape(p) for p in re.split(r"[\\/]+", path.strip("\\/")) if p]
    drive = parts[0].replace(r"\:", ":")
    if re.fullmatch(r"[A-Za-z]:", parts[0].replace("\\", "")):
        drive = f"[{parts[0][0].lower()}{parts[0][0].upper()}](?::|%3A)"
    return drive + r"(?:\\{1,2}|/)" + r"(?:\\{1,2}|/)".join(parts[1:])


def as_pure(path: str) -> PurePath:
    """Treat 'E:\\...' as a Windows path even when the script runs on Linux/macOS."""
    return PureWindowsPath(path) if re.match(r"^[A-Za-z]:", str(path)) else PurePath(path)


def build_patterns(repo_root: PurePath) -> list[tuple[re.Pattern, str]]:
    project = repo_root.parent
    pats = []
    # most specific first
    for root, tag in ((project, "<project>"),):
        pats.append((re.compile(_win(str(root)) + r"(?:\\{1,2}|/)\.venv(?:\\{1,2}|/)[Ll]ib(?:\\{1,2}|/)site-packages"),
                     "<venv>/site-packages"))
        pats.append((re.compile(_win(str(root)) + r"(?:\\{1,2}|/)\.venv(?:\\{1,2}|/)Scripts(?:\\{1,2}|/)python\.exe"),
                     "python"))
    pats.append((re.compile(_win(str(repo_root))), "<repo>"))
    pats.append((re.compile(_win(str(project))), "<project>"))
    # generic home folders (Windows and POSIX)
    pats.append((re.compile(r"[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)[^\\/\s\"']+"), "<home>"))
    pats.append((re.compile(r"<home>/\s\"']+"), "<home>"))
    pats.append((re.compile(r"<home>/\s\"']+"), "<home>"))
    return pats


def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() not in SKIP_SUFFIXES and not (set(p.relative_to(root).parts) & SKIP_DIRS):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--repo-root", default=None,
                    help="absolute path of the original 02_code folder (default: the folder this script lives in)")
    a = ap.parse_args()
    root = Path(a.export_dir).resolve()
    repo_root = as_pure(a.repo_root) if a.repo_root else Path(__file__).resolve().parents[1]
    if isinstance(repo_root, Path) and root == repo_root.resolve() and not a.check:
        sys.exit("Refusing to rewrite the working folder; run on an export copy.")
    pats = build_patterns(repo_root)
    changed, left = 0, {}
    for f in iter_files(root):
        try:
            s = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = s
        for pat, rep in pats:
            new = pat.sub(rep, new)
        if new != s:
            if a.check:
                left[str(f.relative_to(root))] = sum(len(p.findall(s)) for p, _ in pats)
            else:
                f.write_text(new, encoding="utf-8")
                changed += 1
    if a.check:
        for k, v in left.items():
            print(f"  {v:4d} path(s) in {k}")
        print(f"[scrub check] {len(left)} file(s) still contain personal paths")
        sys.exit(1 if left else 0)
    print(f"[scrub] rewrote {changed} file(s) in {root}")


if __name__ == "__main__":
    main()
