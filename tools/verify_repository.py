"""Check repository structure, Python syntax, file sizes, and common leaks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tokenize
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GITHUB_LIMIT = 100 * 1024 * 1024
GITHUB_WARNING = 50 * 1024 * 1024
REQUIRED = {
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "CITATION.cff",
    "CITATION.bib",
    "requirements.txt",
    "environment.yml",
    ".gitignore",
    ".gitattributes",
    "docs/DATASETS.md",
    "docs/CODE_MAP.md",
    "docs/GITHUB_SETUP.md",
    "docs/EXCLUSIONS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/RESULTS.md",
    "docs/local-files-manifest.csv",
    "research_code/README.md",
    "data/README.md",
    "artifacts/README.md",
    "results/README.md",
}
REQUIRED_SAMPLE_SHAPES = {
    "data/samples/theory/LHS_sample.txt": (1200, 5),
    "data/samples/theory/sobol_sample.txt": (1024, 7),
    "data/samples/forward_design/LHS_train_sample.txt": (1200, 5),
    "data/samples/forward_design/LHS_test_sample.txt": (150, 5),
    "data/samples/forward_design/origin_data/train_comsol_inputdata.txt": (1200, 6),
    "data/samples/forward_design/origin_data/test_comsol_inputdata.txt": (150, 6),
    "data/samples/forward_design/3.3.3val.txt": (2855, 7),
}
FORBIDDEN_PARTS = {".idea", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".pth", ".pkl"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".ipynb",
    ".yml",
    ".yaml",
    ".cff",
    ".bib",
}
ABSOLUTE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:kaggle|home|Users)/)")
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SECRET_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret|password|passwd|access[_-]?token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; repository errors are always fatal and warnings remain nonfatal.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_files() -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            path
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )

    repository_paths = result.stdout.decode("utf-8").split("\0")
    return sorted(
        ROOT / name
        for name in repository_paths
        if name and (ROOT / name).is_file()
    )


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return None
    if path.name not in {".gitignore", ".gitattributes", ".python-version", "VERSION"}:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return None
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return None


def whitespace_shape(path: Path) -> tuple[int, int]:
    rows = 0
    columns: int | None = None
    with path.open("r", encoding="utf-8-sig") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            current = len(line.split())
            if columns is None:
                columns = current
            elif current != columns:
                raise ValueError(f"row {rows + 1} has {current} columns, expected {columns}")
            rows += 1
    return rows, columns or 0


def main() -> int:
    parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    files = iter_files()

    missing = sorted(item for item in REQUIRED if not (ROOT / item).is_file())
    errors.extend(f"Missing required file: {item}" for item in missing)

    for path in files:
        rel = relative(path)
        if FORBIDDEN_PARTS.intersection(path.parts):
            errors.append(f"Generated/IDE file is included: {rel}")
        size = path.stat().st_size
        if size >= GITHUB_LIMIT:
            errors.append(f"File exceeds GitHub's 100 MiB limit: {rel} ({size} bytes)")
        elif size >= GITHUB_WARNING:
            warnings.append(f"Large file will trigger a GitHub warning: {rel} ({size} bytes)")

    python_files = [path for path in files if path.suffix.lower() == ".py"]
    for path in python_files:
        try:
            with tokenize.open(path) as stream:
                source = stream.read()
            compile(source, str(path), "exec")
        except Exception as exc:  # noqa: BLE001 - this is a repository audit
            errors.append(f"Python syntax/read failure in {relative(path)}: {exc}")

    notebooks = [path for path in files if path.suffix.lower() == ".ipynb"]
    for path in notebooks:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid notebook JSON in {relative(path)}: {exc}")

    for filename, expected_shape in REQUIRED_SAMPLE_SHAPES.items():
        path = ROOT / filename
        if not path.is_file():
            errors.append(f"Required sample data is missing: {filename}")
            continue
        try:
            actual_shape = whitespace_shape(path)
        except (UnicodeDecodeError, ValueError) as exc:
            errors.append(f"Invalid sample data in {filename}: {exc}")
            continue
        if actual_shape != expected_shape:
            errors.append(
                f"Unexpected sample shape for {filename}: {actual_shape}, expected {expected_shape}"
            )

    for path in files:
        if path.resolve() == Path(__file__).resolve():
            continue
        content = read_text(path)
        if content is None:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"Possible {label} in {relative(path)}")

    absolute_path_hits: list[str] = []
    source_root = ROOT / "research_code"
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".ipynb"}:
            continue
        content = read_text(path)
        if content is None:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if ABSOLUTE_PATH.search(line):
                absolute_path_hits.append(f"{relative(path)}:{line_number}")
    if absolute_path_hits:
        preview = ", ".join(absolute_path_hits[:8])
        remainder = len(absolute_path_hits) - 8
        suffix = f", plus {remainder} more" if remainder > 0 else ""
        warnings.append(
            f"Historical absolute paths found at {len(absolute_path_hits)} locations: {preview}{suffix}"
        )

    hashes: dict[str, list[str]] = defaultdict(list)
    for path in python_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(relative(path))
    for group in hashes.values():
        if len(group) > 1:
            warnings.append(f"Duplicate Python sources: {', '.join(group)}")

    artifact_manifest = ROOT / "artifacts" / "ARTIFACTS.sha256"
    if not artifact_manifest.is_file():
        errors.append("Missing artifacts/ARTIFACTS.sha256")
    else:
        for line_number, line in enumerate(
            artifact_manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                expected, artifact_name = line.split(maxsplit=1)
            except ValueError:
                errors.append(f"Malformed artifact hash at line {line_number}")
                continue
            artifact_path = ROOT / "artifacts" / artifact_name.lstrip("*")
            if not artifact_path.is_file():
                errors.append(f"Artifact listed but missing: {relative(artifact_path)}")
                continue
            actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if actual.lower() != expected.lower():
                errors.append(f"Artifact checksum mismatch: {relative(artifact_path)}")

    for path in [item for item in files if item.suffix.lower() == ".md"]:
        content = read_text(path) or ""
        for target in MARKDOWN_LINK.findall(content):
            target = target.strip().split("#", maxsplit=1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            link_path = (path.parent / target).resolve()
            try:
                link_path.relative_to(ROOT)
            except ValueError:
                errors.append(f"Relative link escapes repository in {relative(path)}: {target}")
                continue
            if not link_path.exists():
                errors.append(f"Broken relative link in {relative(path)}: {target}")

    total_bytes = sum(path.stat().st_size for path in files)
    print(
        f"Checked {len(files)} files ({total_bytes / (1024 * 1024):.2f} MiB), "
        f"{len(python_files)} Python files, and {len(notebooks)} notebook(s)."
    )
    for item in warnings:
        print(f"WARNING: {item}")
    for item in errors:
        print(f"ERROR: {item}")
    print(f"Result: {len(errors)} error(s), {len(warnings)} warning(s).")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
