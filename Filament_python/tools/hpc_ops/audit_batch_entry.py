#!/usr/bin/env python3
"""Audit a Slurm batch entry before any run/lock/job side effect.

The audit checks Bash syntax and rejects an unqualified ``python`` or
``python3`` command before the first ``conda activate``.  A fixed absolute
interpreter or a shell variable whose name contains ``PYTHON`` remains valid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any


SCHEMA = "filament.hpc_batch_entry_audit.v1"
HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
FUNCTION_START_RE = re.compile(
    r"^\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*(?:\(\s*\))?\s*\{"
)
COMMAND_SUBSTITUTION_RE = re.compile(
    r"(?:\$\(\s*|`\s*)(?P<command>python3?)\b"
)
SEPARATORS = {";", "&", "&&", "||", "|", "(", ")", "{", "}"}
RESERVED_PREFIXES = {
    "!", "if", "then", "elif", "else", "while", "until", "do", "time",
}
WRAPPERS = {"env", "command", "sudo", "exec", "nohup", "nice"}
SUDO_OPTIONS_WITH_VALUE = {"-u", "--user", "-g", "--group", "-h", "--host"}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _validate_fixed_python(value: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or path.name not in {"python", "python3"}:
        raise ValueError("--fixed-python must be an absolute POSIX python/python3 path")
    return str(path)


def _shell_lines(text: str) -> list[tuple[int, str]]:
    """Return shell lines while excluding comments and heredoc bodies."""

    result: list[tuple[int, str]] = []
    heredoc_end: str | None = None
    strip_tabs = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if heredoc_end is not None:
            candidate = raw_line.lstrip("\t") if strip_tabs else raw_line
            if candidate.strip() == heredoc_end:
                heredoc_end = None
                strip_tabs = False
            continue

        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append((line_number, raw_line))

        match = HEREDOC_RE.search(raw_line)
        if match:
            heredoc_end = match.group(2)
            strip_tabs = "<<-" in match.group(0)
    return result


def _logical_lines(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    start_line: int | None = None
    parts: list[str] = []
    for line_number, line in lines:
        if start_line is None:
            start_line = line_number
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            parts.append(stripped[:-1])
            continue
        parts.append(line)
        result.append((start_line, " ".join(parts)))
        start_line = None
        parts = []
    if parts and start_line is not None:
        result.append((start_line, " ".join(parts)))
    return result


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|(){}")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _segments(tokens: list[str]) -> list[list[str]]:
    result: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in SEPARATORS or (token and set(token) <= set(";&|(){}")):
            if current:
                result.append(current)
                current = []
            continue
        current.append(token)
    if current:
        result.append(current)
    return result


def _command_index(segment: list[str]) -> int | None:
    index = 0
    while index < len(segment) and (
        segment[index] in RESERVED_PREFIXES or ASSIGNMENT_RE.match(segment[index])
    ):
        index += 1
    while index < len(segment) and segment[index] in WRAPPERS:
        wrapper = segment[index]
        index += 1
        if wrapper == "command" and index < len(segment) and segment[index] in {"-v", "-V"}:
            return None
        while index < len(segment):
            token = segment[index]
            if ASSIGNMENT_RE.match(token) and wrapper == "env":
                index += 1
                continue
            if token == "--":
                index += 1
                break
            if not token.startswith("-"):
                break
            option = token.split("=", 1)[0]
            index += 1
            if wrapper == "sudo" and option in SUDO_OPTIONS_WITH_VALUE and "=" not in token:
                index += 1
        while index < len(segment) and ASSIGNMENT_RE.match(segment[index]):
            index += 1
    return index if index < len(segment) else None


def _qualified_python(command: str) -> bool:
    unquoted = command.strip("'\"")
    return bool(
        re.fullmatch(r"/[^\s]+/python3?", unquoted)
        or re.fullmatch(
            r"\$\{?[A-Za-z_][A-Za-z0-9_]*PYTHON[A-Za-z0-9_]*\}?",
            unquoted,
            re.IGNORECASE,
        )
    )


def _bash_syntax_command(batch_path: Path, bash: str) -> list[str]:
    if os.name != "nt":
        return [bash, "-n", str(batch_path)]

    native_bash = shutil.which(bash)
    system_bash = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "bash.exe"
    if native_bash and Path(native_bash).resolve() != system_bash.resolve():
        return [native_bash, "-n", str(batch_path)]

    if not shutil.which("wsl.exe"):
        raise OSError("no native POSIX bash or WSL is available for bash -n")
    windows_path = str(batch_path).replace("\\", "/")
    converted = subprocess.run(
        ["wsl.exe", "--", "wslpath", "-a", "-u", "--", windows_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if converted.returncode != 0 or not converted.stdout.strip():
        raise OSError("failed to translate the batch path for WSL bash")
    return ["wsl.exe", "--", bash, "-n", converted.stdout.strip()]


def audit_batch_entry(batch_path: Path, fixed_python: str, bash: str = "bash") -> tuple[dict[str, Any], int]:
    fixed_python = _validate_fixed_python(fixed_python)
    if batch_path.is_symlink():
        payload = {
            "schema": SCHEMA,
            "status": "invalid_input",
            "batch_path": str(batch_path.absolute()),
            "fixed_python": fixed_python,
            "failure": "batch path must be a regular non-symlink file",
        }
        return payload, 2
    batch_path = batch_path.resolve()
    if not batch_path.is_file():
        payload = {
            "schema": SCHEMA,
            "status": "invalid_input",
            "batch_path": str(batch_path),
            "fixed_python": fixed_python,
            "failure": "batch path must be a regular non-symlink file",
        }
        return payload, 2

    text = batch_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    syntax_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".sh", delete=False) as handle:
            handle.write(text.encode("utf-8"))
            syntax_path = Path(handle.name)
        syntax = subprocess.run(
            _bash_syntax_command(syntax_path, bash),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        if syntax_path is not None:
            syntax_path.unlink(missing_ok=True)
    if syntax.returncode != 0:
        payload = {
            "schema": SCHEMA,
            "status": "bash_syntax_failed",
            "batch_path": str(batch_path),
            "fixed_python": fixed_python,
            "bash_syntax_exit_code": syntax.returncode,
            "failure": (syntax.stderr.strip().splitlines() or ["bash -n failed"])[-1],
        }
        return payload, 2

    lines = _logical_lines(_shell_lines(text))
    activation_line: int | None = None
    bare: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    preactivation_count = 0
    function_depth = 0
    for line_number, line in lines:
        if activation_line is not None:
            break
        preactivation_count += 1
        starts_function = bool(FUNCTION_START_RE.match(line))
        inside_function = function_depth > 0 or starts_function
        try:
            segments = _segments(_tokens(line))
        except ValueError as exc:
            payload = {
                "schema": SCHEMA,
                "status": "shell_tokenization_failed",
                "batch_path": str(batch_path),
                "fixed_python": fixed_python,
                "bash_syntax_exit_code": 0,
                "failure": f"line {line_number}: {exc}",
            }
            return payload, 2

        for segment in segments:
            index = _command_index(segment)
            if index is None:
                continue
            command = segment[index]
            if not inside_function and command == "conda" and segment[index + 1:index + 2] == ["activate"]:
                activation_line = line_number
                break
            if command in {"python", "python3"}:
                bare.append({"line": line_number, "command": command})
            elif _qualified_python(command):
                qualified.append({"line": line_number, "command": command.strip("'\"")})

        for match in COMMAND_SUBSTITUTION_RE.finditer(line):
            bare.append({"line": line_number, "command": match.group("command")})

        if starts_function:
            function_depth += 1
        if function_depth and line.strip().startswith("}"):
            function_depth -= 1

    status = "passed" if not bare else "bare_python_before_activation"
    payload = {
        "schema": SCHEMA,
        "status": status,
        "batch_path": str(batch_path),
        "fixed_python": fixed_python,
        "bash_syntax_exit_code": 0,
        "activation_line": activation_line,
        "preactivation_shell_line_count": preactivation_count,
        "qualified_python_commands": qualified,
        "bare_python_commands": bare,
    }
    return payload, 0 if not bare else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--fixed-python", required=True)
    parser.add_argument("--bash", default="bash")
    args = parser.parse_args(argv)
    try:
        payload, return_code = audit_batch_entry(args.batch, args.fixed_python, args.bash)
    except (OSError, UnicodeError, ValueError) as exc:
        payload = {
            "schema": SCHEMA,
            "status": "audit_error",
            "batch_path": str(args.batch.resolve()),
            "fixed_python": args.fixed_python,
            "failure": str(exc),
        }
        return_code = 2
    _emit(payload)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
