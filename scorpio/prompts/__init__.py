"""Prompt texts, one file each."""

from pathlib import Path


def load(name: str) -> str:
    # The trailing newline belongs to the file, not to the prompt.
    return (Path(__file__).resolve().parent / f"{name}.txt").read_text().rstrip("\n")
