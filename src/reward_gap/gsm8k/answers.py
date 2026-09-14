"""Frozen, gold-independent answer extraction for the new GSM8K follow-up."""

import re
from fractions import Fraction

VERSION = "gsm8k_followup_numeric_v1"
NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def numeric(text: str) -> Fraction | None:
    text = text.strip().strip("$").strip().replace(r"\,", "")
    fraction = re.fullmatch(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", text)
    if fraction:
        left, right = numeric(fraction[1]), numeric(fraction[2])
        return left / right if left is not None and right not in (None, 0) else None
    if re.fullmatch(NUMBER, text):
        return Fraction(text.replace(",", ""))
    if text.count("/") == 1:
        left, right = (numeric(part) for part in text.split("/"))
        return left / right if left is not None and right not in (None, 0) else None
    return None


def boxed(text: str) -> tuple[Fraction | None, str]:
    positions = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    if len(positions) != 1:
        return None, "missing_box" if not positions else "multiple_boxes"
    start, depth, end = positions[0], 1, positions[0]
    while end < len(text) and depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        return None, "unfinished_box"
    value = numeric(text[start:end - 1])
    return value, "boxed" if value is not None else "nonnumeric_box"


def extract(text: str) -> tuple[Fraction | None, str]:
    value, reason = boxed(text)
    if value is not None or reason != "missing_box":
        return value, reason
    # Explicit final-answer lines only. Never search for a gold-matching number.
    candidates = re.findall(r"(?im)^\s*(?:####|(?:the\s+)?(?:final\s+)?answer\s*(?:is|:))\s*(.*?)\s*$", text)
    if len(candidates) > 1:
        return None, "ambiguous_final_lines"
    if candidates:
        value = numeric(candidates[0].rstrip("."))
        return value, "explicit_final" if value is not None else "unsupported_final"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    value = numeric(lines[-1]) if lines else None
    return value, "bare_final_line" if value is not None else "no_supported_final"


def evaluate_answer(text: str, gold: str, finish_reason: str) -> dict:
    predicted, reason = extract(text)
    expected = numeric(gold)
    if expected is None:
        raise ValueError("Unparseable GSM8K gold answer")
    formatted = boxed(text)[0] is not None
    match = predicted is not None and predicted == expected
    return {"parser_version": VERSION, "extracted": str(predicted) if predicted is not None else None,
            "parse_reason": reason, "numeric_match": match, "format_compliant": formatted,
            "strict_match": match and formatted, "unresolved": predicted is None,
            "numeric_mismatch": predicted is not None and not match, "length_capped": finish_reason == "length"}


def parse_grade(text: str) -> int | None:
    matches = re.findall(r"(?im)^\s*SCORE:\s*([1-5])\s*$", text)
    terminal = re.search(r"(?i)(?:^|\n)\s*SCORE:\s*([1-5])\s*$", text)
    return int(terminal[1]) if terminal and len(matches) == 1 else None
