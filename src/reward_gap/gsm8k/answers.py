"""Frozen, gold-independent answer extraction for the new GSM8K follow-up."""

import re
from fractions import Fraction

VERSION = "gsm8k_followup_numeric_v1"
GRADE_PARSER_VERSION = "gsm8k_score_boundary_complete_v2"
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


def parse_grade_output(text: str, *, complete: bool = True) -> tuple[int | None, str]:
    """Accept one explicit score on the first or last nonempty line.

    A leading score is used only after generation completes: a truncated
    explanation could still revise it. Never infer a grade from other numbers.
    """
    if not complete:
        return None, "incomplete_output"
    # Count even malformed/inline score fields, so a second conflicting field
    # cannot disappear merely because its value is outside the 1..5 range.
    if len(re.findall(r"(?i)\bSCORE\s*:", text)) != 1:
        return None, "missing_or_multiple_score_fields"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "missing_or_multiple_score_fields"
    for line, kind in ((lines[-1], "terminal_score"), (lines[0], "leading_score")):
        match = re.fullmatch(r"SCORE:\s*([1-5])", line, flags=re.IGNORECASE)
        if match:
            return int(match[1]), kind
    return None, "invalid_score_format"


def parse_grade(text: str) -> int | None:
    """Parse complete grading text; generation callers must check completion."""
    return parse_grade_output(text)[0]
