from __future__ import annotations

from itertools import combinations
from typing import Any

from Life.Extraction import config

WEIGHTS = [7, 3, 1]
MRZ_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
FORMATS = {"TD1": (3, 30), "TD3": (2, 44), "TD2": (2, 36)}

# Positions holding a check digit; these are never touched by OCR-confusion repair.
CHECK_POSITIONS = {
    "TD1": {(0, 14), (1, 6), (1, 14), (1, 29)},
    "TD2": {(1, 9), (1, 19), (1, 27), (1, 35)},
    "TD3": {(1, 9), (1, 19), (1, 27), (1, 42), (1, 43)},
}

TRANSLITERATION = {"Ä": "AE", "Ö": "OE", "Ü": "UE", "ß": "SS", "Æ": "AE", "Ø": "OE", "Å": "AA", "Ñ": "N", "Ç": "C", "É": "E", "È": "E", "Ê": "E", "Á": "A", "À": "A", "Ó": "O", "Ú": "U", "Í": "I"}


def char_value(char: str) -> int:
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    raise ValueError(f"Invalid MRZ character: {char}")


def check_digit(text: str) -> str:
    total = 0
    for index, char in enumerate(text):
        total += char_value(char) * WEIGHTS[index % 3]
    return str(total % 10)


# ICAO check digit. A field that is entirely filler may legitimately carry '<' instead of '0'.
def check_field(name: str, value: str, digit: str) -> dict[str, Any]:
    expected = check_digit(value)
    all_filler = bool(value) and set(value) == {"<"}
    valid = digit == expected or (all_filler and digit in {"<", "0"})
    return {"field": name, "input": value, "expected": expected, "actual": digit, "valid": valid}


# Map one OCR character onto the MRZ alphabet. Unknown glyphs become filler so positions stay aligned.
def normalize_char(char: str) -> str:
    upper = char.upper()
    if upper in MRZ_ALPHABET:
        return upper
    return config.MRZ_CHAR_FIXES.get(char, config.MRZ_CHAR_FIXES.get(upper, "<"))


# Normalize an OCR line without deleting anything, so character offsets survive.
def normalize_line(text: str) -> str:
    return "".join(normalize_char(char) for char in text.strip())


# Split OCR text into normalized lines, dropping lines that are pure filler.
def normalize_mrz_lines(text: str) -> list[str]:
    lines = [normalize_line(raw) for raw in text.splitlines()]
    return [line for line in lines if line.strip("<")]


# Fit a line to a canonical width by trimming overflow or padding trailing filler.
def fit_line(line: str, width: int) -> tuple[str, int]:
    if len(line) >= width:
        return line[:width], len(line) - width
    return line.ljust(width, "<"), len(line) - width


# Zones the format defines as optional data, where a mostly-filler reading is filler.
# Bounding the repair to these ranges keeps it from ever rewriting a real data field.
FILLER_ZONES = {
    "TD1": [(0, 15, 30), (1, 18, 29)],
    "TD2": [(1, 28, 35)],
    "TD3": [(1, 28, 42)],
}


# Rewrite an optional-data zone to pure filler when it already reads as mostly filler.
def repair_filler_zones(lines: list[str], fmt: str) -> list[str]:
    repaired = [list(line) for line in lines]
    for line_index, start, end in FILLER_ZONES[fmt]:
        if line_index >= len(repaired):
            continue
        zone = repaired[line_index][start:end]
        if len(zone) >= 4 and zone.count("<") / len(zone) >= 0.6:
            repaired[line_index][start:end] = ["<"] * len(zone)
    return ["".join(line) for line in repaired]


# One candidate per MRZ format and per consecutive window of lines. Sliding rather than
# taking the last N matters: OCR often picks up a stray line above or below the band.
def candidate_variants(lines: list[str]) -> list[dict[str, Any]]:
    variants = []
    for name, (count, width) in FORMATS.items():
        for start in range(len(lines) - count + 1):
            variants.extend(build_variants(name, width, lines[start : start + count]))
    return sorted(variants, key=lambda item: item["width_error"])


# Raw and filler-repaired variants for one window, or nothing if the widths are too far off.
def build_variants(name: str, width: int, window: list[str]) -> list[dict[str, Any]]:
    variants = []
    for line in [window]:
        tail = line
        if any(abs(len(item) - width) > config.MRZ_WIDTH_TOLERANCE for item in tail):
            continue
        width_error = sum(abs(len(item) - width) for item in tail)
        fitted = [fit_line(item, width) for item in tail]
        lines_out = [item[0] for item in fitted]
        deltas = [item[1] for item in fitted]
        variants.append({"format": name, "lines": lines_out, "length_deltas": deltas, "width_error": width_error})
        repaired = repair_filler_zones(lines_out, name)
        if repaired != lines_out:
            variants.append({"format": name, "lines": repaired, "length_deltas": deltas, "width_error": width_error, "filler_repaired": True})
    return variants


def decode_mrz_text(value: str) -> str:
    return " ".join(value.replace("<", " ").split()) or ""


def parse_name(name_field: str) -> tuple[str | None, str | None]:
    parts = name_field.split("<<", 1)
    surname = decode_mrz_text(parts[0]) or None
    given_names = decode_mrz_text(parts[1]) if len(parts) > 1 else None
    return surname, given_names or None


# Replace German umlauts and similar letters with their ICAO MRZ spelling for name comparison.
def transliterate(value: str) -> str:
    return "".join(TRANSLITERATION.get(char, char) for char in value.upper())


# GWG Ausweistyp from the MRZ document code: ID -> P, P -> R, anything else -> S.
def map_ausweistyp(document_code: str) -> str:
    code = (document_code or "").upper().replace("<", "").strip()
    if code.startswith("ID"):
        return "P"
    if code.startswith("P"):
        return "R"
    return "S"


# Fields whose shape is fixed by ICAO. A checksum alone cannot catch a letter inside a date,
# because repair may swap a digit for a letter and the arithmetic still agrees.
def structural_checks(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for field in ("date_of_birth", "expiry_date"):
        value = str(parsed.get(field) or "")
        checks.append({"field": f"{field}_format", "input": value, "expected": "6 digits", "actual": value, "valid": len(value) == 6 and value.isdigit()})
    sex = str(parsed.get("sex") or "")
    checks.append({"field": "sex_format", "input": sex, "expected": "M, F or empty", "actual": sex, "valid": sex in {"M", "F", ""}})
    state = str(parsed.get("issuing_state") or "")
    checks.append({"field": "issuing_state_format", "input": state, "expected": "1-3 letters", "actual": state, "valid": 1 <= len(state) <= 3 and state.isalpha()})
    return checks


def with_aliases(parsed: dict[str, Any]) -> dict[str, Any]:
    parsed["ausweisnummer"] = parsed["document_number"]
    parsed["nationalitaet"] = parsed["nationality"]
    parsed["gueltigkeitsdatum"] = parsed["expiry_date"]
    parsed["check_digit_results"] = {item["field"]: item for item in parsed["checks"]}
    parsed["structural_checks"] = structural_checks(parsed)
    parsed["checks_valid"] = all(item["valid"] for item in parsed["checks"] + parsed["structural_checks"])
    return parsed


def parse_td3(lines: list[str]) -> dict[str, Any]:
    line1, line2 = lines
    surname, given_names = parse_name(line1[5:44])
    checks = [
        check_field("document_number", line2[0:9], line2[9]),
        check_field("date_of_birth", line2[13:19], line2[19]),
        check_field("expiry_date", line2[21:27], line2[27]),
        check_field("optional_data", line2[28:42], line2[42]),
        check_field("composite", line2[0:10] + line2[13:20] + line2[21:43], line2[43]),
    ]
    return with_aliases({
        "format": "TD3",
        "document_code": line1[0:2].replace("<", ""),
        "issuing_state": line1[2:5].replace("<", ""),
        "surname": surname,
        "given_names": given_names,
        "document_number": line2[0:9].replace("<", ""),
        "nationality": line2[10:13].replace("<", ""),
        "date_of_birth": line2[13:19],
        "sex": line2[20].replace("<", ""),
        "expiry_date": line2[21:27],
        "optional_data": line2[28:42].replace("<", ""),
        "checks": checks,
    })


def parse_td2(lines: list[str]) -> dict[str, Any]:
    line1, line2 = lines
    surname, given_names = parse_name(line1[5:36])
    checks = [
        check_field("document_number", line2[0:9], line2[9]),
        check_field("date_of_birth", line2[13:19], line2[19]),
        check_field("expiry_date", line2[21:27], line2[27]),
        check_field("composite", line2[0:10] + line2[13:20] + line2[21:35], line2[35]),
    ]
    return with_aliases({
        "format": "TD2",
        "document_code": line1[0:2].replace("<", ""),
        "issuing_state": line1[2:5].replace("<", ""),
        "surname": surname,
        "given_names": given_names,
        "document_number": line2[0:9].replace("<", ""),
        "nationality": line2[10:13].replace("<", ""),
        "date_of_birth": line2[13:19],
        "sex": line2[20].replace("<", ""),
        "expiry_date": line2[21:27],
        "optional_data": line2[28:35].replace("<", ""),
        "checks": checks,
    })


def parse_td1(lines: list[str]) -> dict[str, Any]:
    line1, line2, line3 = lines
    surname, given_names = parse_name(line3)
    checks = [
        check_field("document_number", line1[5:14], line1[14]),
        check_field("date_of_birth", line2[0:6], line2[6]),
        check_field("expiry_date", line2[8:14], line2[14]),
        check_field("composite", line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29], line2[29]),
    ]
    return with_aliases({
        "format": "TD1",
        "document_code": line1[0:2].replace("<", ""),
        "issuing_state": line1[2:5].replace("<", ""),
        "surname": surname,
        "given_names": given_names,
        "document_number": line1[5:14].replace("<", ""),
        "nationality": line2[15:18].replace("<", ""),
        "date_of_birth": line2[0:6],
        "sex": line2[7].replace("<", ""),
        "expiry_date": line2[8:14],
        "optional_data": (line1[15:30] + line2[18:29]).replace("<", ""),
        "checks": checks,
    })


PARSERS = {"TD1": parse_td1, "TD2": parse_td2, "TD3": parse_td3}


# Parse one fitted line set into a record carrying its own provenance.
def parse_variant(variant: dict[str, Any], lines: list[str]) -> dict[str, Any]:
    parsed = PARSERS[variant["format"]](lines)
    parsed["mrz_lines"] = lines
    parsed["mrz_line_deltas"] = variant["length_deltas"]
    parsed["mrz_padding_applied"] = any(delta < 0 for delta in variant["length_deltas"])
    parsed["mrz_filler_repaired"] = bool(variant.get("filler_repaired"))
    return parsed


# A position may be repaired anywhere; on a check digit only a letter-to-digit swap is allowed.
def repairable(variant: dict[str, Any], line_index: int, char_index: int, char: str) -> bool:
    if char not in config.OCR_CONFUSIONS:
        return False
    if (line_index, char_index) in CHECK_POSITIONS[variant["format"]]:
        return config.OCR_CONFUSIONS[char].isdigit()
    return True


# Every one- and two-character OCR-confusion swap the variant permits.
def repair_candidates(variant: dict[str, Any]) -> list[tuple[list[str], list[tuple[int, int]]]]:
    positions = [
        (line_index, char_index)
        for line_index, line in enumerate(variant["lines"])
        for char_index, char in enumerate(line)
        if repairable(variant, line_index, char_index, char)
    ]
    repairs = []
    for size in (1, 2):
        for selected in combinations(positions, size):
            lines = [list(line) for line in variant["lines"]]
            for line_index, char_index in selected:
                lines[line_index][char_index] = config.OCR_CONFUSIONS[lines[line_index][char_index]]
            repairs.append((["".join(line) for line in lines], list(selected)))
    return repairs


# Accept a variant only if every check digit passes, with or without repair.
def resolve_variant(variant: dict[str, Any]) -> dict[str, Any] | None:
    parsed = parse_variant(variant, variant["lines"])
    if parsed["checks_valid"]:
        parsed["repair_applied"] = False
        return parsed
    for lines, positions in repair_candidates(variant):
        candidate = parse_variant(variant, lines)
        if candidate["checks_valid"]:
            candidate["repair_applied"] = True
            candidate["repair_positions"] = positions
            return candidate
    return None


# Try every plausible MRZ format; return the first that is fully checksum-valid.
def parse_with_repair(text: str) -> dict[str, Any] | None:
    lines = normalize_mrz_lines(text)
    if not lines:
        return None
    for variant in candidate_variants(lines):
        parsed = resolve_variant(variant)
        if parsed:
            return parsed
    return None