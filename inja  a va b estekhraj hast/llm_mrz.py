from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from Life.Extraction import config

MRZ_SYSTEM_PROMPT = """You transcribe the machine readable zone (MRZ) of an identity document.
Return only the MRZ lines, one per line, using only the characters A-Z, 0-9 and <.
Transcribe exactly what is printed. Do not correct, complete or guess any character.
If a character is covered, blurred or unreadable, return < in its place."""

MRZ_USER_PROMPT = """Transcribe every MRZ line visible in this image. Return only the lines."""


# Import the SecureGPT wrapper that lives in the ausweiskopie_page_detection project.
def securegpt_pieces() -> tuple[Any, Any]:
    if str(config.SECUREGPT_WRAPPER_DIR) not in sys.path:
        sys.path.insert(0, str(config.SECUREGPT_WRAPPER_DIR))
    from securegpt_vision import create_securegpt_client, normalize_page_image

    return create_securegpt_client, normalize_page_image


# Keep only MRZ characters per line, without dropping any, so offsets stay aligned.
def clean_answer(text: str) -> str:
    lines = []
    for raw in str(text).splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("```"):
            lines.append(stripped)
    return "\n".join(lines)


# Ask the vision model to read a band we know we cropped correctly. Used only when OCR
# failed on a good crop, typically a finger or glare over part of a line. The result is
# still put through the ICAO check digits, so a wrong transcription cannot be accepted.
def transcribe_mrz(crop_path: Path) -> str:
    create_client, normalize_image = securegpt_pieces()
    client = create_client()
    response = client.new_chat(system_prompt=MRZ_SYSTEM_PROMPT, user_prompt=MRZ_USER_PROMPT, user_image=normalize_image(crop_path))
    answer = response.answer if hasattr(response, "answer") else response
    return clean_answer(answer)
