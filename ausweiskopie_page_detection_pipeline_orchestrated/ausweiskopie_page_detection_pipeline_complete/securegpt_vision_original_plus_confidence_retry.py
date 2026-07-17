from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from axallm.securegpt.v2.providers import OpenAIProvider
from axallm.securegpt.v2.securegpt import SecureGPT


MODEL_NAME = os.getenv("SECUREGPT_MODEL_NAME", "")
MODEL_VERSION = os.getenv("SECUREGPT_MODEL_VERSION", "")

TEMPERATURE = 0
SEED = 42

MAX_ATTEMPTS = 5
RETRY_DELAYS_SECONDS = [1, 2, 4, 8]


class AusweisPageResponse(BaseModel):
    label: Literal[
        "ausweiskopie",
        "not_ausweiskopie",
        "unclear",
    ]

    evidence_code: Literal[
        "id_card_layout",
        "passport_layout",
        "portrait_and_id_layout",
        "mrz_like_area",
        "multiple_id_sides",
        "no_id_features",
        "unreadable_or_ambiguous",
    ]

    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """
You classify one insurance-document page at a time.

Determine whether the supplied page image contains a full or partial copy
of a personal identity document.

Use exactly one label:

- ausweiskopie:
  The page visibly contains a full or partial copy, scan, photograph, front
  side, back side, or embedded image of an identity document such as an
  identity card, passport, or residence permit.

- not_ausweiskopie:
  The page does not visibly contain an identity-document copy.

- unclear:
  The image is unreadable, too incomplete, ambiguous, or technically
  insufficient for a reliable decision.

Rules:

1. Inspect the image itself. Do not use SST, filenames, folder names, or
   external metadata.
2. A page may contain other document content and still be ausweiskopie when
   an identity-document copy is visibly embedded in it.
3. Do not guess.
4. Do not transcribe names, dates of birth, addresses, document numbers,
   MRZ text, signatures, or any other personal value.
5. Return a confidence value between 0.0 and 1.0.
6. Return only the requested structured response.
""".strip()


USER_PROMPT = """
Inspect this page image and classify it as ausweiskopie,
not_ausweiskopie, or unclear.

Return one coarse evidence code and a confidence value between 0.0 and 1.0.
Do not return personal information.
""".strip()


def validate_configuration() -> None:
    if not MODEL_NAME:
        raise ValueError("SECUREGPT_MODEL_NAME is not set.")

    if not MODEL_VERSION:
        raise ValueError("SECUREGPT_MODEL_VERSION is not set.")


def image_to_data_url(image_path: Path) -> str:
    """Convert a local PNG or JPEG image to a Base64 data URL."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    else:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}"
        )

    encoded_image = base64.b64encode(
        image_path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def create_securegpt_client() -> SecureGPT:
    """Create one approved internal AXA SecureGPT client."""
    validate_configuration()

    return SecureGPT(
        provider=OpenAIProvider(),
        model=MODEL_NAME,
        cache_prompts=True,
        seed=SEED,
        temperature=TEMPERATURE,
        debug=False,
    )


def _answer_value(response: Any) -> Any:
    return response.answer if hasattr(response, "answer") else response


def _parse_answer(answer: Any) -> AusweisPageResponse:
    if isinstance(answer, AusweisPageResponse):
        return answer

    if isinstance(answer, dict):
        return AusweisPageResponse.model_validate(answer)

    if isinstance(answer, str):
        return AusweisPageResponse.model_validate_json(answer)

    raise TypeError(
        "Unsupported SecureGPT answer type: "
        f"{type(answer).__name__}"
    )


def screen_image(
    client: SecureGPT,
    image_path: Path,
) -> dict[str, str | float | int]:
    """
    Send one page image to SecureGPT.

    The wrapper is first called with Pydantic structured output. If the
    installed version does not support response_model together with
    user_image, it falls back to strict JSON and validates locally.
    """
    image_data_url = image_to_data_url(image_path)

    for attempt_index in range(MAX_ATTEMPTS):
        try:
            try:
                response = client.new_chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT,
                    user_image=image_data_url,
                    response_model=AusweisPageResponse,
                )
                parsed = _parse_answer(_answer_value(response))

            except TypeError:
                fallback_prompt = (
                    USER_PROMPT
                    + "\nReturn valid JSON only with exactly these keys: "
                    + json.dumps(
                        {
                            "label": "ausweiskopie",
                            "evidence_code": "id_card_layout",
                            "confidence": 0.95,
                        }
                    )
                )

                response = client.new_chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=fallback_prompt,
                    user_image=image_data_url,
                )
                parsed = _parse_answer(_answer_value(response))

            except ValidationError as error:
                raise ValueError(
                    "SecureGPT returned an invalid structured response."
                ) from error

            return {
                "label": parsed.label,
                "evidence_code": parsed.evidence_code,
                "confidence": parsed.confidence,
                "attempt_count": attempt_index + 1,
                "retry_count": attempt_index,
            }

        except ValueError:
            raise

        except Exception:
            if attempt_index == MAX_ATTEMPTS - 1:
                raise

            time.sleep(RETRY_DELAYS_SECONDS[attempt_index])

    raise RuntimeError("SecureGPT screening failed.")
