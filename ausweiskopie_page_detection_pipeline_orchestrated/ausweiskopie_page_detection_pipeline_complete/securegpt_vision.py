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


# Copy the exact complete model ID from SecureGPT Model Hub.
MODEL = "PUT-EXACT-MODEL-ID-HERE"

TEMPERATURE = 0
SEED = 42

# Five total attempts:
# attempt 1, then retries after 1, 2, 4 and 8 seconds.
MAX_ATTEMPTS = 5
RETRY_DELAYS_SECONDS = [1, 2, 4, 8]


class AusweisPageResponse(BaseModel):
    """Structured output returned by SecureGPT."""

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

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the selected label. "
            "Use a value between 0.0 and 1.0."
        ),
    )


class SecureGPTScreeningError(RuntimeError):
    """SecureGPT failure including the number of attempted calls."""

    def __init__(
        self,
        message: str,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts


SYSTEM_PROMPT = """
Du klassifizierst jeweils genau eine Seite eines Versicherungsdokuments.

Prüfe ausschließlich das bereitgestellte Seitenbild und entscheide, ob die
Seite vollständig oder teilweise eine Kopie eines amtlichen persönlichen
Identitätsdokuments enthält.

Verwende genau eines der folgenden Labels:

- ausweiskopie:
  Auf der Seite ist vollständig oder teilweise eine Kopie, ein Scan, ein Foto,
  die Vorderseite, die Rückseite oder eine eingebettete Abbildung eines
  amtlichen Identitätsdokuments sichtbar.

  Dazu gehören insbesondere:
  - Personalausweis
  - Reisepass
  - Aufenthaltstitel

- not_ausweiskopie:
  Auf der Seite ist keine Kopie eines amtlichen persönlichen
  Identitätsdokuments sichtbar.

- unclear:
  Das Bild ist nicht ausreichend lesbar, zu unvollständig, mehrdeutig oder
  technisch ungeeignet, sodass keine zuverlässige Entscheidung möglich ist.

Beachte folgende Regeln:

1. Beurteile ausschließlich den sichtbaren Inhalt des Seitenbildes.
   Verwende keine Dateinamen, Ordnernamen, SST-Werte oder sonstige Metadaten.

2. Eine Seite kann zusätzlich andere Dokumentinhalte enthalten und trotzdem
   als ausweiskopie klassifiziert werden, wenn darauf eine vollständige oder
   teilweise Kopie eines Identitätsdokuments sichtbar ist.

3. Auch eine teilweise sichtbare Vorder- oder Rückseite eines
   Identitätsdokuments gilt als ausweiskopie, sofern sie eindeutig als Teil
   eines Identitätsdokuments erkennbar ist.

4. Krankenversicherungskarten, Bankkarten, Kundenkarten,
   Mitarbeiterausweise, Führerscheine und reine Passfotos gelten nicht als
   Ausweiskopie.

5. Wenn nicht zuverlässig entschieden werden kann, verwende unclear.
   Rate nicht.

6. Gib niemals personenbezogene Inhalte wieder. Transkribiere insbesondere
   keine Namen, Geburtsdaten, Anschriften, Dokumentennummern,
   Unterschriften oder MRZ-Inhalte.

7. Gib zusätzlich einen confidence-Wert zwischen 0.0 und 1.0 zurück:

   - 0.90 bis 1.00:
     Das Ergebnis ist visuell eindeutig.

   - 0.60 bis 0.89:
     Das Ergebnis ist wahrscheinlich, aber nicht vollständig eindeutig.

   - unter 0.60:
     Das Ergebnis ist stark unsicher und sollte manuell geprüft werden.

8. Verwende bei stark unsicheren Bildern vorzugsweise das Label unclear.

9. Gib ausschließlich die angeforderte strukturierte Antwort zurück.
""".strip()


USER_PROMPT = """
Prüfe dieses Seitenbild.

Gib zurück:

- label
- evidence_code
- confidence

Gib keine personenbezogenen Informationen zurück.
""".strip()


def read_image_base4(image_path) -> str;
    """Return one image file as base64/encoded UTF/8 text."""
    with open (image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def create_securegpt_client() -> SecureGPT:
    """Create one SecureGPT client for the complete screening run."""
    if not MODEL or MODEL == "PUT-EXACT-MODEL-ID-HERE":
        raise ValueError(
            "Set MODEL_ID to the exact model identifier from Model Hub."
        )

    return SecureGPT(
        model=MODEL,
        temperature=TEMPERATURE,
        seed=SEED,
        cache_prompts=False,
    )


def is_retryable_securegpt_error(
    error: Exception,
) -> bool:
    """Return True for temporary infrastructure errors."""
    error_text = str(error).lower()

    retryable_markers = [
        "routing failed",
        "backend not available",
        "backend unavailable",
        "esg120",
        "error code: 500",
        "error code: 502",
        "error code: 503",
        "error code: 504",
        "status code: 500",
        "status code: 502",
        "status code: 503",
        "status code: 504",
        "timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "too many requests",
        "rate limit",
        "error code: 429",
        "status code: 429",
    ]

    return any(
        marker in error_text
        for marker in retryable_markers
    )


def screen_image(
    client: SecureGPT,
    image_path: Path,
) -> dict[str, str | float | int]:
    """
    Classify one rendered page.

    Temporary errors are retried with delays of 1, 2, 4 and 8 seconds.
    """
    if not image_path.exists():
        raise FileNotFoundError(
            f"Rendered image not found: {image_path}"
        )

    image_base64 = read_image_base64(
        str(image_path)
    )

    last_error: Exception | None = None

    for attempt_index in range(MAX_ATTEMPTS):
        attempt_number = attempt_index + 1

        try:
            response = client.new_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                user_image=image_base64,
                image_detail="high",
                response_model=AusweisPageResponse,
            )

            parsed = response.answer

            if not isinstance(
                parsed,
                AusweisPageResponse,
            ):
                parsed = AusweisPageResponse.model_validate(
                    parsed
                )

            return {
                "label": parsed.label,
                "evidence_code": parsed.evidence_code,
                "confidence": float(parsed.confidence),
                "attempt_count": attempt_number,
                "retry_count": attempt_index,
            }

        except Exception as error:
            last_error = error

            if not is_retryable_securegpt_error(error):
                raise SecureGPTScreeningError(
                    message=(
                        "Non-retryable SecureGPT error for "
                        f"{image_path}: {error}"
                    ),
                    attempts=attempt_number,
                ) from error

            if attempt_number == MAX_ATTEMPTS:
                break

            delay_seconds = RETRY_DELAYS_SECONDS[
                attempt_index
            ]

            print(
                f"Temporary SecureGPT error for {image_path.name}. "
                f"Attempt {attempt_number}/{MAX_ATTEMPTS}. "
                f"Retrying in {delay_seconds} seconds."
            )

            time.sleep(delay_seconds)

    raise SecureGPTScreeningError(
        message=(
            f"SecureGPT failed after {MAX_ATTEMPTS} attempts "
            f"for {image_path}. Last error: {last_error}"
        ),
        attempts=MAX_ATTEMPTS,
    ) from last_error