from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from axallm.securegpt.v2.providers import OpenAIProvider
from axallm.securegpt.v2.securegpt import SecureGPT


MODEL_NAME = os.getenv("SECUREGPT_MODEL_NAME", "")
MODEL_VERSION = os.getenv("SECUREGPT_MODEL_VERSION", "")

TEMPERATURE = 0
SEED = 42

MAX_ATTEMPTS = 5
RETRY_DELAYS_SECONDS = [1, 2, 4, 8]

# Size normalisation for page images sent to SecureGPT.
#
# Pages containing Ausweiskopien are often not normal A4 pages:
# a directly scanned ID card produces a tiny card-format page, and
# some scans are oversized. Tiny pages get rendered as tiny images
# that the vision model cannot read reliably; oversized pages get
# aggressively downsampled by the API. Both cases are normalised
# here before the image is sent.
MIN_SHORT_SIDE_PX = 1024
MAX_LONG_SIDE_PX = 2200
JPEG_QUALITY = 92


class SecureGPTScreeningError(RuntimeError):
    """SecureGPT failure including the number of attempted calls."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


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

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the selected label. "
            "Use a value between 0.0 and 1.0."
        ),
    )


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

4. Ausweiskopien können in stark unterschiedlicher Größe und in
   unterschiedlichen Seitenformaten erscheinen:
   - als kleine eingebettete Abbildung auf einer normal großen Seite,
   - als eigene Seite im Kartenformat, die deutlich kleiner ist als
     andere Seiten des Dokuments,
   - als vergrößerter oder verkleinerter Scan.
   Beurteile den Inhalt unabhängig von Seitengröße, Auflösung und
   Seitenverhältnis. Prüfe auch kleine Bildbereiche der Seite sorgfältig
   auf Merkmale eines Identitätsdokuments (Kartenlayout, Porträtfoto mit
   Datenfeldern, MRZ-ähnliche Zeilen, Vorder- und Rückseiten).

5. Krankenversicherungskarten, Bankkarten, Kundenkarten,
   Mitarbeiterausweise, Führerscheine und reine Passfotos gelten nicht als
   Ausweiskopie.

6. Wenn nicht zuverlässig entschieden werden kann, verwende unclear.
   Rate nicht.

7. Gib niemals personenbezogene Inhalte wieder. Transkribiere insbesondere
   keine Namen, Geburtsdaten, Anschriften, Dokumentennummern,
   Unterschriften oder MRZ-Inhalte.

8. Gib zusätzlich einen confidence-Wert zwischen 0.0 und 1.0 zurück:

   - 0.90 bis 1.00:
     Das Ergebnis ist visuell eindeutig.

   - 0.60 bis 0.89:
     Das Ergebnis ist wahrscheinlich, aber nicht vollständig eindeutig.

   - unter 0.60:
     Das Ergebnis ist stark unsicher und sollte manuell geprüft werden.

9. Verwende bei stark unsicheren Bildern vorzugsweise das Label unclear.

10. Gib ausschließlich die angeforderte strukturierte Antwort zurück.
""".strip()


USER_PROMPT = """
Prüfe dieses Seitenbild.

Gib zurück:

- label
- evidence_code
- confidence

Gib keine personenbezogenen Informationen zurück.
""".strip()


def validate_configuration() -> None:
    if not MODEL_NAME:
        raise ValueError("SECUREGPT_MODEL_NAME is not set.")

    if not MODEL_VERSION:
        raise ValueError("SECUREGPT_MODEL_VERSION is not set.")


def normalize_page_image(image_path: Path) -> str:
    """
    Load one page image, normalise its pixel size, and return a
    Base64 data URL.

    - Pages whose short side is below MIN_SHORT_SIDE_PX are upscaled
      (card-format ID scans, thumbnail-sized renders).
    - Pages whose long side exceeds MAX_LONG_SIDE_PX are downscaled.
    - Pages already within range are passed through unchanged from
      the original file bytes.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    suffix = image_path.suffix.lower()

    if suffix == ".png":
        mime_type = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    with Image.open(image_path) as image:
        width, height = image.size
        short_side = min(width, height)
        long_side = max(width, height)

        scale = 1.0

        if short_side < MIN_SHORT_SIDE_PX:
            scale = MIN_SHORT_SIDE_PX / short_side

        if long_side * scale > MAX_LONG_SIDE_PX:
            scale = MAX_LONG_SIDE_PX / long_side

        if scale == 1.0:
            encoded = base64.b64encode(
                image_path.read_bytes()
            ).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"

        new_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )

        resized = image.convert("RGB").resize(
            new_size,
            Image.LANCZOS,
        )

        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=JPEG_QUALITY)

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


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


def is_retryable_securegpt_error(error: Exception) -> bool:
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

    return any(marker in error_text for marker in retryable_markers)


def screen_image(
    client: SecureGPT,
    image_path: Path,
) -> dict[str, str | float | int]:
    """
    Classify one rendered page.

    Temporary infrastructure errors are retried with delays of
    1, 2, 4 and 8 seconds. Everything else fails immediately as
    SecureGPTScreeningError.
    """
    image_data_url = normalize_page_image(image_path)

    last_error: Exception | None = None

    for attempt_index in range(MAX_ATTEMPTS):
        attempt_number = attempt_index + 1

        try:
            response = client.new_chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=USER_PROMPT,
                user_image=image_data_url,
                image_detail="high",
                response_model=AusweisPageResponse,
            )

            parsed = response.answer

            if not isinstance(parsed, AusweisPageResponse):
                parsed = AusweisPageResponse.model_validate(parsed)

            return {
                "label": parsed.label,
                "evidence_code": parsed.evidence_code,
                "confidence": float(parsed.confidence),
                "attempt_count": attempt_number,
                "retry_count": attempt_index,
            }

        except ValidationError as error:
            raise SecureGPTScreeningError(
                message=(
                    "SecureGPT returned an invalid structured "
                    f"response for {image_path}: {error}"
                ),
                attempts=attempt_number,
            ) from error

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

            delay_seconds = RETRY_DELAYS_SECONDS[attempt_index]

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