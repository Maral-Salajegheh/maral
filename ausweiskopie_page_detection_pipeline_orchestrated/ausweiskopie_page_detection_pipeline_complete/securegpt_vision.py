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
MAX_LONG_SIDE_PX = 2400
JPEG_QUALITY = 92

# Payload guard against the ESG gateway upload limit (error 413,
# ESG070 "Size limit exceeded"). Images are always sent as JPEG;
# raw PNG renders of photo-heavy pages can exceed the limit. If the
# encoded payload is still too large, quality and then resolution
# are stepped down until it fits. Adjust MAX_PAYLOAD_BYTES if the
# documented ESG limit differs.
MAX_PAYLOAD_BYTES = 4_000_000
FALLBACK_JPEG_QUALITIES = [80, 65]
FALLBACK_SCALE_FACTORS = [0.75, 0.5]


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


def _encode_jpeg_data_url(image: Image.Image, quality: int) -> str:
    """Encode one PIL image as a Base64 JPEG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_page_image(image_path: Path) -> str:
    """
    Load one page image, normalise its pixel size, and return a
    Base64 JPEG data URL that fits the ESG payload limit.

    - Pages whose short side is below MIN_SHORT_SIDE_PX are upscaled
      (card-format ID scans, thumbnail-sized renders).
    - Pages whose long side exceeds MAX_LONG_SIDE_PX are downscaled.
    - Every page is re-encoded as JPEG. Raw PNG renders of
      photo-heavy pages can exceed the ESG upload limit
      (error 413 / ESG070), so PNG is never sent.
    - If the encoded payload still exceeds MAX_PAYLOAD_BYTES, the
      JPEG quality and then the resolution are stepped down until
      it fits.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    with Image.open(image_path) as source:
        width, height = source.size
        short_side = min(width, height)
        long_side = max(width, height)

        scale = 1.0

        if short_side < MIN_SHORT_SIDE_PX:
            scale = MIN_SHORT_SIDE_PX / short_side

        if long_side * scale > MAX_LONG_SIDE_PX:
            scale = MAX_LONG_SIDE_PX / long_side

        image = source.convert("RGB")

        if scale != 1.0:
            new_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            image = image.resize(new_size, Image.LANCZOS)

        data_url = _encode_jpeg_data_url(image, JPEG_QUALITY)

        if len(data_url) <= MAX_PAYLOAD_BYTES:
            return data_url

        # Payload guard ladder: lower quality first, then resolution.
        for quality in FALLBACK_JPEG_QUALITIES:
            data_url = _encode_jpeg_data_url(image, quality)
            if len(data_url) <= MAX_PAYLOAD_BYTES:
                print(
                    f"Payload guard: {image_path.name} sent at "
                    f"JPEG quality {quality}."
                )
                return data_url

        for factor in FALLBACK_SCALE_FACTORS:
            reduced = image.resize(
                (
                    max(1, round(image.width * factor)),
                    max(1, round(image.height * factor)),
                ),
                Image.LANCZOS,
            )
            data_url = _encode_jpeg_data_url(
                reduced,
                FALLBACK_JPEG_QUALITIES[-1],
            )
            if len(data_url) <= MAX_PAYLOAD_BYTES:
                print(
                    f"Payload guard: {image_path.name} sent at "
                    f"{factor:.0%} resolution, JPEG quality "
                    f"{FALLBACK_JPEG_QUALITIES[-1]}."
                )
                return data_url

    raise ValueError(
        f"Image payload for {image_path} cannot be reduced below "
        f"{MAX_PAYLOAD_BYTES} bytes."
    )


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


# Call modes, tried in order. The first mode the installed wrapper
# accepts is cached for the whole run, so incompatible keyword
# arguments (image_detail, response_model) are probed exactly once
# instead of failing every page.
_CALL_MODES = [
    "full",           # image_detail + response_model
    "no_detail",      # response_model only
    "json_fallback",  # plain call, strict JSON parsed locally
]

_active_call_mode: str | None = None

_JSON_FALLBACK_SUFFIX = (
    "\nGib ausschließlich gültiges JSON mit genau diesen "
    'Schlüsseln zurück: {"label": "...", '
    '"evidence_code": "...", "confidence": 0.0}'
)


def _call_securegpt(
    client: SecureGPT,
    image_data_url: str,
) -> AusweisPageResponse:
    """
    Call new_chat with the first call mode the wrapper supports.

    A TypeError from an unsupported keyword argument moves to the
    next mode. The working mode is cached in _active_call_mode.
    """
    global _active_call_mode

    modes = (
        [_active_call_mode]
        if _active_call_mode is not None
        else _CALL_MODES
    )

    last_type_error: TypeError | None = None

    for mode in modes:
        try:
            if mode == "full":
                response = client.new_chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT,
                    user_image=image_data_url,
                    image_detail="high",
                    response_model=AusweisPageResponse,
                )
            elif mode == "no_detail":
                response = client.new_chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=USER_PROMPT,
                    user_image=image_data_url,
                    response_model=AusweisPageResponse,
                )
            else:
                response = client.new_chat(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=(
                        USER_PROMPT + _JSON_FALLBACK_SUFFIX
                    ),
                    user_image=image_data_url,
                )

        except TypeError as error:
            if _active_call_mode is not None:
                # The cached mode worked before; this TypeError
                # is a real bug, not a signature mismatch.
                raise

            last_type_error = error
            continue

        if _active_call_mode is None:
            _active_call_mode = mode
            print(f"SecureGPT call mode: {mode}")

        answer = (
            response.answer
            if hasattr(response, "answer")
            else response
        )

        if isinstance(answer, AusweisPageResponse):
            return answer

        if isinstance(answer, str):
            return AusweisPageResponse.model_validate_json(
                answer
            )

        return AusweisPageResponse.model_validate(answer)

    raise TypeError(
        "No supported new_chat signature found. "
        f"Last TypeError: {last_type_error}"
    ) from last_type_error


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
            parsed = _call_securegpt(
                client=client,
                image_data_url=image_data_url,
            )

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