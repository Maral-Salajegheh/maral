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

    confidence: float = Field(ge=0.0, le=1.0,
                              
                              descriptin= ("confidence in the selectet label."
                                            "Use a value between 0.0 and 1.0")
                              )
    
    
    class SecureGPTScreeningError(RuntimeError):
        """SecureGPT failure including the number of attempted calls."""
        
        def __init__(
             self, 
             message: str,
             attempts: int,
             )-> None:
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
