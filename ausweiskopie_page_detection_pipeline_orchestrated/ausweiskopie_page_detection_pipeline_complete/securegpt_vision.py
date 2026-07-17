from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from axallm.securegpt.v2.providers import OpenAIProvider
from axallm.securegpt.v2.securegpt import SecureGPT


MODEL_NAME = os.getenv("SECUREGPT_MODEL_NAME", "")
MODEL_VERSION = os.getenv("SECUREGPT_MODEL_VERSION", "")

TEMPERATURE = 0
SEED = 42


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

4. Krankenversicherungskarten, Bankkarten, Kundenkarten, Mitarbeiterausweise,
   Führerscheine und reine Passfotos gelten nicht als Ausweiskopie.

5. Wenn nicht zuverlässig entschieden werden kann, verwende unclear.
   Rate nicht.

6. Gib niemals personenbezogene Inhalte wieder. Transkribiere insbesondere
   keine Namen, Geburtsdaten, Anschriften, Dokumentennummern, Unterschriften
   oder MRZ-Inhalte.

7. Gib ausschließlich die angeforderte strukturierte Antwort zurück.
""".strip()


USER_PROMPT = """
Prüfe dieses Seitenbild.

Klassifiziere es als:

- ausweiskopie
- not_ausweiskopie
- unclear

Gib zusätzlich genau einen zulässigen evidence_code zurück.
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
) -> dict[str, str]:
    """
    Send one page image to SecureGPT.

    The wrapper is first called with Pydantic structured output. If the
    installed version does not support response_model together with
    user_image, it falls back to strict JSON and validates locally.
    """
    image_data_url = image_to_data_url(image_path)

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
    }
