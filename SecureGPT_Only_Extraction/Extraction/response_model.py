"""Required, nullable extraction fields. No guessed values or implicit coercion."""
from datetime import datetime
import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FIELDS = ("geburtsort", "ausweisnummer", "ausweistyp", "gueltigkeitsdatum",
          "ausstellende_behoerde", "nationalitaet")
TYPE_CODES = {"personalausweis": "P", "reisepass": "R", "dienstpass": "R",
              "diplomatenpass": "R", "aufenthaltstitel_als_passersatz": "S"}
KINDS = set(TYPE_CODES) | {"not_accepted", "unknown"}
Kind = Literal["personalausweis", "reisepass", "dienstpass", "diplomatenpass",
               "aufenthaltstitel_als_passersatz", "not_accepted", "unknown"]


def valid_date(value):
    if not re.fullmatch(r"[0-9]{6}", value):
        return False
    try:
        datetime(2000 + int(value[:2]), int(value[2:4]), int(value[4:]))
        return True
    except ValueError:
        return False


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value):
        return value.strip() or None if isinstance(value, str) else value


class ExtractedFields(StrictModel):
    geburtsort: str | None
    ausweisnummer: str | None
    ausweistyp: Literal["P", "R", "S"] | None
    gueltigkeitsdatum: str | None
    ausstellende_behoerde: str | None
    nationalitaet: str | None

    @field_validator("gueltigkeitsdatum")
    @classmethod
    def expiry_format(cls, value):
        if value and value != "UNBEFRISTET" and not valid_date(value):
            raise ValueError("Expiry must be YYMMDD, UNBEFRISTET, or null")
        return value


class Identity(StrictModel):
    issuing_state: str | None
    holder_name: str | None
    birth_date: str | None

    @field_validator("birth_date")
    @classmethod
    def birth_format(cls, value):
        if value and not valid_date(value):
            raise ValueError("Birth date must be YYMMDD or null")
        return value


class DocumentObservation(StrictModel):
    document_kind: Kind
    type_evidence: str | None
    fields: ExtractedFields
    identity: Identity

    @model_validator(mode="after")
    def consistent_type(self):
        expected = TYPE_CODES.get(self.document_kind)
        if self.fields.ausweistyp != expected:
            raise ValueError("ausweistyp must match document_kind; unknown/excluded require null")
        if expected and not self.type_evidence:
            raise ValueError("Accepted kind requires visible type evidence")
        return self


class ExtractionResponse(StrictModel):
    documents: list[DocumentObservation]
