"""Professor profile: the consent record behind an avatar or voice clone.

A presenter avatar or cloned voice may only be used with its owner's
signed consent and the vendor's identity verification (Part E). The
profile is the professor's own file; vendor keys never live here (they come
from the environment and never reach any artifact).
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ConsentRecord(BaseModel):
    statement: str = Field(..., min_length=20, description="The signed consent text")
    signed_by: str
    signed_at: str = Field(..., description="ISO date of signature")


class VendorIdentity(BaseModel):
    provider: str
    id: str = Field(..., description="Vendor voice or avatar id")
    verified: bool = Field(
        default=False, description="The vendor's own identity verification passed"
    )


class ProfessorProfile(BaseModel):
    name: str
    consent: Optional[ConsentRecord] = None
    voice: Optional[VendorIdentity] = None
    avatar: Optional[VendorIdentity] = None

    @classmethod
    def load(cls, path: str) -> "ProfessorProfile":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class ConsentError(Exception):
    """Raised before any vendor call when consent or verification is missing."""


def require_consent(
    profile: Optional[ProfessorProfile],
    kind: Literal["voice", "avatar"],
    provider: str,
) -> VendorIdentity:
    """The verified vendor identity to use, or a ConsentError explaining why not."""
    if profile is None:
        raise ConsentError(f"Using a {kind} clone needs a professor profile (--profile) with signed consent.")
    if profile.consent is None:
        raise ConsentError(f"{profile.name}'s profile has no signed consent statement.")
    identity = getattr(profile, kind)
    if identity is None or identity.provider != provider:
        raise ConsentError(f"{profile.name}'s profile has no {provider} {kind} on record.")
    if not identity.verified:
        raise ConsentError(
            f"{profile.name}'s {provider} {kind} has not passed the vendor's identity verification."
        )
    return identity
