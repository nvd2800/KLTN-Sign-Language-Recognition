from __future__ import annotations
import re

VALID_SPECIAL_CLASSES = {"REST", "UNKNOWN"}
ID_RE = re.compile(r"^[a-z0-9_]+$")
SUBJECT_RE = re.compile(r"^S\d{2,}$")
SESSION_RE = re.compile(r"^SES\d{2,}$")


def validate_sign_name(sign: str) -> None:
    if sign in VALID_SPECIAL_CLASSES:
        return
    if not ID_RE.fullmatch(sign):
        raise ValueError("sign must use lowercase ASCII snake_case, or REST/UNKNOWN")


def validate_subject_id(subject_id: str) -> None:
    if not SUBJECT_RE.fullmatch(subject_id):
        raise ValueError("subject_id must look like S01")


def validate_session_id(session_id: str) -> None:
    if not SESSION_RE.fullmatch(session_id):
        raise ValueError("session_id must look like SES01")
