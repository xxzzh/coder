"""Credentials kept out of the knowledge base: table columns titled Password, "password: ..." text,
and product keys. They would otherwise show up in answers and be sent to the answer API as
evidence; nothing else in a document is touched."""

from __future__ import annotations

import re

REDACTED = "[已隐去]"

# A column title or label that names a password.
_CREDENTIAL_LABEL = re.compile(r"pass\s*word|passwd|\bpwd\b|密码|密碼|口令", re.I)

# "Password: value" / "密码：value" written in running text.
_LABELLED_SECRET = re.compile(r"((?:pass\s*word|passwd|\bpwd|密码|密碼|口令)\s*[:：=]\s*)([^\s|，,；;]+)", re.I)

# Windows product keys: five groups of five letters or digits.
_PRODUCT_KEY = re.compile(r"\b[A-Z0-9]{5}(?:-[A-Z0-9]{5}){4}\b", re.I)


def is_credential_label(label: str) -> bool:
    return bool(label) and len(label) <= 30 and bool(_CREDENTIAL_LABEL.search(label))


def redact_secrets(text: str) -> str:
    text = _PRODUCT_KEY.sub(REDACTED, text)
    return _LABELLED_SECRET.sub(lambda match: match.group(1) + REDACTED, text)
