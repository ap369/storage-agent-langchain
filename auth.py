import hmac


def parse_bearer_token(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def verify_token(token: str | None, expected: str) -> bool:
    if not isinstance(token, str) or not token or not expected:
        return False
    return hmac.compare_digest(token, expected)
