from auth import parse_bearer_token, verify_token


def test_parse_bearer_token_extracts_token():
    assert parse_bearer_token("Bearer abc123") == "abc123"


def test_parse_bearer_token_case_insensitive_scheme():
    assert parse_bearer_token("bearer abc123") == "abc123"


def test_parse_bearer_token_returns_none_for_missing_header():
    assert parse_bearer_token(None) is None


def test_parse_bearer_token_returns_none_for_wrong_scheme():
    assert parse_bearer_token("Basic abc123") is None


def test_parse_bearer_token_returns_none_for_malformed_header():
    assert parse_bearer_token("abc123") is None


def test_verify_token_accepts_matching_token():
    assert verify_token("abc123", "abc123") is True


def test_verify_token_rejects_mismatched_token():
    assert verify_token("wrong", "abc123") is False


def test_verify_token_rejects_none():
    assert verify_token(None, "abc123") is False


def test_verify_token_rejects_when_both_token_and_expected_are_empty():
    # A misconfigured expected token (empty string) must never authenticate
    # anything, including a client that sends an empty token to match it.
    assert verify_token("", "") is False


def test_verify_token_rejects_empty_token_against_real_expected():
    assert verify_token("", "abc123") is False


def test_verify_token_rejects_non_string_token():
    # A non-str, truthy token (e.g. a JSON number from a malformed client
    # frame) must be rejected cleanly, not raise inside hmac.compare_digest.
    assert verify_token(12345, "abc123") is False
