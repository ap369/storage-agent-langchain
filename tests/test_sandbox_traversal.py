import pytest

from agent.tools.files import SandboxViolation, resolve_in_sandbox


def test_resolve_in_sandbox_allows_normal_relative_path(tmp_path):
    result = resolve_in_sandbox(tmp_path, "notes.txt")
    assert result == tmp_path / "notes.txt"


def test_resolve_in_sandbox_rejects_absolute_path(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "/etc/passwd")


def test_resolve_in_sandbox_rejects_dotdot(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "../outside.txt")


def test_resolve_in_sandbox_rejects_dotdot_in_middle(tmp_path):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "subdir/../../outside.txt")


def test_resolve_in_sandbox_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")

    escape_link = tmp_path / "escape"
    escape_link.symlink_to(outside)

    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(tmp_path, "escape/secret.txt")


def test_resolve_in_sandbox_allows_new_file_that_does_not_exist_yet(tmp_path):
    result = resolve_in_sandbox(tmp_path, "new/nested/file.txt")
    assert result == tmp_path / "new" / "nested" / "file.txt"
