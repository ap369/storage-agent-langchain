from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool


class SandboxViolation(Exception):
    pass


def resolve_in_sandbox(sandbox_root: Path, user_path: str) -> Path:
    candidate = Path(user_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SandboxViolation(f"path escapes sandbox: {user_path!r}")

    resolved = (sandbox_root / candidate).resolve(strict=False)
    if not resolved.is_relative_to(sandbox_root):
        raise SandboxViolation(f"path escapes sandbox: {user_path!r}")

    return resolved


def build_file_tools(sandbox_root: Path) -> list[BaseTool]:
    async def list_dir(path: str = ".") -> str:
        """List files and directories at a path relative to the sandbox root."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(entries) if entries else "(empty directory)"

    async def search_files(pattern: str, path: str = ".") -> str:
        """Search for files matching a glob pattern under a path relative to the sandbox root."""
        target = resolve_in_sandbox(sandbox_root, path)
        matches = []
        for p in target.rglob(pattern):
            if not p.is_file():
                continue
            # rglob() doesn't resolve ".." segments a pattern can introduce, and
            # Path.relative_to() is purely syntactic — resolve before checking
            # containment, the same check resolve_in_sandbox() makes.
            resolved = p.resolve(strict=False)
            if not resolved.is_relative_to(sandbox_root):
                continue
            matches.append(str(resolved.relative_to(sandbox_root)))
        return "\n".join(sorted(matches)) if matches else "(no matches)"

    async def read_file(path: str) -> str:
        """Read a file's full text content."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        return target.read_text()

    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the given text content."""
        target = resolve_in_sandbox(sandbox_root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    async def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace the first occurrence of old_text with new_text in a file."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        content = target.read_text()
        if old_text not in content:
            raise ValueError(f"text not found in {path}: {old_text!r}")
        target.write_text(content.replace(old_text, new_text, 1))
        return f"edited {path}"

    async def delete_file(path: str) -> str:
        """Delete a file."""
        target = resolve_in_sandbox(sandbox_root, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        target.unlink()
        return f"deleted {path}"

    async def move_file(src: str, dest: str) -> str:
        """Move or rename a file."""
        src_target = resolve_in_sandbox(sandbox_root, src)
        dest_target = resolve_in_sandbox(sandbox_root, dest)
        if not src_target.is_file():
            raise FileNotFoundError(f"not a file: {src}")
        dest_target.parent.mkdir(parents=True, exist_ok=True)
        src_target.rename(dest_target)
        return f"moved {src} to {dest}"

    return [
        tool(list_dir),
        tool(search_files),
        tool(read_file),
        tool(write_file),
        tool(edit_file),
        tool(delete_file),
        tool(move_file),
    ]
