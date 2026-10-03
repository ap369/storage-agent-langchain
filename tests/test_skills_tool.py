import pytest

from agent.skills import Skill
from agent.tools.skills import build_skill_tools


def make_skill(tmp_path, name="purestorage", always_on=False, reference_files=None):
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    reference_dir = None
    if reference_files:
        reference_dir = skill_dir / "reference"
        reference_dir.mkdir()
        for filename, content in reference_files.items():
            (reference_dir / filename).write_text(content)
    return Skill(
        name=name,
        description="A skill.",
        always_on=always_on,
        instructions="Full instructions.",
        reference_dir=reference_dir,
    )


def test_build_skill_tools_empty_list_returns_no_tools():
    assert build_skill_tools([]) == []


def test_build_skill_tools_returns_load_and_read_tools(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name for t in build_skill_tools([skill])}
    assert tools == {"load_skill", "read_skill_file"}


async def test_load_skill_returns_instructions(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert result == "Full instructions."


async def test_load_skill_lists_reference_files(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert "Full instructions." in result
    assert "cheatsheet.md" in result
    assert "Reference files available" in result


async def test_load_skill_omits_reference_note_when_dir_empty(tmp_path):
    skill_dir = tmp_path / "purestorage"
    skill_dir.mkdir()
    (skill_dir / "reference").mkdir()
    skill = Skill(
        name="purestorage", description="d", always_on=False,
        instructions="Full instructions.", reference_dir=skill_dir / "reference",
    )
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["load_skill"].ainvoke({"name": "purestorage"})

    assert result == "Full instructions."


async def test_load_skill_unknown_name_raises(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(ValueError, match="unknown skill"):
        await tools["load_skill"].ainvoke({"name": "nonexistent"})


async def test_read_skill_file_returns_content(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "the content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    result = await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "cheatsheet.md"})

    assert result == "the content"


async def test_read_skill_file_truncates_large_files(tmp_path):
    skill = make_skill(tmp_path, reference_files={"big.md": "x" * 100})
    tools = {t.name: t for t in build_skill_tools([skill], max_reference_bytes=50)}

    result = await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "big.md"})

    assert "truncated at 50 bytes" in result


async def test_read_skill_file_rejects_traversal(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(Exception):
        await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "../../SKILL.md"})


async def test_read_skill_file_unknown_skill_raises(tmp_path):
    skill = make_skill(tmp_path)
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(ValueError, match="unknown skill"):
        await tools["read_skill_file"].ainvoke({"skill": "nonexistent", "path": "x.md"})


async def test_read_skill_file_unknown_path_raises(tmp_path):
    skill = make_skill(tmp_path, reference_files={"cheatsheet.md": "content"})
    tools = {t.name: t for t in build_skill_tools([skill])}

    with pytest.raises(FileNotFoundError):
        await tools["read_skill_file"].ainvoke({"skill": "purestorage", "path": "missing.md"})
