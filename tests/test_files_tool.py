import pytest

from agent.tools.files import build_file_tools


def make_tools(tmp_path):
    return {t.name: t for t in build_file_tools(tmp_path)}


async def test_write_then_read_round_trip(tmp_path):
    tools = make_tools(tmp_path)

    await tools["write_file"].ainvoke({"path": "notes.txt", "content": "hello"})
    result = await tools["read_file"].ainvoke({"path": "notes.txt"})

    assert result == "hello"


async def test_write_file_creates_parent_directories(tmp_path):
    tools = make_tools(tmp_path)

    await tools["write_file"].ainvoke({"path": "a/b/c.txt", "content": "deep"})

    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


async def test_read_file_missing_raises(tmp_path):
    tools = make_tools(tmp_path)

    with pytest.raises(FileNotFoundError):
        await tools["read_file"].ainvoke({"path": "missing.txt"})


async def test_list_dir_lists_entries(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    tools = make_tools(tmp_path)

    result = await tools["list_dir"].ainvoke({"path": "."})

    assert "a.txt" in result
    assert "sub/" in result


async def test_list_dir_empty_directory(tmp_path):
    tools = make_tools(tmp_path)

    result = await tools["list_dir"].ainvoke({"path": "."})

    assert result == "(empty directory)"


async def test_search_files_matches_glob(tmp_path):
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    tools = make_tools(tmp_path)

    result = await tools["search_files"].ainvoke({"pattern": "*.md", "path": "."})

    assert "a.md" in result
    assert "b.txt" not in result


async def test_edit_file_replaces_first_occurrence(tmp_path):
    (tmp_path / "f.txt").write_text("foo bar foo")
    tools = make_tools(tmp_path)

    await tools["edit_file"].ainvoke({"path": "f.txt", "old_text": "foo", "new_text": "baz"})

    assert (tmp_path / "f.txt").read_text() == "baz bar foo"


async def test_edit_file_raises_when_text_not_found(tmp_path):
    (tmp_path / "f.txt").write_text("foo")
    tools = make_tools(tmp_path)

    with pytest.raises(ValueError):
        await tools["edit_file"].ainvoke({"path": "f.txt", "old_text": "missing", "new_text": "x"})


async def test_delete_file_removes_file(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    tools = make_tools(tmp_path)

    await tools["delete_file"].ainvoke({"path": "f.txt"})

    assert not (tmp_path / "f.txt").exists()


async def test_move_file_renames(tmp_path):
    (tmp_path / "src.txt").write_text("x")
    tools = make_tools(tmp_path)

    await tools["move_file"].ainvoke({"src": "src.txt", "dest": "dest.txt"})

    assert not (tmp_path / "src.txt").exists()
    assert (tmp_path / "dest.txt").read_text() == "x"


async def test_move_file_missing_source_raises(tmp_path):
    tools = make_tools(tmp_path)

    with pytest.raises(FileNotFoundError):
        await tools["move_file"].ainvoke({"src": "missing.txt", "dest": "dest.txt"})


def test_build_file_tools_returns_all_seven_tools(tmp_path):
    names = {t.name for t in build_file_tools(tmp_path)}
    assert names == {
        "list_dir", "search_files", "read_file",
        "write_file", "edit_file", "delete_file", "move_file",
    }
