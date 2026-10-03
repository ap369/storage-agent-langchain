import logging

import pytest

from agent.skills import DuplicateSkillName, load_skills


def write_skill(base, folder_name, name=None, description="A skill.", always_on=False, body="Do the thing.", reference_files=None):
    name = name or folder_name
    skill_dir = base / folder_name
    skill_dir.mkdir()
    always_on_line = str(always_on).lower()
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\nalways_on: {always_on_line}\n---\n{body}\n"
    )
    if reference_files:
        ref_dir = skill_dir / "reference"
        ref_dir.mkdir()
        for filename, content in reference_files.items():
            (ref_dir / filename).write_text(content)
    return skill_dir


def test_load_skills_parses_valid_skill(tmp_path):
    write_skill(tmp_path, "purestorage", description="PureStorage conventions.", always_on=False, body="Full instructions here.")

    skills = load_skills(tmp_path)

    assert len(skills) == 1
    skill = skills[0]
    assert skill.name == "purestorage"
    assert skill.description == "PureStorage conventions."
    assert skill.always_on is False
    assert skill.instructions == "Full instructions here."
    assert skill.reference_dir is None


def test_load_skills_detects_reference_dir(tmp_path):
    write_skill(tmp_path, "purestorage", reference_files={"api.md": "cheatsheet"})

    skills = load_skills(tmp_path)

    assert skills[0].reference_dir == tmp_path / "purestorage" / "reference"


def test_load_skills_returns_empty_list_for_missing_dir(tmp_path):
    assert load_skills(tmp_path / "does-not-exist") == []


def test_load_skills_skips_folder_without_skill_md(tmp_path):
    (tmp_path / "empty-folder").mkdir()
    write_skill(tmp_path, "purestorage")

    skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_skips_missing_frontmatter(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("just some text, no frontmatter")
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]
    assert "broken" in caplog.text


def test_load_skills_skips_missing_name_or_description(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: broken\n---\nbody")
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_skips_non_boolean_always_on(tmp_path, caplog):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        '---\nname: broken\ndescription: d\nalways_on: "false"\n---\nbody'
    )
    write_skill(tmp_path, "purestorage")

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert [s.name for s in skills] == ["purestorage"]


def test_load_skills_raises_on_duplicate_name(tmp_path):
    write_skill(tmp_path, "purestorage", name="purestorage")
    write_skill(tmp_path, "purestorage-2", name="purestorage", description="dup")

    with pytest.raises(DuplicateSkillName):
        load_skills(tmp_path)


def test_load_skills_warns_on_large_always_on_instructions(tmp_path, caplog):
    write_skill(tmp_path, "big", always_on=True, body="x" * 25_000)

    with caplog.at_level(logging.WARNING):
        skills = load_skills(tmp_path)

    assert len(skills) == 1
    assert "big" in caplog.text
