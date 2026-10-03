from agent.prompt import build_system_prompt
from agent.skills import Skill


def make_skill(name, description="d", always_on=False, instructions="do the thing"):
    return Skill(
        name=name, description=description, always_on=always_on,
        instructions=instructions, reference_dir=None,
    )


def test_build_system_prompt_returns_base_prompt_unchanged():
    result = build_system_prompt("base prompt")

    assert result == "base prompt"


def test_build_system_prompt_appends_always_on_skill_instructions():
    skill = make_skill("safety", instructions="Always confirm before deleting a volume.")

    result = build_system_prompt("base prompt", always_on_skills=[skill])

    assert "base prompt" in result
    assert "Always confirm before deleting a volume." in result


def test_build_system_prompt_appends_on_demand_catalog():
    skill = make_skill("purestorage", description="PureStorage conventions.")

    result = build_system_prompt("base prompt", on_demand_skills=[skill])

    assert "purestorage" in result
    assert "PureStorage conventions." in result
    assert "load_skill" in result


def test_build_system_prompt_no_catalog_section_when_no_on_demand_skills():
    always_on = make_skill("safety", instructions="Confirm before deleting.")

    result = build_system_prompt("base prompt", always_on_skills=[always_on])

    assert "Available skills" not in result


def test_build_system_prompt_combines_both_kinds():
    always_on = make_skill("safety", instructions="Confirm before deleting.")
    on_demand = make_skill("purestorage", description="PureStorage conventions.")

    result = build_system_prompt("base prompt", always_on_skills=[always_on], on_demand_skills=[on_demand])

    assert "Confirm before deleting." in result
    assert "purestorage" in result
