from agent.skills import Skill


def build_system_prompt(
    base_prompt: str,
    always_on_skills: list[Skill] | None = None,
    on_demand_skills: list[Skill] | None = None,
) -> str:
    always_on_skills = always_on_skills or []
    on_demand_skills = on_demand_skills or []

    sections = [base_prompt]

    for skill in always_on_skills:
        sections.append(f"## Skill: {skill.name}\n\n{skill.instructions}")

    if on_demand_skills:
        catalog_lines = "\n".join(f"- {s.name} — {s.description}" for s in on_demand_skills)
        sections.append(
            "## Available skills\n\n"
            "Call load_skill(name) to load full instructions for one of these "
            "when relevant to the task:\n\n" + catalog_lines
        )

    return "\n\n".join(sections)
