from langchain_core.tools import BaseTool
from langchain.tools import tool

from agent.skills import Skill
from agent.tools.files import resolve_in_sandbox

DEFAULT_MAX_REFERENCE_FILE_BYTES = 1_000_000


def build_skill_tools(
    on_demand_skills: list[Skill],
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_FILE_BYTES,
) -> list[BaseTool]:
    if not on_demand_skills:
        return []

    skills_by_name = {skill.name: skill for skill in on_demand_skills}

    async def load_skill(name: str) -> str:
        """Load a storage-domain skill's instructions by name."""
        skill = skills_by_name.get(name)
        if skill is None:
            raise ValueError(f"unknown skill: {name!r}")

        result = skill.instructions

        if skill.reference_dir is not None:
            files = sorted(
                str(p.relative_to(skill.reference_dir))
                for p in skill.reference_dir.rglob("*")
                if p.is_file()
            )
            if files:
                file_list = "\n".join(f"- {f}" for f in files)
                result += (
                    f"\n\nReference files available (use read_skill_file to view one):\n{file_list}"
                )

        return result

    async def read_skill_file(skill: str, path: str) -> str:
        """Read one reference file belonging to a loaded skill. `path` is relative to the skill's reference/ directory."""
        skill_obj = skills_by_name.get(skill)
        if skill_obj is None:
            raise ValueError(f"unknown skill: {skill!r}")
        if skill_obj.reference_dir is None:
            raise ValueError(f"skill {skill!r} has no reference files")

        target = resolve_in_sandbox(skill_obj.reference_dir, path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")

        data = target.read_bytes()
        text = data[:max_reference_bytes].decode(errors="replace")
        if len(data) > max_reference_bytes:
            text += f"\n... truncated at {max_reference_bytes} bytes"
        return text

    return [
        tool(load_skill),
        tool(read_skill_file),
    ]
