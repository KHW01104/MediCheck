"""Retrieve short document-based explanations for MediCheck risk factors."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_PATH = PROJECT_ROOT / "docs" / "medical_knowledge.md"
FALLBACK_EXPLANATION = (
    "이 요인은 MediCheck 예측 모델에서 참고한 입력 특성입니다. "
    "일반적으로 관련이 있을 수 있는 정보로만 해석해야 하며, 진단, 치료, 처방 조언을 의미하지 않습니다."
)


def _normalize(value: str) -> str:
    """Normalize a factor name for simple alias matching."""
    return value.strip().lower().replace("_", " ")


def _parse_sections(markdown_text: str) -> dict[str, dict[str, object]]:
    """Parse medical_knowledge.md sections and their aliases."""
    sections: dict[str, dict[str, object]] = {}
    current_key: str | None = None
    aliases: list[str] = []
    body_lines: list[str] = []

    def flush() -> None:
        if current_key is None:
            return
        explanation = "\n".join(line for line in body_lines if line.strip()).strip()
        sections[current_key] = {
            "aliases": aliases,
            "explanation": explanation,
        }

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            flush()
            current_key = line.removeprefix("## ").strip()
            aliases = [current_key]
            body_lines = []
            continue
        if current_key is None:
            continue
        if line.startswith("Aliases:"):
            alias_text = line.removeprefix("Aliases:").strip()
            aliases.extend(alias.strip() for alias in alias_text.split(",") if alias.strip())
            continue
        body_lines.append(line)

    flush()
    return sections


@lru_cache(maxsize=1)
def load_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> dict[str, str]:
    """Load factor aliases mapped to short explanations."""
    knowledge_path = Path(path)
    if not knowledge_path.exists():
        return {}

    sections = _parse_sections(knowledge_path.read_text(encoding="utf-8"))
    alias_map: dict[str, str] = {}
    for section in sections.values():
        explanation = str(section["explanation"])
        for alias in section["aliases"]:
            alias_map[_normalize(str(alias))] = explanation
    return alias_map


def get_factor_explanation(factor_name: str) -> str:
    """Return a short general explanation for a MediCheck risk factor."""
    alias_map = load_knowledge_base()
    normalized = _normalize(factor_name)
    if normalized in alias_map:
        return alias_map[normalized]

    for alias, explanation in alias_map.items():
        if alias in normalized or normalized in alias:
            return explanation
    return FALLBACK_EXPLANATION

