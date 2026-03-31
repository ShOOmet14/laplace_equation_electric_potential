# internationalization file

import json
from functools import lru_cache
from typing import Any, cast

from src.common.paths import PROJECT_ROOT


TRANSLATIONS_PATH = PROJECT_ROOT / "src" / "resources" / "translations.json"


@lru_cache(maxsize=1)
def _load_translations() -> dict[str, Any]:
    with TRANSLATIONS_PATH.open("r", encoding="utf-8") as file:
        return cast(dict[str, Any], json.load(file))


class Translator:
    def __init__(self, language: str) -> None:
        all_translations = _load_translations()

        if language not in all_translations:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self.data = cast(dict[str, Any], all_translations[language])

    def t(self, key: str, **kwargs: Any) -> str:
        value: Any = self.data

        for part in key.split("."):
            value = value[part]

        if not isinstance(value, str):
            raise TypeError(f"Translation key '{key}' does not point to a string")

        return value.format(**kwargs)

    def section(self, key: str) -> dict[str, str]:
        value: Any = self.data

        for part in key.split("."):
            value = value[part]

        if not isinstance(value, dict):
            raise TypeError(f"Translation key '{key}' does not point to a dictionary")

        return cast(dict[str, str], value)
