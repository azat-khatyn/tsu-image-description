from typing import Dict, List


class ThemeInferencer:
    def infer(self, metadata: Dict) -> Dict:
        objects = metadata.get("objects", [])
        tags = metadata.get("tags", [])

        theme = "generic illustrated scene"
        mood = "decorative"

        if "egg" in objects and "chicks" in objects:
            theme = "spring or Easter"
            mood = "festive"

        if "children" in objects and theme == "generic illustrated scene":
            theme = "children's scene"

        return {
            "theme": theme,
            "mood": mood
        }
