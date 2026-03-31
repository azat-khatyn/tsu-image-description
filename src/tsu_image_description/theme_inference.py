class ThemeInferencer:
    def infer(self, metadata: dict) -> dict:
        theme_field = metadata.get("theme", {})
        mood_field = metadata.get("mood", {})

        theme = theme_field.get("label") if theme_field.get("confident") else None
        mood = mood_field.get("label") if mood_field.get("confident") else None

        return {
            "theme": theme,
            "mood": mood,
            "theme_confidence": theme_field.get("score"),
            "mood_confidence": mood_field.get("score"),
        }
