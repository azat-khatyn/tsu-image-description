class ThemeInferencer:
    def infer(self, metadata: dict) -> dict:
        return {
            "theme": metadata.get("theme", {}),
            "mood": metadata.get("mood", {}),
        }