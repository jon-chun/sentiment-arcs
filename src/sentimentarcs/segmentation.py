import re

def _segment_pysbd(text: str, language: str = "en") -> list[str]:
    import pysbd
    seg = pysbd.Segmenter(language=language, clean=False)
    return seg.segment(text)

def _segment_regex(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def _segment_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

def _segment_window(text: str, window: int) -> list[str]:
    base = _segment_regex(text)
    return [" ".join(base[i:i+window]) for i in range(0, len(base), window)]

def segment_text(text_clean: str, clean_tech: str = "pysbd") -> list[str]:
    if clean_tech == "pysbd":
        return _segment_pysbd(text_clean)
    if clean_tech == "regex":
        return _segment_regex(text_clean)
    if clean_tech == "paragraph":
        return _segment_paragraphs(text_clean)
    if clean_tech.startswith("win"):
        return _segment_window(text_clean, int(clean_tech[3:]))
    raise ValueError(f"Unknown clean_tech: {clean_tech}")
