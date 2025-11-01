from cleantext import clean as _clean

def clean_text(text_input: str) -> str:
    cleaned = _clean(text_input, fix_unicode=True, to_ascii=True, lower=False)
    cleaned = " ".join(cleaned.replace("\r", " ").replace("\n", " ").split())
    return cleaned
