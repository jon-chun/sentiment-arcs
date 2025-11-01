# sentimentarcs User Manual

**Important name note**

- PyPI package: `sentimentarcs`
- GitHub repo: **`sentiment-arcs`** (dash!)
- Python import: `import sentimentarcs`

This manual matches the code in: https://github.com/jon-chun/sentiment-arcs

---

## 1. Overview

`sentimentarcs` turns long texts into **sentiment time series** ("arcs") and finds **crux** passages.
It is designed to be extracted from Colab notebooks and published as a reusable PyPI package.

## 2. Install

From PyPI:

```bash
pip install sentimentarcs
```

From GitHub:

```bash
git clone https://github.com/jon-chun/sentiment-arcs.git
cd sentiment-arcs
pip install -e .
```

## 3. Configuration

- Global defaults: `config.yaml`
- Secrets: `.env`

## 4. CLI

```bash
sentimentarcs novel_the-great-gatsby_f-scott-fitzgerald.txt --model vader
```

## 5. Debug

- Wrong filename? Must look like: `novel_the-great-gatsby_f-scott-fitzgerald.txt`
- No scipy? Install: `pip install "sentimentarcs[scipy]"`
- No transformers? Install: `pip install "sentimentarcs[hf]"`

## 6. API (short)

```python
from sentimentarcs.pipeline import run_pipeline
run_pipeline("novel_the-great-gatsby_f-scott-fitzgerald.txt")
```
