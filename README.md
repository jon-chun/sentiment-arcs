# sentimentarcs

This is the codebase for the **PyPI package** `sentimentarcs`, but the GitHub repo name uses a dash:

👉 https://github.com/jon-chun/sentiment-arcs

So the mapping is:

- **PyPI name**: `sentimentarcs`
- **GitHub repo**: `jon-chun/sentiment-arcs`
- **Python package import**: `import sentimentarcs`

## Install

From PyPI (once published):

```bash
pip install sentimentarcs
```

From GitHub:

```bash
git clone https://github.com/jon-chun/sentiment-arcs.git
cd sentiment-arcs
pip install -e .
```

## Usage

```bash
sentimentarcs novel_the-great-gatsby_f-scott-fitzgerald.txt --model vader
```

Outputs will go to `./outputs` and will be named like:

```text
sentimentarcs_plot_novel_the-great-gatsby-f-scott-fitzgerald_vader_sma.png
```

See `docs-pipy/user_manual.md` for the full pipeline spec.
