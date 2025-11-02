from pathlib import Path
import matplotlib.pyplot as plt

def plot_sentiments(df, *, output_path, title, subtitle):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["line_no"], df["sentiment_norm"], label="sentiment (norm)")
    ax.set_xlabel("Line / Segment")
    ax.set_ylabel("Sentiment [-1, 1]")
    ax.set_title(title)
    ax.text(0.01, 0.01, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
