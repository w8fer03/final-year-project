"""Read-only deployment check. Run: python scripts/check_research_data.py"""
from pathlib import Path
import io
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.data import load_dataset, sentiment_summary
from dashboard.research import ResearchDataError, read_research_source


def main():
    try:
        source = read_research_source(ROOT)
        dataset = load_dataset(io.BytesIO(source.content), source_name=source.path.name)
    except (ResearchDataError, ValueError) as exc:
        print(f"Research data setup required: {exc}", file=sys.stderr)
        print("See docs/deployment.md. No user upload can substitute for the research data.", file=sys.stderr)
        return 1
    summary = sentiment_summary(dataset.reviews)
    print(f"Approved research data ready: {source.path.name}")
    print(f"SHA-256: {source.sha256}")
    print(f"Source records: {len(dataset.records):,}; binary reviews: {summary['total_reviews']:,}; products: {summary['total_products']}")
    print(f"Positive: {summary['positive_reviews']:,}; negative: {summary['negative_reviews']:,}")
    dated = dataset.reviews.review_date.dropna()
    if len(dated):
        print(f"Observed period: {dated.min():%d %b %Y} – {dated.max():%d %b %Y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
