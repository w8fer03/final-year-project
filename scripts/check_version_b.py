"""Read-only package checks and optional new-text smoke inference; never evaluate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.final_evaluations import load_final_evaluations
from dashboard.prediction_backend import get_predictor, readiness
from dashboard.version_b import ARCHITECTURES, VersionBError, model_directory
from dashboard.version_b_preprocessing import VersionBPreprocessor
from dashboard.version_b_research import load_version_b_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', action='store_true', help='Verify the explicitly provisioned Version B research source and metadata')
    parser.add_argument('--check-preprocessing', action='store_true', help='Compare original comments to saved cleaned text, without classifier evaluation')
    parser.add_argument('--smoke-tfidf', action='store_true', help='Load actual TF-IDF artifacts and classify three new sample comments')
    parser.add_argument('--require-all', action='store_true', help='Exit nonzero if any of the four deployment architectures is unavailable')
    args = parser.parse_args()
    try:
        report = load_final_evaluations()
        if report is None:
            raise VersionBError('The two final aggregate evaluation exports are missing.')
        print(f'Recorded evaluations: {len(report.rows)} configurations; 80:20 and 70:30 remain separate.')
        available = []
        for key, label in ARCHITECTURES.items():
            ready, message = readiness(key)
            available.append(ready)
            print(f'{label}: {"READY TO LOAD" if ready else "UNAVAILABLE"} — {message}')
        if args.research or args.check_preprocessing:
            research = load_version_b_research()
            print(f'Version B source verified: {len(research.reviews):,} reviews, {research.reviews.product_name.nunique()} products; original comments and keyed dates preserved.')
        if args.check_preprocessing:
            import pandas as pd
            from dashboard.version_b import RESOURCES, read_json
            manifest = read_json(RESOURCES/'research_manifest.json')
            frame = pd.read_csv(model_directory()/manifest['dataset_file'], keep_default_na=False)
            preprocessor = VersionBPreprocessor(cldr_dir=model_directory()/'cldr')
            differences = int(frame.comment.map(preprocessor.transform).ne(frame.cleaned_comment).sum())
            print(f'Preprocessing parity: {differences} differences in {len(frame):,} saved cleaned comments. No model/test-set evaluation was run.')
            if differences:
                return 1
        if args.smoke_tfidf:
            samples = ['Barang bagus!', 'Sound rosak, x boleh guna.', 'Barang bagus!']
            result = get_predictor('tfidf_xgb').predict(samples)
            print(f'Actual saved TF-IDF new-text smoke labels: {result.labels.tolist()} (0=Negative, 1=Positive). Threshold: {result.threshold!r}.')
            print('This is an inference smoke check, not an accuracy or held-out evaluation.')
        return 1 if args.require_all and not all(available) else 0
    except (VersionBError, OSError, ValueError) as exc:
        print(f'Version B setup required: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
