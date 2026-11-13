"""Read the eight supplied held-out summaries. Never run test-set inference or tune.

Arithmetic checks compare aggregate exports for consistency; displayed metrics
are the original CSV values. PR-AUC is read, not reconstructed from a matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import pandas as pd

from dashboard.version_b import ARCHITECTURES, EVALUATIONS, VersionBError

METRICS = {
    'macro_f1': 'Macro F1', 'negative_f1': 'Negative F1', 'accuracy': 'Accuracy',
    'negative_precision': 'Negative precision', 'negative_recall': 'Negative recall',
    'positive_f1': 'Positive F1', 'weighted_f1': 'Weighted F1', 'negative_PR_AUC': 'Negative PR-AUC',
}
CM_COLUMNS = ['actual_negative_pred_negative', 'actual_negative_pred_positive', 'actual_positive_pred_negative', 'actual_positive_pred_positive']


@dataclass(frozen=True)
class FinalEvaluationReport:
    rows: pd.DataFrame
    metrics_bytes: bytes
    confusion_bytes: bytes

    def group(self, split: str) -> pd.DataFrame:
        if split not in ('80_20', '70_30'):
            raise VersionBError('Choose one of the two recorded test splits.')
        return self.rows[self.rows.split.eq(split)].copy()


def load_final_evaluations(directory: Path = EVALUATIONS) -> FinalEvaluationReport | None:
    metrics_path = directory / 'day3_all_models_test_metrics.csv'
    confusion_path = directory / 'day3_all_models_confusion.csv'
    if not metrics_path.exists() and not confusion_path.exists():
        return None
    try:
        metrics = pd.read_csv(metrics_path)
        confusion = pd.read_csv(confusion_path)
        metrics_bytes, confusion_bytes = metrics_path.read_bytes(), confusion_path.read_bytes()
    except (OSError, ValueError) as exc:
        raise VersionBError('Both final held-out CSV exports must be installed and readable.') from exc
    keys = ['architecture', 'split']
    required = keys + ['method', 'threshold', 'test_count', 'actual_negative', 'predicted_negative'] + list(METRICS)
    if not set(required).issubset(metrics) or not set(keys + CM_COLUMNS).issubset(confusion):
        raise VersionBError('The final result CSV schema does not match the supplied eight-configuration export.')
    expected = {(name, split) for name in ARCHITECTURES for split in ('80_20', '70_30')}
    if len(metrics) != 8 or len(confusion) != 8 or metrics.duplicated(keys).any() or confusion.duplicated(keys).any():
        raise VersionBError('Final results require exactly eight unique architecture/split configurations.')
    if set(map(tuple, metrics[keys].to_numpy())) != expected or set(map(tuple, confusion[keys].to_numpy())) != expected:
        raise VersionBError('Final result architecture/split identities are incomplete or unsupported.')
    rows = metrics.merge(confusion, on=keys, how='inner', validate='one_to_one', sort=False)
    numeric = list(METRICS) + ['threshold', 'test_count', 'actual_negative', 'predicted_negative'] + CM_COLUMNS
    for name in numeric:
        rows[name] = pd.to_numeric(rows[name], errors='coerce')
        if rows[name].isna().any() or not rows[name].map(math.isfinite).all():
            raise VersionBError(f'Recorded result values are not finite: {name}')
    for name in list(METRICS) + ['threshold']:
        if not rows[name].between(0, 1).all():
            raise VersionBError(f'Recorded result is outside [0, 1]: {name}')
    for name in ['test_count', 'actual_negative', 'predicted_negative'] + CM_COLUMNS:
        if not rows[name].ge(0).all() or not rows[name].mod(1).eq(0).all():
            raise VersionBError('Recorded test supports and matrix counts must be nonnegative integers.')
        rows[name] = rows[name].astype(int)
    matrices = []
    for row in rows.itertuples(index=False):
        a, b, c, d = (getattr(row, name) for name in CM_COLUMNS)
        n = a + b + c + d
        if n <= 0 or row.test_count != n or row.actual_negative != a + b or row.predicted_negative != a + c:
            raise VersionBError('Recorded support counts disagree with the confusion export.')
        ratio = lambda x, y: x / y if y else 0.0
        neg_f1, pos_f1 = ratio(2*a, 2*a+b+c), ratio(2*d, 2*d+b+c)
        checks = {'accuracy': (a+d)/n, 'negative_precision': ratio(a,a+c), 'negative_recall': ratio(a,a+b),
                  'negative_f1': neg_f1, 'positive_f1': pos_f1, 'macro_f1': (neg_f1+pos_f1)/2,
                  'weighted_f1': ((a+b)*neg_f1+(c+d)*pos_f1)/n}
        if any(not math.isclose(getattr(row, key), value, abs_tol=1e-10) for key, value in checks.items()):
            raise VersionBError('Recorded metrics disagree with the supplied aggregate confusion counts.')
        matrices.append([[a,b],[c,d]])
    for split in ('80_20', '70_30'):
        group = rows[rows.split.eq(split)]
        if group.test_count.nunique() != 1 or group.actual_negative.nunique() != 1:
            raise VersionBError('Architectures in a split have inconsistent test supports and cannot share a comparison group.')
    rows['confusion_matrix'] = matrices
    return FinalEvaluationReport(rows, metrics_bytes, confusion_bytes)
