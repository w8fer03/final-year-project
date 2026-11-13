"""Explicit Version B research migration using verified source IDs, never text/position."""
from __future__ import annotations

import pandas as pd

from dashboard.data import Dataset
from dashboard.products import PRODUCT_NAMES
from dashboard.version_b import RESOURCES, VersionBError, deployment_config, file_sha256, local_file, model_directory, read_json


def load_version_b_research(directory=None, resources=RESOURCES) -> Dataset:
    root = model_directory() if directory is None else directory
    manifest = read_json(resources / 'research_manifest.json')
    config = deployment_config(resources)
    source = local_file(root, manifest['dataset_file'])
    metadata = local_file(root, manifest['metadata_file'])
    if manifest['dataset_sha256'] != config['training_source_sha256'] or file_sha256(source) != manifest['dataset_sha256'] or file_sha256(metadata) != manifest['metadata_sha256']:
        raise VersionBError('Version B research source or date metadata does not match its approved provenance.')
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        dates = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise VersionBError('Version B research files cannot be read.') from exc
    if not {'source_row_id','stars','sentiment','comment','cleaned_comment','item_id'}.issubset(frame) or not {'source_row_id','datetime'}.issubset(dates):
        raise VersionBError('Version B model-ready data and date metadata schemas are incomplete.')
    for table in (frame, dates):
        if table.source_row_id.eq('').any() or table.source_row_id.duplicated().any():
            raise VersionBError('Version B source_row_id values must be nonempty and unique.')
    if len(frame) != manifest['expected_rows'] or set(frame.source_row_id) != set(dates.source_row_id):
        raise VersionBError('Version B data and metadata must describe exactly the same approved source IDs.')
    joined = frame.merge(dates[['source_row_id','datetime']], on='source_row_id', how='left', validate='one_to_one', sort=False)
    if joined.source_row_id.tolist() != frame.source_row_id.tolist():
        raise VersionBError('A metadata merge changed the approved source order.')
    ratings = pd.to_numeric(joined.stars, errors='coerce')
    expected = ratings.map({1:'0',2:'0',4:'1',5:'1'})
    if expected.isna().any() or not joined.sentiment.eq(expected).all():
        raise VersionBError('Version B approved labels disagree with eligible star ratings.')
    product_names = joined.item_id.map(PRODUCT_NAMES)
    if product_names.isna().any():
        raise VersionBError('Version B data contains product identifiers outside the approved Baseus mapping.')
    records = pd.DataFrame({'review_id': manifest['review_id_namespace'] + joined.source_row_id,
                            'product_id': joined.item_id, 'product_name': product_names,
                            'review_text': joined.comment, 'rating': ratings,
                            'review_date': pd.to_datetime(joined.datetime, format='mixed', errors='coerce', utc=True).dt.tz_convert(None),
                            'sentiment': joined.sentiment.map({'0':'Negative','1':'Positive'}), 'eligible': True})
    quality = {'total_records':len(records),'binary_records':len(records),'missing_dates':int(joined.datetime.eq('').sum()),
               'invalid_dates':int((joined.datetime.ne('') & records.review_date.isna()).sum()),
               'empty_reviews':int(records.review_text.str.strip().eq('').sum()),'duplicate_rows':int(frame.duplicated().sum())}
    return Dataset('Version B CLDR · approved model-ready source', records, records.copy(), quality)


def load_version_b_predictions(directory=None, resources=RESOURCES):
    """Optional saved whole-research inference outputs; no inference/evaluation run here."""
    root = model_directory() if directory is None else directory
    csv_path, manifest_path = root/'research_predictions.csv', root/'research_predictions_manifest.json'
    if not csv_path.exists() and not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    config = deployment_config(resources)
    if manifest.get('training_source_sha256') != config['training_source_sha256'] or manifest.get('predictions_sha256') != file_sha256(csv_path):
        raise VersionBError('Saved research predictions are not bound to the approved Version B source.')
    architecture = manifest.get('architecture')
    if architecture not in config['models'] or manifest.get('negative_threshold') != config['models'][architecture]['threshold']:
        raise VersionBError('Saved research predictions require the approved architecture and full-precision negative threshold.')
    data = load_version_b_research(root, resources)
    try:
        frame = pd.read_csv(csv_path, dtype={'source_row_id':str}, keep_default_na=False)
    except (OSError, ValueError, UnicodeError) as exc:
        raise VersionBError('Saved Version B research predictions cannot be read.') from exc
    if not {'source_row_id','predicted_sentiment'}.issubset(frame) or frame.source_row_id.eq('').any() or frame.source_row_id.duplicated().any():
        raise VersionBError('Saved Version B predictions require unique source_row_id and predicted_sentiment columns.')
    frame['review_id'] = 'version_b:' + frame.source_row_id
    if not set(frame.review_id).issubset(set(data.reviews.review_id)):
        raise VersionBError('Saved prediction IDs are not in the approved Version B research source.')
    # Reuse the existing strict label/provenance join validation. Scores are not confidence.
    from dashboard.artifacts import attach_predictions
    return attach_predictions(data.reviews, frame[['review_id','predicted_sentiment']])[['review_id','predicted_sentiment']].dropna(subset=['predicted_sentiment'])
