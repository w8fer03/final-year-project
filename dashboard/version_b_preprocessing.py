"""Exact owner-supplied Version B CLDR normalizer, never a replacement cleaner."""
from __future__ import annotations

from hashlib import sha1
import importlib.util
from pathlib import Path

from dashboard.version_b import RESOURCES, VersionBError, file_sha256, read_json


class VersionBPreprocessor:
    def __init__(self, resources: Path = RESOURCES, cldr_dir: Path | None = None):
        manifest = read_json(resources / 'CLDR_reference_manifest.json')
        for name in ('preprocessv3_cldr_ab.py', 'malay_normalization_map_v3.json', 'cldr_ms_observed_tts.tsv'):
            if file_sha256(resources / name) != manifest['sha256'][name]:
                raise VersionBError(f'Version B preprocessing resource checksum mismatch: {name}')
        # The fixed, reviewed vendored source is owner-approved, not uploaded code.
        spec = importlib.util.spec_from_file_location('sentilokal_verified_version_b_preprocessing', resources / 'preprocessv3_cldr_ab.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.emoji_package = None  # Training manifest records the regex fallback.
        self.module = module
        self.full_cldr = cldr_dir is not None and (cldr_dir / 'ms.xml').is_file() and (cldr_dir / 'ms_derived.xml').is_file()
        if self.full_cldr:
            for name, key in [('ms.xml', 'CLDR_malay_annotations_git_blob_sha1'), ('ms_derived.xml', 'CLDR_malay_derived_annotations_git_blob_sha1')]:
                content = (cldr_dir / name).read_bytes()
                digest = sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()
                if digest != manifest[key]:
                    raise VersionBError(f'Official CLDR XML differs from the recorded training pin: {name}')
        self.emoji_map = module.load_cldr_map(subset_path=resources / 'cldr_ms_observed_tts.tsv', cldr_dir=cldr_dir if self.full_cldr else None)
        norm = module.load_json_map(resources / 'malay_normalization_map_v3.json')
        # This is exactly the Version B branch of the supplied script's main().
        self.norm_map = {k: v for k, v in norm.items() if k.casefold() not in {'padu', 'mantap', 'hauk', 'baguss'}}
        self.pattern = module.compile_normalization_pattern(self.norm_map)

    def transform(self, text: str, *, require_coverage: bool = True) -> str:
        if not isinstance(text, str):
            raise VersionBError('Every inference review must be original text.')
        if require_coverage and not self.full_cldr and self.module.find_unmapped_emojis(text, self.emoji_map):
            raise VersionBError('This input contains an emoji outside the training corpus subset. Install the two pinned full Malay CLDR XML files before analyzing it.')
        return self.module.preprocess_text(text, self.emoji_map, self.norm_map, self.pattern, preserve_case=True, variant='B')
