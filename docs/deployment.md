# Provision the official research dataset

The final Version B experiment package is integrated separately from the legacy
source described below. Its two aggregate evaluation exports are included in
Git. New private Version B CSVs and model weights require owner provisioning.
See [Version B final research source](#version-b-final-research-source) for the
explicit migration; no legacy filename/checksum is silently replaced.

The current dashboard branch includes the approved cleaned CSV at
`data/dataset_preprocessed_v3_clean.csv`, committed by the project owner. There
is also an owner-added duplicate at the repository root; the app does not read
that duplicate. An ordinary clone of this branch therefore has the default input
and automatically loads it, without a dashboard upload or external download.
Neither copy is modified by the application.

A deployment that omits data files needs the approved read-only file provisioned
separately. Do not assume that ignored files from an old Codespace exist on a new
host. Do not force-add more datasets or bypass upload restrictions.

The approved processed input is `dataset_preprocessed_v3_clean.csv`. Its SHA-256 is:

```text
6164d277eece96037fb4d47972d08aed2e7b67c1eef88d8383aeaf0cc3c80ae0
```

This checksum identifies the source file; every review count, percentage, date,
and product summary is calculated from the file at runtime. The application
retains the existing 17-to-five product mapping and preprocessing exclusions.

## Local machine or GitHub Codespace

Use your original approved CSV from a private location. In Codespaces, use the
Explorer's **Upload** action to place it in `data/`; this is owner provisioning,
not an upload through the customer dashboard. This path is already tracked in the current branch; do not commit revised or additional raw data as part of deployment setup.
Alternatively, from the repository directory, copy a local original:

```bash
mkdir -p data
cp /absolute/private/path/dataset_preprocessed_v3_clean.csv data/dataset_preprocessed_v3_clean.csv
python scripts/check_research_data.py
streamlit run app.py
```

Replace `/absolute/private/path/` with the actual private file location. On
Windows, copy the file with Explorer into the repository's `data` folder, then
run the same Python check and Streamlit commands. Do not rename the labelled
alternative dataset to the cleaned dataset's name; checksum verification will
reject it.

## Read-only storage for deployment

Provision the approved CSV separately using your hosting provider's approved
private storage or persistent-volume mechanism. Mount or expose it as a local
**read-only file** in the application server, for example:

```text
/srv/sentilokal/research/dataset_preprocessed_v3_clean.csv
```

Set this environment variable in the host's application configuration, then run:

```bash
export SENTILOKAL_RESEARCH_DATA_PATH=/srv/sentilokal/research/dataset_preprocessed_v3_clean.csv
python scripts/check_research_data.py
streamlit run app.py
```

The default checksum above is verified automatically. The server must have read
permission on the file and its parent directories; the application never needs
write permission. The host must preserve or remount the file after restarts or
redeployments. Application code performs no remote downloads and requires no
storage credentials when a file is mounted. Do not put a private signed storage
URL into the path variable; it accepts a local filesystem path only.

For a container deployment, use your host's volume configuration to mount the
private directory at `/srv/sentilokal/research` with `readOnly: true` (or Docker's
`:ro` mount option), and set the path variable as above. This project does not
create a storage account, copy data to an external service, or publish a
container. The provider-specific provisioning remains the project owner's step.

If a hosting product only deploys repository files and cannot supply a private
local file, its approved data provisioning mechanism must be configured first,
or choose a host that supports a private mounted file. A code-only deployment
will show **Research dataset setup · Project owner**, rather than invented
results or a dashboard uploader that substitutes for the official data.

The previous `SHOPEE_DATA_PATH` variable is supported as a fallback alias only
when `SENTILOKAL_RESEARCH_DATA_PATH` is absent. The new variable takes precedence.
There is no directory-wide CSV picker or fallback to another dataset. An invalid
or missing configured file produces a setup state instead of silently changing
research inputs.

## Future approved processed versions

Only the project owner should change deployment configuration. After separately
approving a newer processed dataset, point `SENTILOKAL_RESEARCH_DATA_PATH` at its
read-only file and set `SENTILOKAL_RESEARCH_DATA_SHA256` to its actual SHA-256.
For example, calculate that value locally with `sha256sum /path/to/approved.csv`
or PowerShell's `Get-FileHash -Algorithm SHA256`. Then run the check command
before restarting the app. Do not change the configured checksum merely to
suppress a mismatch: first confirm the file is the intended research version.
No research-source rewriting, balancing, or training is performed. Optional
upload inference uses the exact supplied Version B text preprocessing.

## Version B final research source

The approved final source and its date metadata have different, independently
verified provenance from the legacy cleaned file:

| File | SHA-256 |
| --- | --- |
| `version_B_model_ready.csv` | `1f0803d6fcf57b05e4186fda3fd8e4c94233d39a39ad71336a95d147cb3fceb5` |
| `version_B_dashboard_metadata.csv` | `7d33b56d1fba59a6ab0a6069294b361a65923ee1b3e3b681dffb845e68e495bf` |

Provision both files in an approved private directory alongside the model
artifacts, for example `/srv/sentilokal/version_b`. Mount it read-only and set:

```bash
export SENTILOKAL_MODEL_DIR=/srv/sentilokal/version_b
python scripts/check_version_b.py --research --check-preprocessing
export SENTILOKAL_RESEARCH_PROFILE=version_b
streamlit run app.py
```

In the current prepared workspace these files are already under ignored
`models/version_b/`; omit the directory override to use them locally. Keep
`SENTILOKAL_RESEARCH_PROFILE=legacy` (the default), or unset it, to retain the
original research source. A missing/invalid Version B source shows setup errors
and never falls back silently to the legacy file. Do not set the legacy source
checksum variable to the Version B checksum.

Version B preserves the five-product mapping, original comments, and dates.
Its 18,705 unique `source_row_id` values are joined to the date metadata by key,
with identical ID sets required; source order and duplicates in comment text
are retained. All source rows were already approved upstream with 39 phase-1
exclusions applied. The dashboard performs no additional balancing/eligibility
changes. The private review-key namespace is `version_b:` and cannot match
legacy predictions accidentally.

The active research profile is shown on the dashboard. Final evaluation panels
always identify Version B independently of the current research chart source.
No whole-research predictions are generated automatically; optional saved
predictions need the stable-key/provenance contract in
[model_integration.md](model_integration.md#research-source-and-saved-predictions).

For deployment inference, also provision the saved models/vectorizer and pinned
CLDR XML. The real TF-IDF artifacts work in this workspace; the three transformer
choices remain unavailable until their original fine-tuned safetensors weights
and frozen encoder are supplied. The code-only kit's saved tokenizers are present
locally, with the verified reference settings included in code. The complete file layout, expected hashes,
missing-file list, and validation limits are in
[model_integration.md](model_integration.md#remaining-transformer-requirements).
No model download occurs when a user opens the app.

`python scripts/check_version_b.py --require-all` fails until all four choices
have their required artifacts and settings. This checks availability, not actual
four-model inference parity. It never trains, balances, or re-evaluates held-out
reviews. Git deployment alone does not carry ignored private files from this
workspace to a new host.

## Public fields and source access

Both existing tracked CSV copies contain original customer comments and internal
source/preprocessing fields. Source repository access and application access are
separate: hiding fields in Streamlit cannot remove them from a Git repository.
Keep repository and host access appropriate for the approved data. If the source
must remain private, use a private deployment repository or an approved private
read-only mount; separately review removal of tracked raw copies/history. This
change does not delete the owner's datasets or rewrite Git history.

The app reads the source without altering it. After computing eligibility and
quality counts, the cached dashboard frame keeps only original review text,
ratings, dates, product names, eligibility, and the two private keys needed for
product grouping and future prediction joins. Unrelated source and preprocessing
columns are discarded from that cache. Customer screens and exports apply a
stricter allowlist: product, original comment, stars, date, rating-derived label,
and genuine prediction/confidence fields only when available. Listing/review/shop
IDs, preprocessing diagnostics and cleaned text are not public table/export fields.
Original customer comments are intentionally displayed as required by the project;
this projection is not a claim to redact personal information typed inside a review.

For production, prefer the mounted read-only file path above. Repository checkout
files are also opened only for reading by the app, but are not automatically
protected against editing by a server administrator. Apply host read-only mount
permissions if filesystem enforcement is required.

## What is tested

Deployment tests cover three cases: the current repository layout with the actual
approved dataset, a minimal code-only deployment with explicit missing-data setup,
and a separately provisioned read-only volume with the actual approved dataset.
The ready cases load all five products and seven research charts without a CSV
uploader. The missing-data case shows no invented findings, and the readiness
command exits nonzero. Checksums and original records remain unchanged.

A deployment from the present branch normally needs no dataset copy step. A
code-only/private deployment still needs its file and environment setting. Run
`python scripts/check_research_data.py` in the actual host before starting Streamlit.
