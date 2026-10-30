# Provision the official research dataset

The Git branch contains application code, not private review CSVs. A new clone,
Codespace, or deployment **does not** inherit files from this Codex workspace.
Do not force-add datasets, encode them into source code, or bypass upload controls.

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
not an upload through the customer dashboard. The file stays ignored by Git.
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
No preprocessing, balancing, or training is performed by this application.

## What has been verified

Deployment tests start the app in an isolated code-only directory without data.
They verify the explicit missing-setup state and the check command's nonzero exit.
They then provision a separate read-only temporary volume containing the actual
approved CSV and verify that the app automatically loads its real data, all five
products, and all seven charts with no uploader on the research dashboard.

This confirms the application's read-only file integration. It does **not** mean
that an arbitrary new remote host has been provisioned: that host still needs
the private file and environment setting described above.
