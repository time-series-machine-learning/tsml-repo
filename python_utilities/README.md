# Python Utilities

Command-line tools for maintaining the [TSML Zenodo community](https://zenodo.org/communities/tsml/records) inventories, dataset statistics catalogue, and record metadata.

## Safety & Credentials

See [`AGENTS.md`](../AGENTS.md) for repository safety boundaries:

- **Read-only by default**: Inventory refreshers, live audits, and default previews make only unauthenticated `GET` requests to public Zenodo endpoints and never load `.zenodo_token`.
- **Credential isolation**: Production mutations (`--apply`) are human-run only and read `.zenodo_token` after an interactive `Type APPLY to continue:` prompt. Sandbox runs (`--sandbox --apply`) read `.zenodo_sandbox_token` and target `https://sandbox.zenodo.org/api`. Cross-wiring tokens and hosts is rejected automatically, and authenticated openers block HTTP redirects.

## Available Utilities

| Module | Purpose |
| --- | --- |
| [`refresh_zenodo_inventories.py`](refresh_zenodo_inventories.py) | Single-pass anonymous refresh of `zenodo_records/*`, `manifest/zenodo_versions.jsonl`, `manifest/zenodo_files.jsonl`, `manifest/snapshot.jsonl`, and `manifest/records.jsonl`, with a follow-up action checklist. |
| [`generate_dataset_catalog.py`](generate_dataset_catalog.py) | Verify local `.ts` splits (or `--archive` `.zip` releases) against `manifest/zenodo_files.jsonl` sizes/checksums and compute `manifest/datasets.jsonl` statistics. |
| [`audit_zenodo_records.py`](audit_zenodo_records.py) | Read-only live audit comparing public Zenodo versions, files, licenses, creators, and keywords against repository manifests and `record_lists/*`. Also runs weekly via [`.github/workflows/zenodo-audit.yml`](../.github/workflows/zenodo-audit.yml). |
| [`sync_zenodo_tags.py`](sync_zenodo_tags.py) | Compute expected dataset keywords from `manifest/records.jsonl` + `manifest/datasets.jsonl`, preview diffs read-only, and publish deltas with `--apply`. |
| [`add_zenodo_tags.py`](add_zenodo_tags.py) / [`remove_zenodo_tags.py`](remove_zenodo_tags.py) | Preview or apply targeted keyword additions or removals for a record list or `--record` targets. |
| [`replace_zenodo_creators.py`](replace_zenodo_creators.py) | Preview or replace the `creators` list on selected Zenodo records. |
| [`update_zenodo_descriptions.py`](update_zenodo_descriptions.py) | Preview or update record `description` HTML (`--description-file`, `--replace`, `--append-html`) and/or `related_identifiers` (`--add-related`, `--remove-related`). |
| [`generate_zenodo_record_lists.py`](generate_zenodo_record_lists.py) / [`generate_zenodo_snapshots.py`](generate_zenodo_snapshots.py) / [`generate_catalog.py`](generate_catalog.py) | Lower-level building blocks used by `refresh_zenodo_inventories.py` (can also be run individually). |

## Common Maintainer Workflows

Run commands from the repository root using the development environment (`pip install -r requirements-dev.txt`).

### 1. Audit live Zenodo against the repository

```sh
python -m python_utilities.audit_zenodo_records
# Optional flags for upcoming metadata tracks:
python -m python_utilities.audit_zenodo_records --check-descriptions --check-length-tags --check-missing-tags
```

### 2. Refresh the repository after publishing a new dataset or version

1. Add any newly published record title to the appropriate file(s) in [`record_lists/`](../record_lists/README.md).
2. Refresh all Zenodo lists, snapshots, and `manifest/records.jsonl` in one pass:

   ```sh
   python -m python_utilities.refresh_zenodo_inventories
   ```

   *(Use `--dry-run --yes` first if you want to preview changes without writing local files.)*
3. If a brand-new record was added, set its `"kind"` (`"dataset"` or `"aggregate"`) in `manifest/records.jsonl`.
4. Compute or refresh dataset statistics from local `.ts` files that match the published MD5 checksums:

   ```sh
   python -m python_utilities.generate_dataset_catalog <DatasetName> <local_data_dir> --replace-existing
   ```

   *(For ZIP-only dataset records such as `S2Agri-17`, also pass `--archive <path/to/published.zip>`.)*
5. Run the offline schema and consistency suite:

   ```sh
   python -m pytest -v
   ```

### 3. Preview and apply Zenodo metadata updates (human-run)

Every maintenance CLI runs a read-only public preview by default. Append `--apply` (and type `APPLY` at the prompt) only after reviewing the diff:

```sh
# Synchronize standard tags across all dataset records (preview, then apply)
python -m python_utilities.sync_zenodo_tags
python -m python_utilities.sync_zenodo_tags --apply

# Include base series-length and missing-value tags when ready
python -m python_utilities.sync_zenodo_tags --include-length-tags --include-missing-tags

# Targeted tag addition or removal
python -m python_utilities.add_zenodo_tags --record Sleep --tag univariate
python -m python_utilities.remove_zenodo_tags --record BenzeneConcentration_disc --tag multiverse-core

# Fix text in a record description or link a benchmark paper DOI
python -m python_utilities.update_zenodo_descriptions \
    --record 18300287 \
    --replace "processed to equal length" "truncated to unequal lengths"
python -m python_utilities.update_zenodo_descriptions \
    record_lists/rehab \
    --add-related 10.48550/arXiv.2507.21018 isSupplementTo
```

To test mutations against `sandbox.zenodo.org` using `.zenodo_sandbox_token`, pass `--sandbox`. If a previous interrupted run left an unedited draft session open (`state='inprogress'`), pass `--discard-existing-edit` alongside `--apply`.
