# Dataset catalogue

The [TSML community on Zenodo](https://zenodo.org/communities/tsml/records) hosts individual datasets and bulk archives. Cite the version you use, since a later version can change dataset files or metadata.

| File | Contents |
| --- | --- |
| [Current versions](zenodo_records/current-versions) | Record titles and current Zenodo record IDs |
| [Previous versions](zenodo_records/previous-versions) | Record titles and superseded record IDs |
| [Manifest](manifest/) | JSON Lines inventories of records, versions, and attached files, plus the snapshot history |
| [Record lists](record_lists/) | Groups of records by archive or task |

## Refresh the catalogue

Install the development dependencies from the repository root.

```sh
python -m pip install -r requirements-dev.txt
```

`python -m python_utilities.generate_zenodo_snapshots` retrieves public version and file metadata. Run `python -m python_utilities.generate_catalog` after refreshing the snapshot to rebuild the record catalogue while retaining reviewed record kinds.

Maintainers can refresh the human-readable version lists with `python -m python_utilities.generate_zenodo_record_lists`. This command can load a local credential. See [the repository’s Zenodo instructions](AGENTS.md) before running it.

Generate statistics for an individual dataset from a directory containing its local TRAIN and TEST `.ts` files.

```sh
python -m python_utilities.generate_dataset_catalog Adiac path/to/data
```

The generator writes `manifest/datasets.jsonl`. It also picks up complete `_eq` and `_nmv` pairs for the same dataset. Use `--replace-existing` when supplying the complete set of variants for an existing dataset.

```sh
python -m pytest python_utilities/tests
```
