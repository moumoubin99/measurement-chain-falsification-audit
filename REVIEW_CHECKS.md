# Review-export checks

The review export was checked without accessing sample-level gaze data, contacting participants, downloading models/environments or rerunning a research experiment.

| Check | Result | Evidence |
| --- | --- | --- |
| Original scientific payload | 265 retained source artifacts copied byte for byte | `provenance/source_files.json` |
| R105 scientific code lock | All 17 per-file hashes and the combined hash match the original lock | `provenance/original_locked_code.json`; `tools/verify_release.py` |
| Complete result schema | All 207 result JSON files, the plotting-data JSON and both NPZ archives inventoried | `provenance/data_inventory.json` |
| Public-data boundary | No raw sample-level gaze, demographic table, private credential/path, operational log, private chat or historical Git bundle included; public upstream pseudonyms retained | `provenance/privacy_audit.json`; `provenance/export_changes.json` |
| Import and syntax smoke checks | 11 core package modules and 2 lock/path helpers import; all Python files parse; the family enumerates 276 members | `provenance/import_validation.json` |
| Derived-data reconstruction | Six data figures and eight tables regenerated; plotting-data values match at tolerance 1e-12; all eight tables are byte-identical to retained tables | `provenance/rebuild_validation.json` |
| Preview metadata | All seven PNG previews open correctly and have no personal author/EXIF metadata | `provenance/import_validation.json` |
| File inventory | SHA256 and byte counts recorded, with a separate checksum for the manifest | `MANIFEST.json`; `MANIFEST.sha256` |

The source field/array review and pattern scan are useful checks, not a guarantee of irreversible anonymisation. The retained pseudonyms and summary features can be linked to the same public source data. Hosting under a normal GitHub account is not author-blind review.

The numerical and rendering checks above test reconstruction from retained derived results. They do not verify the original raw-to-result analyses, independently reproduce the benchmark, validate every scientific interpretation, or imply submission/acceptance. Historical analysis/run programs are not imported as modules during the smoke check because some perform file I/O at import time. Their original source is preserved and syntax-checked.

See `REPRODUCIBILITY.md` for the separate raw-data and intermediate-context requirements, and `experiments/results/README_DEPOSIT.md` for the internal-code-lock terminology and retained exploratory/provisional flags.
