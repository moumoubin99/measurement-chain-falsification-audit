# Measurement-chain falsification audit

Review data and analysis-code companion for **A declared measurement-chain family fails a falsifiable fixation test for comparing saccade velocity across desktop and virtual-reality eye trackers**.

Prepared review export: 2026-10-23. Repository: https://github.com/moumoubin99/measurement-chain-falsification-audit .

This repository contains retained derived results, the original locked analysis source, the original R105 lock, and the generators for the data figures and tables. It does not contain participant demographics, sample-level gaze recordings, raw archives, personal chats, private operational logs, historical Git bundles, trained models or executable environments. Preparing or publishing this deposit does not establish journal acceptance or reproduce the original experiments.

## Start here

1. Read [DATA_DICTIONARY.md](DATA_DICTIONARY.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the original [deposit reading notes](experiments/results/README_DEPOSIT.md).
2. Verify file integrity and the 17-file scientific code lock:

   ```sh
   python tools/verify_release.py
   ```

3. With NumPy, SciPy, pandas and Matplotlib available, regenerate the retained data figures and tables from derived results, without downloading data or running an experiment:

   ```sh
   python tools/rebuild_figures.py --output-dir review_outputs
   ```

The conceptual Figure 1 is an exported SVG/PNG, rather than a computed statistical plot. Its source is preserved. The reconstruction command regenerates Figures 2–5, two supplementary data figures, eight tables and the plotting-data JSON.

## Repository contents

| Location | Contents |
| --- | --- |
| `experiments/kincost/` | Analysis package and scientific run scripts; 17 locked source files retain their exact original hashes |
| `experiments/results/` | 207 result JSON files, two numeric NPZ bootstrap archives, the R105 lock and reading notes |
| `paper/figures/` | Plot/table generators, plotting-data JSON, PNG previews, conceptual SVG and table/caption snippets |
| `provenance/` | Source-to-export hashes, original lock metadata, export policy, field inventory and review checks |
| `tools/` | Manifest, privacy/inventory audit, integrity verification and derived-output reconstruction tools |

All original result JSON, NPZ, retained code and plot inputs are copied without numeric edits. Synthetic smoke results and exploratory/provisional metadata remain explicitly labelled in their original files; they must not be substituted for the real-data results. `README_DEPOSIT.md` explains superseded diagnostics, the internal code lock and compute-accounting distinctions.

## Participant identifiers and privacy scope

The retained `pid`, split lists, benchmark quarters and data-member names use the source datasets' **already public pseudonymous identifiers**. They are kept unchanged so the frozen development/confirmatory partition, paired observations and random-stream structure remain interpretable. There is no new identity mapping, demographic table or contact information in this export. Pseudonyms and aggregate features may still be linkable to the upstream public data; this repository does **not** claim irreversible anonymisation or author-blind hosting.

## Original data and licences

Obtain raw data independently from the official, versioned [GazeBase v3](https://doi.org/10.6084/m9.figshare.12912257.v3) and [GazeBaseVR v2](https://doi.org/10.6084/m9.figshare.21308391.v2) releases. Both official versions state CC BY 4.0. Their creators' attribution and licence notices are retained in [DATA_SOURCES.md](DATA_SOURCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No new open-source licence is granted for the original project code or documentation; this is a public review deposit under the default rights described in [LICENSE](LICENSE). Upstream materials retain their existing terms, described in [LICENSE-DATA.md](LICENSE-DATA.md). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what the deposit can rebuild and what additionally requires raw data and intermediate calibration context.
