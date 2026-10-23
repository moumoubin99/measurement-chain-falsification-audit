# Reproducibility boundaries

## Level 1: verify and reconstruct the retained derived outputs

`python tools/verify_release.py` checks the release manifest's file sizes/SHA256 values and recomputes every per-file and combined hash recorded by `experiments/results/R105/lock.json`. This is a file-integrity and code-lock check, not an experiment result.

`python tools/rebuild_figures.py --output-dir review_outputs` creates an isolated temporary work tree, runs only the original derived-data/plot/table generators, and copies their products to the chosen output directory. It neither reads raw data nor calls the original scientific run scripts. The command compares reconstructed plotting-data JSON against the retained snapshot and checks all eight generated tables; six data plots are rendered. Raster/font/PDF byte identity can depend on the plotting stack and is not promised. The original conceptual Figure 1 SVG/PNG is copied as an existing illustration.

Requirements: Python 3.10+ with NumPy, SciPy, pandas and Matplotlib. `requirements.txt` declares package names rather than an unverified lock of the historical server environment. Numba is optional for expensive analysis; the original code has a pure-Python fallback. No environment, model, downloaded wheel or interpreter is bundled.

## Level 2: inspect/reanalyse the retained derived statistics

The 207 result JSON files and the two numeric NPZ archives support auditing the reported summary values, chain comparisons, benchmark replicates, confidence/coverage summaries and post hoc bootstrap diagnostics. Original public pseudonymous IDs, D/C membership, benchmark quarters, pairing and random-stream constants are retained. `provenance/data_inventory.json` describes all fields and arrays.

This deposit does not provide every intermediate participant/realisation cache. Reconstructing figures from retained results is not an independent reproduction of calibration, detection, benchmark generation or confirmatory processing.

## Level 3: recompute analyses from upstream raw recordings

This requires separately obtaining and verifying both official archives in `DATA_SOURCES.md`, and sufficient CPU time. Do not substitute the synthetic/provisional GB smoke path for real GazeBase data. No such recomputation was performed while preparing this public export.

The original code resolves paths relative to this repository, with overrides `GBVR_ZIP`, `GB_ROOT` and `KINCOST_RESULTS`. Its default raw paths are `data/collected/gazebasevr/gazebasevr.zip` and `data/collected/gazebase/`. GazeBase is a nested ZIP: unpack the outer archive and then the subject ZIPs to a separate CSV tree before using `GB_ROOT`, or provide a directory of subject ZIPs for the general loader. The historical R105 file-size checks expect an extracted CSV tree, so use extracted recordings when verifying the locked manifest. GBVR is read from its original ZIP. Never place raw data in Git history.

The baseline is round 1/session 1 RAN; right-eye and GB session-2 diagnostics use their named branches. Preserve the recorded seed, sorted public-pseudonym order and D/C partition: GB D=100/C=222, GBVR D=50/C=357. The 276-member family, detector/hold constants, benchmark settings and stop/decision rules are recorded in `R105/lock.json`.

The original expensive post-lock scripts require a frozen `R105/ctx.pkl` calibration context, checked against the lock. That private intermediate cache is **not** distributed here. A raw-data rerun therefore must regenerate the real-D calibration context, verify its scientific fields against the retained lock, and place the verified context at that expected path in a separate working results directory before invoking post-lock runs. Merely downloading raw ZIPs is not enough to reproduce all historical commands. Raw-to-context reconstruction and numerical agreement across software versions have not been validated for this export.

The source `R105/PREREG.md` uses historical terminology for an internal code lock, not a public registry preregistration. The scientific source files are exported unchanged. The original combined SHA256 is `127d438f03feaf0e543b42f67427b803403bb7ba651ef97456db89b4abed2fa7`; the historical commit is recorded separately. The public repository starts new history and does not contain the original Git bundle. Do not re-run the lock-writing command over the retained historical lock; a new run needs its own provenance and results directory.

## Historical reading notes and excluded operational records

The byte-identical `README_DEPOSIT.md` refers to original run logs and internal code-review records used by the authors. Those operational records and private conversations are not part of this public deposit. Their exclusion does not alter any retained number, code hash or lock constant. See `provenance/export_changes.json` for the allowlist and exclusions.
