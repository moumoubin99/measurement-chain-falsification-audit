# Results deposit: reading notes

These notes go with the result files cited in the manuscript (`R0xx/`, `R1xx/`, `R2xx/`, `R302_*/`, `A006/`). They explain names, flags and timestamps a reader might otherwise misread. No numbers have been changed.

## 1. "prereg" names an internal code lock

A number of artefacts contain "prereg", "pre-registered" or "preregistration":

- the file `R105/PREREG.md`;
- the git tag `prereg-R105` and the commit message "R105 preregistration lock" inside `R105/R105.bundle`;
- JSON notes in `R004/failure_profile.json`, `R004/gap_pv_diagnostics.json`, `A006/gap_pv.json` and `A006/d_adm.json`;
- docstrings in the analysis scripts and in the hash-locked modules (`fastfit.py`, `endpoint.py`, `r201_confirm.py`, `r105_lock.py`, `lock_git.sh`).

Every one of these refers to the **internal R105 code lock**. That lock was made on 2026-10-21 at 13:14 CST, in commit `d194a794`. It froze the code, the constants, the D/C participant split, the calibration context and the decision rule, all before any confirmatory saccade analysis. No public registry preregistration exists. The manuscript calls the procedure "prespecified, code-locked".

The locked files were left unedited, because editing them would invalidate the hashes recorded in `R105/lock.json`. That file, not `PREREG.md`, is the reference for the lock.

## 2. Superseded exploratory profile: `R004/failure_profile.json`

Two fields in this file are superseded or exploratory.

- **`clean_window_profile.ratio_min` (1.135)** compares a 10-coordinate statistic against the 12-coordinate critical value *c*. It is superseded by the recalibrated A006 value of 1.23 (`A006/d_adm.json`, field `summary.clean10`), which uses its own critical value, *c10*.
- **`exploratory_drop_tests`** are reduced coordinate sets compared against the full 12-coordinate *c*, without recalibration. They are therefore not valid admissibility tests. Dropping `gap_frac_median` together with `bp_4_10` leaves 46 members below *c*. Supplementary Note S5 reports this as a post hoc descriptive.

There is also a field `z_median_over_members`. It is the median of the *signed* z. The "median |z|" shown in Fig. S1b comes from `A006/d_adm.json`, field `summary.z_median_abs`.

## 3. Provenance flags and timestamps

- **`R001/summary.json`** has `_provisional: true`. It is the GazeBaseVR audit summary from the first pass (2026-10-20). Its values match `logs/r001.log` verbatim. The flag is metadata from the first pass and does not mean the values are wrong.
- **`R001/gb_summary.json`** (2026-10-22 21:51) was reconstructed from `logs/R001_gb.log`; all fields are taken verbatim from that log.
- **`R005/existence.json`** has a modification time of 2026-10-23 17:28, although the run itself was on 2026-10-20. The file was re-emitted later from the same run. Its values agree with `logs/r005.log`, for example "PV10 gap 0.14492697". In this file, PV is relative to the analytic latent PV.
- **`R004_synthetic_provisional/`** is a synthetic smoke-test calibration. No manuscript number comes from it.
- **`R104/ladder.json`** has `_provisional: true` by design. The flag is set to `placeholder` (`r104_ladder_real.py`, l.184), and the calibrated set was empty (`rung4_basis`: "A empty: fixed placeholder"). So rung 4 used the fixed placeholder chain `B1_bw4_30_post_a1`, as the prespecified rule requires and as the manuscript states (Methods, R104 ladder). The values themselves are final.
- **`R003/identity.json`** has `_provisional: true` because the identity check runs on synthetic GazeBase-like input with real D-GazeBaseVR donors (field `note`). The call in `r002_r003_chain.py`, l.99, does not override the default flag for synthetic input. The reported worst relative error, 1.38e-13, is final.

## 4. Compute accounting

The manuscript gives two compute figures.

- **About 476 CPU-hours** is the sum of the run-level `cpu_h` fields:
  - `R004/profile.json`, field `calib_cpu_h`: 27.4;
  - `R102/summary.json`: 386.2;
  - `R103/summary.json`: 58.9;
  - R104, R201–R203 and R302 together: about 4.
- **About 245 CPU-hours** is the pre-run profiled cost of the same work.

The two differ because the shared server was oversubscribed during the first part of R102. The CPU quota was 16, but 120 workers were running. A throttle added at 09:44 CST on 2026-10-22 resolved this. Results are unaffected, since every replicate has a fixed seed.

## 5. Code review records

The pre-lock code review ran for four rounds, with Codex as an independent language-model reviewer. All four are recorded in `refine-logs/CODE_REVIEW.md`. The round 3 and round 4 bundles and responses are archived in `refine-logs/code_review_rounds_3_4/`.
