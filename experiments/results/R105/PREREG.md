# PREREG (R105 lock)

Created: 2026-10-21 13:14:30 CST
Code hash (sha256 over 17 files): `127d438f03feaf0e543b42f67427b803403bb7ba651ef97456db89b4abed2fa7`

## Locked files
- `kincost/__init__.py` e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- `kincost/benchmark.py` 6f02abfb14b2cbbe96c009f1f3163b93def76f7e0d2a1dfb504b14d1e47de4b5
- `kincost/calibrate.py` 1cb9f05d1eafe80abc7be369aa6dff5fc00e714ce9bb20d06cc2f62e87cb00f5
- `kincost/chain.py` c92232b1df6e3ef4a402b1f2dbc1df5cd914699b99cf8baebc5e016f0dcb304f
- `kincost/detector.py` c48bc296df42795111c9839535437f82458d87c5e56ce8bcd5f9f0915cbe40d0
- `kincost/endpoint.py` 5762052d0c98bf7a5ba05268d222db58821e65647760b47ac4893b433faeecf4
- `kincost/fastfit.py` 4b296c18569574e0c7984799ab62f14a519cffddffa26f20e43cd42729e2db36
- `kincost/holds.py` 246eeca823aec931a931ed4a60fcada08dd27bf1bc681853f80df59b9b0fe168
- `kincost/io_data.py` da830c02a1f32c8cec31b50394856b2c274575bb312e5930481fa270ca2b7a0a
- `kincost/pipeline.py` 016270ae421e313526344c3834030b9c89f6891adc61e9f59aeabde932032884
- `kincost/runs/common.py` 60f5104cfd70711e21b96bd32c68335a9da007a26c3e0604c1e5da3edf8dc63d
- `kincost/runs/r006_ladder.py` 9e5502de2103ae9c60fb964bb87a6cd8c0186dbe874f96c0bf7ed87811e34d0a
- `kincost/runs/r102_benchmark.py` b3c4281efbb2974663c592d31438a36dbface4d2b00967108e85a3fda185e660
- `kincost/runs/r104_ladder_real.py` 741b4003384136bb54846e3b3904b60e83194df0ca67be88ccfb980951dfd009
- `kincost/runs/r105_lock.py` a4dbf76f4551e60f9f007146eae22193474426195c35cd73ab15c14caca13cf6
- `kincost/runs/r201_confirm.py` 288e8bdd89a076bc5ce5298eb72c89bd0803b55701d541be8c475d188a6f9dad
- `kincost/synth.py` ca258947ff6d4b748b05b09995323c978264b24bc4a2937e67e82821a49888d1

## Calibration outcome (D split)
- |A| = 0: (empty)
- window 512.0 ms; E_set 108.0 ms; knots (deg) [5.0, 8.379, 11.523, 15.0]
- x_ref [2.302585092994046, 0.3031321016740752, 0.009498041726378645, 0.06196598985931836, 0.01589554969713393, 0.16569718721559426, -0.0042332786073733955]

## Split
- GBVR D 50 / C 357; GB D 100 / C 222; seed 20261020

## Constants
```json
{
 "R_confirm": 20,
 "B_confirm": 2000,
 "stream_confirm": 5000,
 "seed_band_confirm": 11000,
 "R_cal": 10,
 "B_cal": 2000,
 "seed_r203": 12000,
 "n_low_lambda_r203": 10,
 "margin": [
  -0.05129329438755058,
  0.04879016416943205
 ],
 "margin_descriptive": [
  -0.08338160893905101,
  0.0769610411361284
 ],
 "band_level": 0.9,
 "P_elig_min": 0.6,
 "n812_min": 500.0,
 "cond_max": 100000000.0,
 "min_win": 10,
 "k_win": 40,
 "primary_lat": [
  80.0,
  500.0
 ],
 "primary_amp": [
  5.0,
  15.0
 ],
 "det_min_dur_ms": 12.0,
 "grid_dt": 4.0,
 "seed_base": 820261018,
 "calib_seed": 2026,
 "calib_R": 10,
 "calib_B": 2000
}
```

## Benchmark configuration
```json
{
 "truths": [
  "light",
  "medium",
  "heavy",
  "b2",
  "vel_adaptive",
  "kalman_rts"
 ],
 "reps_per_truth": 26,
 "B_adm": 500,
 "B_band": 1000,
 "R_bm": 4,
 "oracle": true,
 "K_oracle": 6,
 "alpha_mc": 0.05,
 "n812_scale": 0.11363636363636363,
 "streams": {
  "pseudo": 1000,
  "oracle": 2000,
  "eval": 3000
 },
 "r103": {
  "truths": [
   "medium"
  ],
  "reps": 20,
  "R_bm": 20,
  "oracle": false,
  "compare": "R102"
 }
}
```

## Ladder configuration (R104)
```json
{
 "R": 4,
 "R_delta": 4,
 "B": 2000,
 "stream_delta": 6000,
 "stream_ladder": 60,
 "amp_bin": [
  8.0,
  12.0
 ],
 "selection": "member of A whose finite D-only delta is closest to the median finite delta (ties -> lower mid); non-finite deltas excluded and listed; none finite -> rung 4 unavailable; A empty -> fixed placeholder B1_bw4_30_post_a1 (flagged)"
}
```

## Confirmatory analyses
- **R201**: C-GBVR native (left eye, R = 1) vs C-GB (session 1) under every member of A at its D eta, R = 20 geometries (stream 5000), shared participant bootstrap B = 2000, max-T 90 % bands, decision rule EQUIVALENCE / MISMATCH / INCONCLUSIVE(tag) on [log .95, log 1.05], support thresholds unscaled; m*, +-8 % descriptive, native GB delta; A empty -> CALIBRATION FAILURE, no C endpoint test
- **R202**: same, 8-12 deg events only, design [log A, 4 harmonics], x_ref = [log 10, harmonic means]
- **R203**: Lambda / c recomputed on C-split holds for A u the 10 lowest D Lambda/c members, B = 2000, R_cal = 10 (diagnostic only)
- **R302**: diagnostics: GBVR right eye; GB session 2 (same members, eta, seeds; the session-2 analysis uses the locked C-GB participants that have a session-2 recording, missing ones listed)

## Deviations
- benchmark (R102/R103) and ladder (R104) run after the lock, with their code and configuration frozen in this lock (R102 / R103 / R104 verify the lock and read their settings from it); no benchmark output feeds R201-R203
- benchmark replicates per truth set by the 250 CPU-h gate (plan: 100); the CI half-width <= 0.1 criterion is then not guaranteed and the achieved precision is reported
- R004 did not persist the calibration GBVR participant IDs; the GBVR D cohort is the deterministic split (seed, salt 1) of the same zip, whose size and member names are recorded here and checked by every post-lock run

Per-member eta, Lambda and c, and the data manifest are in `lock.json`.
