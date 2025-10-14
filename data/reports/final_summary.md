# Champion Pack — Summary

- #instances: 56
- Champion share by method:

  - Gamma1: 27 (48.2%)
  - SAA16-b0p3: 27 (48.2%)
  - Q120: 2 (3.6%)

## Averages by family (from champions.csv)
| family   |   distance |   vehicles |   ontime_p50 |   ontime_p95 |
|:---------|-----------:|-----------:|-------------:|-------------:|
| C        |    1083.96 |    10.2353 |      42.7499 |      43.2234 |
| R        |    1498.24 |    11.6923 |      47.1795 |      50.9038 |

## Method means from evaluation (step8_eval)
| method     |   ontime_p50 |   ontime_p95 |   ontime_mean |   tard_mean |
|:-----------|-------------:|-------------:|--------------:|------------:|
| Q120       |      46.6667 |      51.3333 |       46.1367 |     133.389 |
| Gamma1     |      46.4095 |      48.9797 |       46.3372 |     252.209 |
| Champions  |      45.8348 |      48.5723 |       45.7542 |     220.263 |
| SAA16-b0p3 |      41.9323 |      44.8019 |       41.8611 |     238.244 |

## Champion vs best baseline per instance (Δ p50 = champ - best baseline) — top 10
| instance   |   champion_p50 |   best_baseline_p50 |   delta_p50 |
|:-----------|---------------:|--------------------:|------------:|
| C101       |        24      |             24      |           0 |
| C102       |        38      |             38      |           0 |
| C103       |        53      |             53      |           0 |
| C104       |        74.7475 |             74.7475 |           0 |
| C105       |        21      |             21      |           0 |
| C106       |        23      |             23      |           0 |
| C107       |        23      |             23      |           0 |
| C108       |        26      |             26      |           0 |
| C109       |        38      |             38      |           0 |
| C201       |        29      |             29      |           0 |