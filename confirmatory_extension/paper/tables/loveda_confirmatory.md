# LoveDA post-hoc confirmatory extension

This extension was frozen after the original n=3 analysis and before seeds 53/67 were run.

| Cohort | Method | n | mIoU mean | sample SD | min | max |
|---|---|---:|---:|---:|---:|---:|
| original_n3 | FFFF | 3 | 45.9950 | 0.2943 | 45.6554 | 46.1718 |
| original_n3 | GSHPS_AUTO | 3 | 45.2208 | 2.2058 | 43.9442 | 47.7677 |
| original_n3 | RAND_PATH | 3 | 44.9344 | 0.5250 | 44.4869 | 45.5123 |
| confirmatory_n2 | FFFF | 2 | 46.3704 | 2.3200 | 44.7299 | 48.0108 |
| confirmatory_n2 | GSHPS_AUTO | 2 | 45.3686 | 0.3224 | 45.1406 | 45.5965 |
| confirmatory_n2 | RAND_PATH | 2 | 46.3637 | 1.4211 | 45.3588 | 47.3685 |
| combined_n5 | FFFF | 5 | 46.1452 | 1.1963 | 44.7299 | 48.0108 |
| combined_n5 | GSHPS_AUTO | 5 | 45.2799 | 1.5701 | 43.9442 | 47.7677 |
| combined_n5 | RAND_PATH | 5 | 45.5061 | 1.1205 | 44.4869 | 47.3685 |

Combined n=5 GSHPS-Auto/FFFF SD ratio: **1.3125**.
Combined n=5 GSHPS-Auto/RAND-PATH-K SD ratio: **1.4012**.

This is a descriptive stability diagnosis, not an experiment designed to rescue GSHPS.
