# LoveDA 事后确认性扩展报告

本扩展在原始三种子结果之后提出，但在读取 seeds 53/67 的正式结果之前冻结。
它对 FFFF、GSHPS-Auto 与 RAND-PATH-K 对称地增加相同两个种子，每个运行 5000 updates。
研究目的仅是判断原始 LoveDA 高方差是否持续，不用于挽救 GSHPS，也不改写原始预注册。

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
