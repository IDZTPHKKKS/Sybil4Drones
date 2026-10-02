# Results summary

Means over maps (10 seeds x 4 families unless fewer have finished). 'Space seen' counts only cells observed with its own camera by a drone that reports truthfully and that most honest drones still trust at that time.

## Sybil attack (camera, 8 real drones, 1 compromised)

'Seen' = space the honest drones observed with their own sensors. 'Fake identities caught' counts every identity used, including replacements.

| condition | maps | seen at end | min | time to 90% | wrong cells / map | fake identities caught | first caught after | honest drones revoked | crashes |
|---|---|---|---|---|---|---|---|---|---|
| none-D0 | 20 | 99.6% | 95.9% | 135 s | 73 | 0.0 | - | 0 | 0 |
| none-D3 | 20 | 99.4% | 95.5% | 161 s | 77 | 0.0 | - | 0 | 0 |
| sybil_weak1-D0 | 20 | 89.9% | 30.2% | 144 s (14/20) | 32081 | 0.0 | - | 0 | 0 |
| sybil_weak2-D0 | 20 | 84.1% | 22.8% | 150 s (10/20) | 36834 | 0.0 | - | 0 | 0 |
| sybil_weak4-D0 | 20 | 77.8% | 9.6% | 154 s (7/20) | 43526 | 0.0 | - | 0 | 0 |
| sybil_weak8-D0 | 20 | 67.9% | 12.5% | 128 s (2/20) | 50135 | 0.0 | - | 0 | 0 |
| sybil_strong1-D0 | 20 | 96.6% | 80.0% | 145 s (18/20) | 15842 | 0.0 | - | 0 | 0 |
| sybil_strong2-D0 | 20 | 96.2% | 87.8% | 161 s (19/20) | 21615 | 0.0 | - | 0 | 0 |
| sybil_strong4-D0 | 20 | 85.8% | 26.1% | 162 s (13/20) | 34565 | 0.0 | - | 0 | 0 |
| sybil_strong8-D0 | 20 | 80.7% | 26.9% | 141 s (11/20) | 45511 | 0.0 | - | 0 | 0 |
| sybil_stealth1-D0 | 20 | 98.1% | 92.5% | 148 s | 16007 | 0.0 | - | 0 | 0 |
| sybil_stealth2-D0 | 20 | 95.2% | 74.5% | 140 s (17/20) | 27307 | 0.0 | - | 0 | 0 |
| sybil_stealth4-D0 | 20 | 93.9% | 43.8% | 149 s (18/20) | 39437 | 0.0 | - | 0 | 0 |
| sybil_stealth8-D0 | 20 | 92.0% | 48.9% | 134 s (15/20) | 46891 | 0.0 | - | 0 | 0 |
| sybil_weak4-D1 | 20 | 99.2% | 94.7% | 167 s | 263 | 4.0 | 10 s | 0 | 0 |
| sybil_weak4-D2 | 20 | 99.0% | 90.7% | 169 s | 1281 | 4.0 | 10 s | 0 | 0 |
| sybil_weak4-D3 | 20 | 98.7% | 92.1% | 174 s | 94 | 4.0 | 9 s | 0 | 0 |
| sybil_weak8-D3 | 20 | 98.8% | 86.7% | 192 s (19/20) | 360 | 8.0 | 9 s | 0 | 0 |
| sybil_strong4-D1 | 20 | 99.5% | 95.5% | 173 s | 17485 | 7.0 | 13 s | 0 | 0 |
| sybil_strong4-D2 | 20 | 99.8% | 95.6% | 192 s | 15723 | 7.2 | 12 s | 0 | 0 |
| sybil_strong4-D3 | 20 | 99.6% | 95.6% | 210 s | 10490 | 9.2 | 11 s | 0 | 0 |
| sybil_strong8-D3 | 20 | 99.6% | 96.0% | 229 s | 12219 | 13.8 | 11 s | 0 | 0 |
| sybil_stealth4-D1 | 20 | 99.6% | 96.1% | 180 s | 16081 | 6.5 | 14 s | 0 | 0 |
| sybil_stealth4-D2 | 20 | 99.4% | 93.2% | 189 s | 14875 | 6.5 | 13 s | 0 | 0 |
| sybil_stealth4-D3 | 20 | 99.6% | 96.0% | 224 s | 9787 | 8.9 | 12 s | 0 | 0 |
| sybil_stealth8-D3 | 20 | 99.5% | 96.0% | 220 s | 11618 | 14.1 | 12 s | 0 | 0 |
| 2x_sybil_strong4-D0 | 20 | 76.0% | 22.7% | 155 s (12/20) | 52430 | 0.0 | - | 0 | 0 |
| 2x_sybil_strong4-D3 | 20 | 99.5% | 95.8% | 240 s | 16243 | 17.8 | 11 s | 0 | 0 |

## Camera vs LiDAR (4 fake identities)

| condition | camera: seen | LiDAR: seen | camera: wrong cells | LiDAR: wrong cells |
|---|---|---|---|---|
| none-D0 | 99.6% | 99.9% | 73 | 0 |
| none-D3 | 99.4% | 99.8% | 77 | 0 |
| sybil_weak4-D0 | 77.8% | 98.4% | 43526 | 59504 |
| sybil_weak4-D3 | 98.7% | 99.9% | 94 | 0 |
| sybil_strong4-D0 | 85.8% | 97.7% | 34565 | 43012 |
| sybil_strong4-D3 | 99.6% | 99.9% | 10490 | 8478 |
| sybil_stealth4-D0 | 93.9% | 99.6% | 39437 | 30246 |
| sybil_stealth4-D3 | 99.6% | 99.9% | 9787 | 15034 |

