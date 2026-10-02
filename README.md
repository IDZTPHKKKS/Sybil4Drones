# Sybil attacks on multi-drone 3D mapping

A team of drones explores an unknown building, tunnel network, warehouse or forest and builds a
shared 3D voxel map. One compromised drone creates fake identities that fly, scan and claim
frontiers like real drones, so the rest of the team leaves the areas they "explored" untouched.
This repository simulates the team, the attack and defences that use only the drones' own
sensors.

![Strong Sybil attack, no defence (left) vs full defence (right)](media/sybil_demo.gif)

*Office map, 8 drones, one of them compromised and running 4 fake identities at a time.
Red blocks are walls that do not exist. Left: no defence, the team stops with 60% of the office
seen. Right: full defence, 13 fake identities caught and replaced, 100% seen.
[Full video](media/sybil_demo.mp4)*

## What is simulated

**Worlds.** Four procedural families (office, warehouse, forest, tunnels; 48 × 32 m, 0.2 m
voxels, seeded).

![World families](media/worlds.png)

**Drones.** Forward depth camera (90° × 60°, 6 m, range noise and dropouts) or a spinning lidar
(360° × 59°, 30 m), plus up and down range sensors. Each drone keeps its own log-odds voxel map,
picks frontiers by size per distance, plans on a 0.4 m grid with 0.4 m clearance, follows the
path with pure pursuit and brakes for obstacles in its map.

**Team.** Map updates are gossiped over a radio link that weakens through walls, with limited
bandwidth, packet loss and retransmission. Drones take off one after another and announce the
frontier they are heading for so the others spread out.

**Attacks.**

| | |
|---|---|
| `sybil_weak` | fake identities join at launch next to the attacker, fly openly, send careless scans |
| `sybil_strong` | identities join one by one, send consistent scans, build trust by relaying real data under their names, and are replaced when revoked |
| `sybil_stealth` | `sybil_strong`, but fake identities only claim positions no honest sensor can check |
| `fakewall`, `fakefree` | a drone reports walls or free space that do not exist under its own name |
| `blackhole`, `greyhole` | a drone drops the map data it should relay |

**Defences.**

| level | adds |
|---|---|
| D0 | nothing |
| D1 | first-hand audit of received map data, Beta trust per identity, revocation and roll-back |
| D2 | only vouched identities can corroborate a claim; presence check (is there a drone where an identity says it is?) |
| D3 | trust requires being seen in person; wall claims judged separately; drones visit areas reported by identities they cannot vouch for |
| D4 | quarantine: map data and claims are used only from identities seen in person or vouched for by two drones that were |
| signed | message signatures: fake identities are impossible |

## Install

```bash
pip install -r requirements.txt
python3 -m pytest -q tests
```

## Use

Watch a mission in 3D (macOS: `mjpython`, elsewhere `python3`). Left-drag rotates, right-drag
pans, scroll zooms.

```bash
mjpython scripts/view3d.py office 0                                   # 8 honest drones
mjpython scripts/view3d.py office 0 sybil_strong --fakes 4            # attack, no defence
mjpython scripts/view3d.py office 0 sybil_strong --fakes 4 --defence D3
mjpython scripts/view3d.py tunnels 3 sybil_weak --fakes 8 --sensor lidar
```

Map families, seeds 0-9 are the experiment maps:

```bash
mjpython scripts/show_world3d.py warehouse 3      # N / P: next / previous seed, F: next family
python3 scripts/show_worlds.py 0 9                # top views -> media/worlds_0-9.png
```

Experiments and figures:

```bash
python3 scripts/calibrate_trust.py                # honest contradiction rates on held-out maps
python3 scripts/run.py sybil --seeds 0 1 2 3 4    # one JSON per run in results/, resumable
python3 scripts/run.py sybil_lidar --seeds 0 1 2 3 4
python3 scripts/figures.py                        # media/fig_*.png and .pdf
python3 scripts/render3d.py office 0 sybil_strong 4 D0 D3
```

## Layout

| path | |
|---|---|
| `swarm/worlds.py` | procedural worlds |
| `swarm/kernels.py` | ray casting, map fusion, planning grid, search, audit (Numba) |
| `swarm/sim.py` | drones, radio, coordination, defences, metrics |
| `swarm/attacks.py` | attackers |
| `swarm/scene3d.py`, `swarm/viz.py` | 3D and top-down views |
| `scripts/` | viewers, experiment runner, calibration, figures, video |
| `tests/` | unit and integration tests |

## Results

20 maps (5 per family), 8 drones with one compromised, depth camera unless noted. "Explored" is
the share of the reachable space seen by the honest drones' own sensors; "false cells" is the
median number of cells per honest map that contradict the true world. Raw runs are in
`results/`, figures are made with `scripts/figures.py`.

![Exploration over time](media/fig_coverage_time.png)

Exploration over time with 4 fake identities, median and interquartile range over maps.

![Damage vs number of fake identities](media/fig_damage.png)

Unexplored space without defence as the number of fake identities grows (mean, 95% interval).

![Defence levels](media/fig_defences.png)

Explored space and false map data for 4 fake identities at each defence level.

| attack | none | D3 | D4 |
|---|---|---|---|
| no attacker | 99.6% / 63 | 99.4% / 63 | 99.4% / 56 |
| weak, 8 identities | 67.9% / 38244 | 98.8% / 67 | 99.1% / 61 |
| strong, 8 identities | 80.7% / 24229 | 99.6% / 1183 | 99.1% / 83 |
| stealth, 8 identities | 92.0% / 21900 | 99.5% / 1602 | 99.6% / 55 |
| 2 compromised drones, 4 identities each | 76.0% / 34553 | 99.5% / 2543 | 99.5% / 2718 |

Explored / false cells. Time to 90% explored without an attacker: 135 s with no defence, 161 s
with D3, 156 s with D4.

| attack, 4 identities | camera, none | lidar, none | camera, D3 | lidar, D3 |
|---|---|---|---|---|
| weak | 77.8% / 22074 | 98.4% / 7498 | 98.7% / 58 | 99.9% / 0 |
| strong | 85.8% / 17520 | 97.7% / 9746 | 99.6% / 1848 | 99.9% / 1675 |
| stealth | 93.9% / 16536 | 99.6% / 5481 | 99.6% / 1777 | 99.9% / 2062 |

No honest drone was revoked in any run. On held-out maps the largest honest contradiction rate
is 0.04% (0.43% for wall claims) against a 2% threshold. Two colluding drones can vouch for each
other's fake identities, which D4 does not prevent. In one lidar run, free space reported by a
stealth identity led two honest drones into an obstacle below the lidar's field of view.

Runs on all 40 maps are in progress.

## License

MIT
