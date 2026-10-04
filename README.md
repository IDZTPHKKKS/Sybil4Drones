# Sybil attacks on multi-drone 3D mapping of indoor and confined spaces

A team of drones explores an unknown building, tunnel network, warehouse or forest and builds a
shared 3D voxel map. One compromised drone creates fake identities that fly, scan and claim
frontiers like real drones, so the rest of the team leaves the areas they "explored" untouched.
This repository simulates the team, the attack and defences that use only the drones' own
sensors.

![Strong Sybil attack, no defence (left) vs full defence (right)](media/sybil_demo.gif)

*Office map, 8 drones, one of them compromised and running 4 fake identities at a time.
Red blocks are walls that do not exist. Left: no defence, the team stops with 60% of the office
seen. Right: full defence, fake identities are revoked and the attacker replaces them 9 times, 100% seen.
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

Experiments:

```bash
python3 scripts/calibrate_trust.py                          # honest contradiction rates on held-out maps
python3 scripts/run.py sybil --seeds 0 1 2 3 4 5 6 7 8 9    # one JSON per run in results/, resumable
python3 scripts/render3d.py office 0 sybil_strong 4 D0 D3
```

## Extending

Watch any setup in action:

```bash
mjpython scripts/view3d.py warehouse 2 sybil_strong --drones 12 --fakes 6 --defence D4 --speed 3
```

`--drones` is the team size, `--fakes` the number of fake identities active at a time,
`--defence` one of D0-D4, `--sensor` camera or lidar. The last drone of the team is the
compromised one.

For batches, add an experiment to `conditions()` in `scripts/run.py`. Each condition is
`(name, parameter overrides, attack)`, where the attack is `(name, compromised drones, settings)`
or `None`:

```python
if exp == "team":
    out = []
    for n in (4, 8, 12):
        out.append((f"n{n}-none-D0", dict(n_drones=n), None))
        for d in ("D0", "D4"):
            out.append((f"n{n}-strong4-{d}", dict(DEFENCES[d], n_drones=n),
                        ("sybil_strong", 1, dict(n_sybil=4))))
    out.append(("2att-strong6-D4", dict(D4), ("sybil_strong", 2, dict(n_sybil=6, spawn_every=2.0))))
    return out
```

```bash
python3 scripts/run.py team --families office warehouse --seeds 0 1 2
```

Results go to `results/team/`, one JSON per run, with the time series of every metric.

Team and environment (`Params` in `swarm/sim.py`):

| parameter | default | |
|---|---|---|
| `n_drones` | 8 | team size |
| `sensor` | `"camera"` | `"camera"` or `"lidar"` |
| `max_range`, `hfov`, `vfov` | 6 m, 90°, 60° | depth camera |
| `lidar_range` | 30 m | lidar |
| `v_max` | 1.5 m/s | flight speed |
| `comm_range` | 25 m | radio range |
| `loss` | 0 | packet loss rate |
| `t_max` | 600 s | mission length (set in `scripts/run.py`) |
| `claim_radius` | 6 m | frontier claim radius |

Attack settings (class attributes in `swarm/attacks.py`, passed as the third element of the
attack):

| setting | default | |
|---|---|---|
| `n_sybil` | 4 | fake identities active at a time |
| `speed` | 1.2 m/s | fake identity speed |
| `first_spawn` | 8 s | first fake identity (strong, stealth) |
| `spawn_every` | 4 s | gap between new identities (strong, stealth) |
| `relabel` | 0.8 | share of relayed real data sent under fake names (strong, stealth) |

Defence switches are in the same `Params` (`audit_vouched`, `presence`, `trust_needs_body`,
`trust_by_type`, `verify_free`, `quarantine`, `attest_quorum`); D0-D4 in `scripts/run.py` are
combinations of them. A new attack is a subclass of `_Sybil` (or `Attacker`) in
`swarm/attacks.py` added to `ATTACKS`; for a Sybil variant, `_lengths` sets what each fake scan
reports. A new world family is a function in `swarm/worlds.py` added to `FAMILIES` and `make`.

## Layout

| path | |
|---|---|
| `swarm/worlds.py` | procedural worlds |
| `swarm/kernels.py` | ray casting, map fusion, planning grid, search, audit (Numba) |
| `swarm/sim.py` | drones, radio, coordination, defences, metrics |
| `swarm/attacks.py` | attackers |
| `swarm/scene3d.py`, `swarm/viz.py` | 3D and top-down views |
| `scripts/` | viewers, experiment runner, calibration, video |
| `tests/` | unit and integration tests |

## Results

40 maps (10 per family), 8 drones with one compromised, depth camera. "Explored" is
the share of the reachable space seen by the honest drones' own sensors (mean over maps); "false
cells" is the median number of cells per honest map that contradict the true world. Raw runs are
in `results/`.

Explored space without any defence, by number of fake identities:

| attack | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| weak | 92.7% | 87.0% | 81.3% | 75.1% |
| strong | 97.2% | 97.2% | 90.3% | 86.1% |
| stealth | 98.1% | 96.6% | 95.2% | 91.4% |

Without an attacker the team explores 99.8%.

Explored / false cells by defence level:

| attack | none | D3 | D4 |
|---|---|---|---|
| no attacker | 99.8% / 62 | 99.7% / 63 | 99.7% / 58 |
| weak, 8 identities | 75.1% / 30828 | 99.4% / 68 | 99.5% / 58 |
| strong, 8 identities | 86.1% / 26661 | 99.8% / 1618 | 99.5% / 64 |
| stealth, 8 identities | 91.4% / 24058 | 99.7% / 1373 | 99.8% / 61 |
| 2 compromised drones, 4 identities each | 76.3% / 35486 | 99.7% / 1831 | 99.7% / 1386 |

Time to 90% explored without an attacker: 134 s with no defence, 153 s with D3, 155 s with D4.

4 fake identities, explored / false cells by defence level:

| attack | none | D1 | D2 | D3 | D4 |
|---|---|---|---|---|---|
| weak | 81.3% / 24208 | 99.4% / 60 | 99.2% / 57 | 99.3% / 58 | 99.8% / 56 |
| strong | 90.3% / 18726 | 99.7% / 4776 | 99.9% / 3465 | 99.7% / 1816 | 99.7% / 61 |
| stealth | 95.2% / 16234 | 99.8% / 3002 | 99.7% / 3686 | 99.8% / 1143 | 99.8% / 63 |

Across all 1,440 runs, no honest drone was revoked. On the held-out maps, honest contradiction rates remained very low: at most 0.04% (0.43% for wall claims alone), against a 2% threshold. Note that D4 cannot stop two colluding drones from vouching for each other’s fake identities for now.

## Contributing

Contributions are welcome. Please open an issue to report a bug or propose a new attack, defence
or world, and submit changes as a pull request. If you use this code in your research or work, please
cite this repository.

## License

MIT
