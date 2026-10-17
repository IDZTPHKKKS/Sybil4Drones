# Crazyflie physics flight

In `swarm/` each drone is a kinematic point that gets the velocity it asks for every 0.2 s step,
within 1.5 m/s and 3 m/s², with a level camera. Here each drone is a Crazyflie 2 rigid body
flown by a geometric controller at 500 Hz, without changes to `swarm/`. Guidance, planning,
mapping, radio, attacks and defences are the same.

| | kinematic | physics |
|---|---|---|
| motion | velocity set directly | thrust and body moments, the drone tilts to accelerate |
| model | point | 27 g, Menagerie `cf2` inertia, thrust up to 0.35 N, moments up to 4e-3 N m, 20 ms motor lag, rotor drag |
| sensors | level camera, yaw only | camera and up/down rangers fixed to the body |
| collisions | commanded position | physical position |

Each step the drone's guidance asks for a velocity change. This becomes a reference that starts
at the physical state and has constant acceleration over the step, and the controller tracks
it. `physics(cls)` gives any drone class, honest or attacker, physics flight.

| file | |
|---|---|
| `dynamics.py` | rigid-body dynamics, semi-implicit Euler at 2 ms |
| `controller.py` | geometric controller from `wind_drone_mujoco`, with feed-forward |
| `drone.py` | `physics(cls)` |
| `run_physics.py` | static experiments with physics flight, paired with the kinematic runs of the same maps |
| `test_crazyflie.py` | free fall, hover, step response, line tracking, yaw, 60 s mission |

```bash
python3 -m pytest -q crazyflie
python3 crazyflie/run_physics.py --check      # kinematic rerun matches the stored run
python3 crazyflie/run_physics.py              # 3 maps per family, 5 conditions -> results/physics/
python3 crazyflie/run_physics.py --summary
```

Conditions: no attacker with no defence and with D4, strong attacker with 4 fake identities
with no defence, D4 and D5. The kinematic runs of the same maps are in `results/sybil` and
`results/bind`. A physics mission takes about 2.5 min on one core.

The state is known exactly, collisions are checked once per step, and there is no downwash or
wind.

## Results

12 maps (seeds 0-2 of each family), kinematic / physics:

| condition | explored % | false cells (median) | crashed drones | honest drones expelled |
|---|---|---|---|---|
| no attacker, no defence | 99.7 / 100.0 | 63 / 67 | 0 / 1 | 0 / 0 |
| no attacker, D4 | 99.7 / 99.7 | 59 / 61 | 0 / 4 | 0 / 0 |
| strong, no defence | 80.6 / 93.1 | 16549 / 21691 | 0 / 1 | 0 / 0 |
| strong, D4 | 99.6 / 100.0 | 63 / 65 | 0 / 1 | 0 / 0 |
| strong, D5 | 99.7 / 100.0 | 62 / 60 | 0 / 1 | 0 / 0 |

Median tracking error 0.7 cm per step, at most 9.9 cm; maximum tilt 34 degrees. With D4 and D5
the results match the kinematic runs. Without a defence the damage varies a lot from map to map
in both cases (office 1: 30% explored kinematic, 94% physics; office 0: 72% and 60%).
8 drones crashed in 6 of the 60 physics missions, 4 of them in one tunnels mission, after
overshooting by 6-8 cm in narrow passages; the guidance brakes for a drone that reacts at once.
