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
