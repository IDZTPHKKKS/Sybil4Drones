#!/usr/bin/env python3
"""Live 3D view of a mission in a multi-floor world (strong attack, 4 fake identities).

    mjpython scripts/view_floors.py multistorey 0 D0 5         # family, seed, defence, speed; whole building
    mjpython scripts/view_floors.py multistorey 0 D0 5 side    # floors side by side
"""
import os
import sys
import time

import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_floors as R  # noqa: E402


def main(family="multistorey", seed=0, defence="D0", speed=5.0, side=False):
    t_max = 1200.0 if family in R.FLOOR_H else 600.0
    s = R.build(family, seed, "sybil_strong", 4, defence, t_max)
    fl = R.Floors(s, family, stacked=not side)
    names = ("ground floor", "first floor", "second floor") if fl.n > 1 else ("cave",)
    width = fl.n * fl.L + (fl.n - 1) * fl.gap
    with mujoco.viewer.launch_passive(fl.model, fl.data, show_left_ui=False, show_right_ui=False) as v:
        if fl.stacked:
            v.cam.lookat[:] = [fl.L / 2, fl.Wd / 2, fl.H / 3]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.1 * fl.L, -30, 90
        else:
            v.cam.lookat[:] = [width / 2, fl.Wd / 2 - 3, 0]
            v.cam.distance, v.cam.elevation, v.cam.azimuth = (0.56 * width, -64, 90) if fl.n > 1 else (1.05 * fl.L, -55, 100)
        start, next_draw = time.time(), 0.0
        while v.is_running():
            target = (time.time() - start) * speed
            while s.t < target and not s.done():
                s.step()
                if s.t - s.m.t[-1] >= 1.0:
                    s.record()
            if time.time() >= next_draw:
                next_draw = time.time() + 0.5
                with v.lock():
                    v.user_scn.ngeom = 0
                    fl.draw(v.user_scn)
                    cov = fl.floor_coverage()
                    v.set_texts((None, None, f"{family}, strong attack, 4 fake identities, "
                                 f"{'no defence' if defence == 'D0' else defence}    t = {s.t:.0f} s"
                                 + ("    mission complete" if s.done() else ""),
                                 "   ".join(f"{n} {c:.0f}%" for n, c in zip(names, cov))))
            v.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "multistorey", int(a[1]) if len(a) > 1 else 0, a[2] if len(a) > 2 else "D0",
         float(a[3]) if len(a) > 3 else 5.0, len(a) > 4 and a[4] == "side")
