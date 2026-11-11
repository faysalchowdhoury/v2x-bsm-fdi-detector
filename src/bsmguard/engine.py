"""Streaming feature engine: one call per batch of BSMs received in the same 100-ms slot.

Per vehicle it keeps the previous message and short histories; per message it returns

instantaneous features (physics residuals; 0 for a vehicle's first message)
    ``r_disp``    |distance moved - mean reported speed x dt|               (m)
    ``r_acc``     |speed change / dt - mean reported acceleration|          (m/s^2)
    ``jerk``      |acceleration change / dt|                                (m/s^3)
    ``r_lat``     displacement perpendicular to the reported heading        (m)
    ``off_road``  distance outside the road's lateral extent                 (m)
    ``overlap``   deepest footprint overlap with any other vehicle           (m)

window features (last ``window`` messages), the input of the learned models
    max of each residual, mean overlap, number of distinct vehicles overlapped in the last
    ``partner_window`` messages, reported speed and |acceleration|.

Context features need every vehicle's latest message, which a roadside unit has. Overlap is symmetric
(a ghost and the real car it sits on both overlap), so the number of *distinct* partners is what
separates a ghost moving through traffic from its victims, who each meet it once.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

INSTANT = ("r_disp", "r_acc", "jerk", "r_lat", "off_road", "overlap")
WINDOW = ("max_r_disp", "max_r_acc", "max_jerk", "max_r_lat", "max_off_road", "mean_overlap", "partners", "speed", "abs_accel")


@dataclass
class _Veh:
    t: float
    x: float
    y: float
    v: float
    a: float
    ang: float
    hist: deque = field(default_factory=lambda: deque(maxlen=10))
    partners: deque = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class Engine:
    road_x: tuple[float, float] = (-1.0, 25.0)
    window: int = 10
    partner_window: int = 30
    stale_s: float = 2.0
    state: dict[int, _Veh] = field(default_factory=dict)

    def process(self, t: float, ids: np.ndarray, x: np.ndarray, y: np.ndarray, v: np.ndarray, a: np.ndarray,
                ang: np.ndarray, length: np.ndarray, width: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """All arrays describe messages stamped ``t``. Returns (instant (n, 6), window (n, 9))."""
        n = len(ids)
        inst = np.zeros((n, len(INSTANT)))
        # --- per-vehicle physics residuals against the previous message
        prev = [self.state.get(int(i)) for i in ids]
        has = np.array([p is not None and 0 < t - p.t <= self.stale_s for p in prev])
        if has.any():
            k = np.flatnonzero(has)
            pt = np.array([prev[j].t for j in k])
            px = np.array([prev[j].x for j in k])
            py = np.array([prev[j].y for j in k])
            pv = np.array([prev[j].v for j in k])
            pa = np.array([prev[j].a for j in k])
            pang = np.array([prev[j].ang for j in k])
            dt = t - pt
            dx, dy = x[k] - px, y[k] - py
            dpos = np.hypot(dx, dy)
            inst[k, 0] = np.abs(dpos - 0.5 * (v[k] + pv) * dt)
            inst[k, 1] = np.abs((v[k] - pv) / dt - 0.5 * (a[k] + pa))
            inst[k, 2] = np.abs((a[k] - pa) / dt)
            mid = 0.5 * (ang[k] + pang)
            inst[k, 3] = np.abs(-dx * np.cos(mid) + dy * np.sin(mid))
        lo, hi = self.road_x
        inst[:, 4] = np.maximum(0.0, np.maximum(lo - x, x - hi))
        # --- cross-vehicle overlap (footprints: width across, length along; y is the front bumper)
        partners: list[set[int]] = [set() for _ in range(n)]
        if n > 1:
            cy = y - 0.5 * length
            tree = cKDTree(np.c_[x, cy])
            pairs = tree.query_pairs(r=12.0, output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                ox = 0.5 * (width[i] + width[j]) - np.abs(x[i] - x[j])
                oy = 0.5 * (length[i] + length[j]) - np.abs(cy[i] - cy[j])
                depth = np.where((ox > 0) & (oy > 0), np.minimum(ox, oy), 0.0)
                np.maximum.at(inst[:, 5], i, depth)
                np.maximum.at(inst[:, 5], j, depth)
                for a_, b_, d in zip(i[depth > 0.3], j[depth > 0.3], depth[depth > 0.3]):
                    partners[a_].add(int(ids[b_]))
                    partners[b_].add(int(ids[a_]))
        # --- update state and build window features
        win = np.zeros((n, len(WINDOW)))
        for m in range(n):
            vid = int(ids[m])
            s = self.state.get(vid)
            if s is None or not has[m]:
                s = _Veh(t, x[m], y[m], v[m], a[m], ang[m])
                self.state[vid] = s
            s.t, s.x, s.y, s.v, s.a, s.ang = t, x[m], y[m], v[m], a[m], ang[m]
            s.hist.append(inst[m])
            s.partners.append(partners[m])
            h = np.asarray(s.hist)
            win[m, :5] = h[:, :5].max(axis=0)
            win[m, 5] = h[:, 5].mean()
            win[m, 6] = len(set().union(*s.partners))
            win[m, 7] = v[m]
            win[m, 8] = abs(a[m])
        return inst, win

    def evict(self, t: float) -> int:
        """Drop vehicles not heard from for ``stale_s``; returns how many were dropped."""
        old = [k for k, s in self.state.items() if t - s.t > self.stale_s]
        for k in old:
            del self.state[k]
        return len(old)
