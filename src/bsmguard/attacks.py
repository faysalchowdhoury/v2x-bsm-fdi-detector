"""False-data injection on BSM streams (after the VeReMi misbehaviour taxonomy, plus two subtle attacks).

An attacker is an insider: a vehicle with valid credentials that transmits falsified content about
itself, so message signatures (IEEE 1609.2) verify and only the content can give it away.

==================  =====================================================================  ==========
attack              what the attacker reports                                               kinematics
==================  =====================================================================  ==========
constant_position   a fixed position (where it was at attack onset)                         inconsistent
constant_offset     true position + a fixed 25 m forward offset                             consistent after onset
random_position     a random position anywhere on the section, each message                 inconsistent
random_offset       true position + a random offset up to 10 m, each message                inconsistent
speed_high          a constant 35 m/s speed, true position                                  inconsistent
accel_spoof         random longitudinal accelerations in +/-6 m/s^2                         inconsistent
eventual_stop       frozen position and zero speed from onset (fake stalled vehicle)        consistent after onset
replay              another vehicle's messages from 5 s earlier, under its own ID           consistent after onset
drift               forward offset growing to 1 m/s (ramped over 3 s), speed raised to match **consistent**
lane_shift          lateral shift of one lane (3.7 m), ramped smoothly over 3 s             **consistent**
==================  =====================================================================  ==========

``drift`` and ``lane_shift`` are built to pass every single-vehicle physics check; only context (other
vehicles, the road) can expose them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ATTACKS = ("constant_position", "constant_offset", "random_position", "random_offset", "speed_high", "accel_spoof",
           "eventual_stop", "replay", "drift", "lane_shift")


def inject(df: pd.DataFrame, *, t_from: float, share: float = 0.1, onset_s: float = 3.0, seed: int = 0,
           min_duration_s: float = 10.0) -> pd.DataFrame:
    """Turn ``share`` of the vehicles that appear after ``t_from`` into attackers, attack types assigned
    round robin, each attack starting ``onset_s`` after the vehicle's first message."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    span = out.groupby("vehicle_id")["t"].agg(["min", "max"])
    cands = span[(span["min"] >= t_from) & (span["max"] - span["min"] >= min_duration_s + onset_s)].index.to_numpy()
    chosen = rng.choice(cands, size=max(1, int(share * len(cands))), replace=False)
    y_lo, y_hi = out["y"].quantile([0.01, 0.99])
    x_lo, x_hi = out["x"].quantile([0.01, 0.99])
    by_t = out.set_index(["vehicle_id", "t"])
    for k, vid in enumerate(chosen):
        kind = ATTACKS[k % len(ATTACKS)]
        idx = out.index[(out["vehicle_id"] == vid) & (out["t"] >= span.loc[vid, "min"] + onset_s)]
        if len(idx) == 0:
            continue
        rows = out.loc[idx]
        n = len(idx)
        tau = (rows["t"] - rows["t"].iloc[0]).to_numpy()
        if kind == "constant_position":
            out.loc[idx, ["x", "y"]] = rows[["x", "y"]].iloc[0].to_numpy()
        elif kind == "constant_offset":
            out.loc[idx, "y"] = rows["y"] + 25.0
        elif kind == "random_position":
            out.loc[idx, "x"] = rng.uniform(x_lo, x_hi, n)
            out.loc[idx, "y"] = rng.uniform(y_lo, y_hi, n)
        elif kind == "random_offset":
            out.loc[idx, "x"] = rows["x"] + rng.uniform(-10, 10, n)
            out.loc[idx, "y"] = rows["y"] + rng.uniform(-10, 10, n)
        elif kind == "speed_high":
            out.loc[idx, "speed"] = 35.0
        elif kind == "accel_spoof":
            out.loc[idx, "accel"] = rng.uniform(-6, 6, n)
        elif kind == "eventual_stop":
            out.loc[idx, ["x", "y"]] = rows[["x", "y"]].iloc[0].to_numpy()
            out.loc[idx, ["speed", "accel"]] = 0.0
        elif kind == "replay":
            others = [v for v in out.loc[(out["t"] >= rows["t"].iloc[0] - 5.0) & (out["t"] <= rows["t"].iloc[0] - 4.9), "vehicle_id"].unique()
                      if v != vid]
            if not others:
                continue
            src = rng.choice(others)
            for c in ("x", "y", "speed", "accel", "angle"):
                vals = [by_t[c].get((src, round(t - 5.0, 1)), np.nan) for t in rows["t"]]
                out.loc[idx, c] = vals
            out.loc[idx] = out.loc[idx].ffill().bfill()
        elif kind == "drift":
            # the extra speed ramps in smoothly over 3 s (an abrupt +1 m/s step would itself be an
            # acceleration spike, which a careful attacker avoids); position integrates the same offset
            r = np.clip(tau / 3.0, 0, 1)
            dv = 1.0 * (3 * r**2 - 2 * r**3)
            dacc = 1.0 * (6 * r - 6 * r**2) / 3.0 * (tau < 3.0)
            out.loc[idx, "y"] = rows["y"] + np.r_[0.0, np.cumsum(0.5 * (dv[1:] + dv[:-1]) * np.diff(tau))]
            out.loc[idx, "speed"] = rows["speed"] + dv
            out.loc[idx, "accel"] = rows["accel"] + dacc
        elif kind == "lane_shift":
            ramp = np.clip(tau / 3.0, 0, 1)
            shift = 3.7 * (3 * ramp**2 - 2 * ramp**3)  # smooth step
            out.loc[idx, "x"] = rows["x"] + shift
            dshift = 3.7 * (6 * ramp - 6 * ramp**2) / 3.0 * (tau < 3.0)
            out.loc[idx, "angle"] = np.arctan2(rows["speed"] * np.sin(rows["angle"]) + dshift, rows["speed"] * np.cos(rows["angle"]))
        out.loc[idx, "attack"] = kind
    return out
