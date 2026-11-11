from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bsmguard.api.app import create_app
from bsmguard.attacks import inject
from bsmguard.data import LocalFrame
from bsmguard.detector import Detector, ewma_by_vehicle, load, save
from bsmguard.engine import Engine
from bsmguard.j2735 import NBYTES, SPEC, TOTAL_BITS, BSMCore, J2735Error
from bsmguard.study import calibrate, featurize


def test_uper_round_trip_and_ranges() -> None:
    assert TOTAL_BITS == 290 and NBYTES == 37
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = BSMCore(*[int(rng.integers(lo, hi + 1)) for _, lo, hi in SPEC])
        assert BSMCore.decode(m.encode()) == m
    lo_vals = [lo for _, lo, _ in SPEC]
    assert BSMCore.decode(BSMCore(*lo_vals).encode()) == BSMCore(*lo_vals)
    bad = BSMCore(*[*lo_vals[:10], 9000, *lo_vals[11:]])  # speed above 8191
    with pytest.raises(J2735Error):
        bad.encode()
    with pytest.raises(J2735Error):
        BSMCore.decode(b"\x00" * 36)
    m = BSMCore.from_si(msg_cnt=130, temp_id=7, t_s=61.25, lat_deg=37.84, lon_deg=-122.3, speed_mps=24.13, heading_deg=-20.0,
                        accel_mps2=-1.234, width_m=1.9, length_m=4.6)
    assert m.msg_cnt == 2 and m.sec_mark == 1250 and m.speed_mps == pytest.approx(24.13, abs=0.011)
    assert m.heading_deg == pytest.approx(340.0) and m.accel_mps2 == pytest.approx(-1.23)


def test_local_frame_round_trip() -> None:
    f = LocalFrame()
    x, y = np.array([0.0, 3.7, 20.0]), np.array([0.0, 150.0, 480.0])
    lat, lon = f.to_geo(x, y)
    x2, y2 = f.from_geo(np.round(lat * 1e7) / 1e7, np.round(lon * 1e7) / 1e7)  # J2735 resolution
    assert np.allclose(x, x2, atol=0.02) and np.allclose(y, y2, atol=0.02)
    assert f.heading_from_road(np.array([0.0]))[0] == pytest.approx(f.bearing_deg)


def _straight(n: int = 50, v: float = 20.0) -> pd.DataFrame:
    t = np.round(np.arange(n) * 0.1, 1)
    rows = [{"vehicle_id": 1, "t": tt, "x": 5.0, "y": 10 + v * tt, "speed": v, "accel": 0.0, "angle": 0.0, "length": 4.5,
             "width": 1.8, "attack": "none"} for tt in t]
    rows += [{"vehicle_id": 2, "t": tt, "x": 5.0, "y": 40 + v * tt, "speed": v, "accel": 0.0, "angle": 0.0, "length": 4.5,
              "width": 1.8, "attack": "none"} for tt in t]
    return pd.DataFrame(rows).sort_values(["t", "vehicle_id"]).reset_index(drop=True)


def test_engine_residuals_and_overlap() -> None:
    df = _straight()
    inst, win = featurize(df, (-1.0, 25.0))
    assert np.abs(inst[:, :5]).max() < 1e-6 and inst[:, 5].max() == 0.0  # consistent, on road, 30 m apart
    eng = Engine()
    eng.process(0.0, np.array([1]), np.array([5.0]), np.array([10.0]), np.array([20.0]), np.zeros(1), np.zeros(1),
                np.array([4.5]), np.array([1.8]))
    i, _ = eng.process(0.1, np.array([1]), np.array([5.0]), np.array([40.0]), np.array([20.0]), np.zeros(1), np.zeros(1),
                       np.array([4.5]), np.array([1.8]))
    assert i[0, 0] == pytest.approx(28.0)  # teleported 30 m where 2 m was possible
    i, w = eng.process(0.2, np.array([1, 2]), np.array([5.0, 5.3]), np.array([42.0, 43.0]), np.full(2, 20.0), np.zeros(2),
                       np.zeros(2), np.full(2, 4.5), np.full(2, 1.8))
    assert (i[:, 5] > 1.0).all() and (w[:, 6] == 1).all()  # both overlap, one partner each


def test_attacks_label_and_drift_is_kinematically_consistent() -> None:
    t = np.round(np.arange(200) * 0.1, 1)
    df = pd.concat([pd.DataFrame({"vehicle_id": vid, "t": t + 100, "x": 5.0 + 3.7 * k, "y": 10 + 15 * t + 30 * k, "speed": 15.0,
                                  "accel": 0.0, "angle": 0.0, "length": 4.5, "width": 1.8, "attack": "none"})
                    for k, vid in enumerate(range(1, 21))]).sort_values(["t", "vehicle_id"]).reset_index(drop=True)
    out = inject(df, t_from=100.0, share=0.5, seed=1)
    assert set(out["attack"]) - {"none"} and (out.loc[out["attack"] == "none", ["x", "y"]].to_numpy()
                                              == df.loc[out["attack"] == "none", ["x", "y"]].to_numpy()).all()
    drift_vid = out.loc[out["attack"] == "drift", "vehicle_id"]
    if len(drift_vid):
        inst, _ = featurize(out[out["vehicle_id"] == drift_vid.iloc[0]].reset_index(drop=True), (-1.0, 80.0))
        assert inst[:, 0].max() < 0.05 and inst[:, 1].max() < 0.5  # no physics residual: only context can catch it


def test_detector_fit_calibrate_and_persist(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    inst = np.abs(rng.normal(0, [0.1, 0.3, 2.0, 0.1, 0.0, 0.2], (4000, 6)))
    win = np.abs(rng.normal(0, 1, (4000, 9))) + [0.2, 0.5, 3, 0.2, 0, 0.1, 0, 20, 0.5]
    det = Detector.fit(inst, win, epochs=3)
    bad_inst = inst[:5].copy()
    bad_inst[:, 0] = 30.0
    assert det.rule_hits(bad_inst, np.full(5, 20.0), np.zeros(5)).all()
    assert det.rule_hits(inst[:200], np.full(200, 20.0), np.zeros(200)).mean() < 0.01
    ids = np.repeat(np.arange(10), 20)
    e = np.zeros(200, bool)
    e[20:25] = True  # vehicle 1: five suspicious messages in a row
    th, curve = calibrate(ids, e, 0.25, target=0.0)
    assert ewma_by_vehicle(ids, e).max() > 0.7 and th > 0.7 and curve[0][1] == pytest.approx(0.1)
    p = save(det, (-1.0, 25.0), tmp_path / "m.pt")
    det2, road = load(p)
    assert road == (-1.0, 25.0) and np.allclose(det2.if_scores(win[:10]), det.if_scores(win[:10]))


def test_api_scores_batches() -> None:
    rng = np.random.default_rng(1)
    inst = np.abs(rng.normal(0, [0.1, 0.3, 2.0, 0.1, 0.0, 0.2], (3000, 6)))
    win = np.abs(rng.normal(0, 1, (3000, 9))) + [0.2, 0.5, 3, 0.2, 0, 0.1, 0, 20, 0.5]
    det = Detector.fit(inst, win, epochs=2)
    det.if_thr = det.ae_thr = float("inf")  # toy training data: exercise the rule layer and the service plumbing only
    cl = TestClient(create_app(detector=det, road_x=(-1.0, 25.0)))
    f = LocalFrame()

    def bsm(vid: int, t: float, y: float, v: float) -> str:
        lat, lon = f.to_geo(np.array([5.0]), np.array([y]))
        return BSMCore.from_si(msg_cnt=int(t * 10), temp_id=vid, t_s=t, lat_deg=float(lat[0]), lon_deg=float(lon[0]),
                               speed_mps=v, heading_deg=f.bearing_deg, accel_mps2=0.0, width_m=1.8, length_m=4.5).encode().hex()

    honest = [bsm(0xAA, 10 + 0.1 * k, 50 + 2.0 * k, 20.0) for k in range(10)]
    liar = [bsm(0xBB, 10 + 0.1 * k, 200 + 2.0 * k, 35.0) for k in range(10)]  # moves 20 m/s, claims 35
    r = cl.post("/v1/bsm", json={"messages": honest + liar}).json()["results"]
    assert len(r) == 20 and not any(x["rules"] for x in r[:10]) and sum(x["rules"] for x in r[10:]) >= 8
    top = cl.get("/v1/vehicles").json()[0]
    assert top["temp_id"] == "000000bb" and top["alert"]
    assert cl.post("/v1/bsm", json={"messages": ["00ff"]}).status_code == 422
    assert cl.get("/health").json()["tracked"] == 2
