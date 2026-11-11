"""End-to-end experiment on NGSIM I-80: benign training, injected attacks, detector comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .attacks import ATTACKS, inject  # noqa: E402
from .data import LocalFrame, benign_stream, load_ngsim  # noqa: E402
from .detector import Detector, ewma_by_vehicle, save  # noqa: E402
from .engine import Engine  # noqa: E402
from .j2735 import BSMCore  # noqa: E402

VARIANTS = {"rules": ("rules",), "iforest": ("iforest",), "autoencoder": ("autoencoder",),
            "rules+iforest": ("rules", "iforest"), "all": ("rules", "iforest", "autoencoder")}
T_FIT = 200.0  # detectors are fitted on benign traffic before this time
T_SPLIT = 300.0  # benign validation in [T_FIT, T_SPLIT) sets the alert threshold; attacks start after
TARGET_FP = 0.02  # at most 2% of benign validation vehicles may be flagged
GRID = np.round(np.arange(0.5, 0.96, 0.05), 2)


def quantize(df: pd.DataFrame, frame: LocalFrame) -> pd.DataFrame:
    """Round every field to its J2735 resolution via the geodetic round trip the RSU would do."""
    out = df.copy()
    lat, lon = frame.to_geo(out["x"].to_numpy(), out["y"].to_numpy())
    x, y = frame.from_geo(np.round(lat * 1e7) / 1e7, np.round(lon * 1e7) / 1e7)
    out["x"], out["y"] = x, y
    out["speed"] = np.clip(np.round(out["speed"] / 0.02) * 0.02, 0, 8190 * 0.02)
    out["accel"] = np.clip(np.round(out["accel"] / 0.01) * 0.01, -20.0, 20.0)
    hdg = frame.heading_from_road(out["angle"].to_numpy())
    out["angle"] = np.radians(((np.round(hdg / 0.0125) * 0.0125) - frame.bearing_deg + 180) % 360 - 180)
    out["length"] = np.round(out["length"] * 100) / 100
    out["width"] = np.round(out["width"] * 100) / 100
    return out


def featurize(df: pd.DataFrame, road_x: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    eng = Engine(road_x=road_x)
    inst = np.zeros((len(df), 6))
    win = np.zeros((len(df), 9))
    cols = [df[c].to_numpy() for c in ("vehicle_id", "x", "y", "speed", "accel", "angle", "length", "width")]
    tvals = df["t"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tvals[1:] != tvals[:-1]])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        i, w = eng.process(float(tvals[s]), *(c[s:e] for c in cols))
        inst[s:e], win[s:e] = i, w
        if s % 200 == 0:
            eng.evict(float(tvals[s]))
    return inst, win


def _near_attackers(d: pd.DataFrame, attacker_ids: set[int], radius: float = 6.0, min_msgs: int = 10) -> set[int]:
    """Benign vehicles that had an attacker's *reported* position within ``radius`` for at least
    ``min_msgs`` messages (1 s): a ghost sitting on them makes their own context features abnormal too.
    Momentary passes (random-position attacks jump everywhere) do not count."""
    from collections import Counter

    from scipy.spatial import cKDTree

    near: Counter[int] = Counter()
    att = d[d["attack"] != "none"]
    ben = d[~d["vehicle_id"].isin(attacker_ids)]
    bg = dict(tuple(ben.groupby("t")))
    for t, g in att.groupby("t"):
        b = bg.get(t)
        if b is None:
            continue
        tree = cKDTree(b[["x", "y"]].to_numpy())
        hit: set[int] = set()
        for hits in tree.query_ball_point(g[["x", "y"]].to_numpy(), r=radius):
            hit.update(int(v) for v in b["vehicle_id"].to_numpy()[hits])
        near.update(hit)
    return {v for v, n in near.items() if n >= min_msgs}


def _evidence(ev: dict[str, np.ndarray], parts: tuple[str, ...]) -> np.ndarray:
    e = np.zeros(len(next(iter(ev.values()))), bool)
    for p in parts:
        e |= ev[p]
    return e


def calibrate(ids: np.ndarray, e: np.ndarray, alpha: float, target: float = TARGET_FP) -> tuple[float, list[tuple[float, float]]]:
    """Lowest alert threshold that flags at most ``target`` of benign validation vehicles."""
    c = ewma_by_vehicle(ids, e, alpha)
    vids = np.unique(ids)
    curve = [(float(th), float(len(np.unique(ids[c >= th])) / len(vids))) for th in GRID]
    ok = [th for th, fp in curve if fp <= target]
    return (ok[0] if ok else float(GRID[-1])), curve


def codec_bench(n: int = 20000) -> dict[str, float]:
    m = BSMCore.from_si(msg_cnt=5, temp_id=0xA1B2C3D4, t_s=12.3, lat_deg=37.84, lon_deg=-122.3, speed_mps=24.1,
                        heading_deg=340.0, accel_mps2=-0.8, width_m=1.9, length_m=4.6)
    t0 = time.perf_counter()
    for _ in range(n):
        b = m.encode()
    enc = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        BSMCore.decode(b)
    dec = (time.perf_counter() - t0) / n
    return {"bytes": len(b), "encode_us": enc * 1e6, "decode_us": dec * 1e6}


def run(raw_dir: str, out_dir: str | Path, *, share: float = 0.1, seed: int = 0,
        save_model: str | Path | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    frame = LocalFrame()
    raw = load_ngsim(raw_dir)
    df = benign_stream(raw, seed=seed)
    df = inject(df, t_from=T_SPLIT, share=share, seed=seed)
    df = quantize(df, frame)
    tr = df["t"] < T_FIT
    va = ((df["t"] >= T_FIT) & (df["t"] < T_SPLIT)).to_numpy()
    # road extent: everything benign vehicles occupied in training, plus 3 sd of GNSS error
    road_x = (float(df.loc[tr, "x"].min()) - 1.0, float(df.loc[tr, "x"].max()) + 1.0)
    t0 = time.perf_counter()
    inst, win = featurize(df, road_x)
    feat_s = time.perf_counter() - t0
    det = Detector.fit(inst[tr.to_numpy()], win[tr.to_numpy()], seed=seed)

    # operating point from benign validation traffic only
    dv = df.loc[va]
    ev_va = det.evidence(inst[va], win[va], dv["speed"].to_numpy(), dv["accel"].to_numpy())
    thresholds, curves = {}, {}
    for name, parts in VARIANTS.items():
        thresholds[name], curves[name] = calibrate(dv["vehicle_id"].to_numpy(), _evidence(ev_va, parts), det.alpha)
    det.alert_threshold = thresholds["all"]
    if save_model:
        save(det, road_x, save_model)

    te = (df["t"] >= T_SPLIT).to_numpy()
    d = df.loc[te].reset_index(drop=True)
    ti, tw = inst[te], win[te]
    t0 = time.perf_counter()
    ev = det.evidence(ti, tw, d["speed"].to_numpy(), d["accel"].to_numpy())
    score_s = time.perf_counter() - t0
    is_att = (d["attack"] != "none").to_numpy()
    attacker_ids = set(d.loc[is_att, "vehicle_id"])
    onset = d[is_att].groupby("vehicle_id")["t"].min()
    kind_of = d[is_att].groupby("vehicle_id")["attack"].first()
    benign_rows = ~d["vehicle_id"].isin(attacker_ids).to_numpy()

    # message-level separability of the continuous scores
    s_if, s_ae = det.if_scores(tw), det.ae_scores(tw)
    auc: dict[str, dict[str, float]] = {}
    for a in ATTACKS:
        m = (d["attack"] == a).to_numpy()
        if m.sum() == 0:
            continue
        y = np.r_[np.zeros(benign_rows.sum()), np.ones(m.sum())]
        auc[a] = {"iforest": float(roc_auc_score(y, np.r_[s_if[benign_rows], s_if[m]])),
                  "autoencoder": float(roc_auc_score(y, np.r_[s_ae[benign_rows], s_ae[m]])),
                  "rules_tpr": float(ev["rules"][m].mean())}

    # vehicle-level detection with the EWMA confidence
    ids = d["vehicle_id"].to_numpy()
    tt = d["t"].to_numpy()
    veh: dict[str, Any] = {}
    conf_all: dict[str, np.ndarray] = {}
    benign_vids = set(d["vehicle_id"]) - attacker_ids
    collateral = _near_attackers(d, attacker_ids)
    for name, parts in VARIANTS.items():
        c = ewma_by_vehicle(ids, _evidence(ev, parts), det.alpha)
        conf_all[name] = c
        alert = c >= thresholds[name]
        alerted = set(ids[alert])
        per: dict[str, Any] = {}
        for a in ATTACKS:
            vids = [v for v in kind_of.index if kind_of[v] == a]
            ttd = []
            for v in vids:
                hit = tt[(ids == v) & alert & (tt >= onset[v])]
                ttd.append(float(hit[0] - onset[v]) if len(hit) else None)
            found = [x for x in ttd if x is not None]
            per[a] = {"n": len(vids), "detected": len(found) / max(len(vids), 1),
                      "median_ttd_s": float(np.median(found)) if found else None}
        fp_veh = len(alerted & benign_vids) / len(benign_vids)
        clean = benign_vids - collateral
        fp_clean = len(alerted & clean) / len(clean)
        veh[name] = {"threshold": thresholds[name], "validation_curve": curves[name], "attacks": per, "benign_vehicles_flagged": fp_veh, "benign_not_near_attacker_flagged": fp_clean,
                     "benign_message_alert_rate": float(alert[benign_rows].mean()),
                     "attack_message_alert_rate": float(alert[is_att].mean())}

    summary = {
        "messages": {"total": len(df), "train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum()),
                     "attack": int(is_att.sum())},
        "vehicles": {"test": int(d["vehicle_id"].nunique()), "attackers": len(attacker_ids), "near_attacker": len(collateral)},
        "road_x_m": road_x, "rule_thresholds": dict(zip(("r_disp", "r_acc", "jerk", "r_lat", "off_road", "overlap"),
                                                        map(float, det.rule_thr))),
        "auc": auc, "vehicle_level": veh,
        "throughput": {"features_msgs_per_s": len(df) / feat_s, "scoring_msgs_per_s": len(d) / score_s, **codec_bench()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, d, conf_all["all"], kind_of, onset, out / "figures", thresholds["all"])
    write_report(summary, out / "RESULTS.md")
    return summary


def _plot(s: dict[str, Any], d: pd.DataFrame, conf: np.ndarray, kind_of: pd.Series, onset: pd.Series, fig_dir: Path,
          threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    names = list(VARIANTS)
    mat = np.array([[s["vehicle_level"][v]["attacks"][a]["detected"] for v in names] for a in ATTACKS])
    ax = axes[0]
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(ATTACKS)), ATTACKS)
    for i in range(len(ATTACKS)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8, color="k" if mat[i, j] < 0.6 else "w")
    ax.set_title("Attacking vehicles detected (calibrated thresholds)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax = axes[1]
    ids, tt = d["vehicle_id"].to_numpy(), d["t"].to_numpy()
    for a, col in (("drift", "#c62828"), ("lane_shift", "#6a1b9a"), ("eventual_stop", "#ef6c00"), ("speed_high", "#1565c0")):
        vs = [v for v in kind_of.index if kind_of[v] == a]
        if vs:
            m = ids == vs[0]
            ax.plot(tt[m] - onset[vs[0]], conf[m], color=col, label=a)
    benign = [v for v in pd.unique(ids) if v not in set(kind_of.index)][:3]
    for k, v in enumerate(benign):
        m = ids == v
        ax.plot(tt[m] - tt[m][0] - 3.0, conf[m], color="0.6", lw=0.8, label="benign" if k == 0 else None)
    ax.axhline(threshold, color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-3, 30)
    ax.set_xlabel("seconds from attack onset")
    ax.set_ylabel("attack confidence (all detectors)")
    ax.set_title("Per-vehicle confidence over time")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "detection.png", dpi=150)
    plt.close(fig)


def write_report(s: dict[str, Any], path: Path) -> Path:
    m, v = s["messages"], s["vehicles"]
    lines = ["# BSM false-data-injection detection on NGSIM I-80", "",
             f"{m['total']:,} BSMs ({m['train']:,} benign training, {m['validation']:,} benign validation, "
             f"{m['test']:,} test of which {m['attack']:,} falsified); "
             f"{v['test']} vehicles in the test period, {v['attackers']} attackers.", "",
             "## Attacking vehicles detected / median time to detect (s)", "",
             "| Attack | " + " | ".join(VARIANTS) + " |", "|---" * (len(VARIANTS) + 1) + "|"]
    for a in ATTACKS:
        cells = []
        for name in VARIANTS:
            r = s["vehicle_level"][name]["attacks"][a]
            ttd = "-" if r["median_ttd_s"] is None else f"{r['median_ttd_s']:.1f}"
            cells.append(f"{r['detected']:.0%} / {ttd}")
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    lines += ["", f"## False alarms ({v['test'] - v['attackers']} benign vehicles in the test period, {v['near_attacker']} of them "
              "had an attacker's reported position within 6 m for 1 s or more)", "",
              "| Detector | Alert threshold (set on benign validation) | Benign vehicles flagged | ...excluding those near an attacker | Benign message alert rate |",
              "|---|---|---|---|---|"]
    for name in VARIANTS:
        r = s["vehicle_level"][name]
        lines.append(f"| {name} | {r['threshold']:.2f} | {r['benign_vehicles_flagged']:.1%} | {r['benign_not_near_attacker_flagged']:.1%} | "
                     f"{r['benign_message_alert_rate']:.2%} |")
    lines += ["", "## Message-level ROC AUC of the learned scores", "", "| Attack | Isolation forest | Autoencoder | Rules TPR |", "|---|---|---|---|"]
    for a, r in s["auc"].items():
        lines.append(f"| {a} | {r['iforest']:.3f} | {r['autoencoder']:.3f} | {r['rules_tpr']:.2f} |")
    th = s["throughput"]
    lines += ["", "## Throughput (one CPU core)", "",
              f"* feature engine: {th['features_msgs_per_s']:,.0f} messages/s; scoring: {th['scoring_msgs_per_s']:,.0f} messages/s",
              f"* UPER BSMcoreData: {th['bytes']} bytes, encode {th['encode_us']:.1f} us, decode {th['decode_us']:.1f} us", "",
              "![Detection](figures/detection.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
