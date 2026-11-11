"""Detectors on engine features, all fitted on benign traffic only (no attack labels are used).

* **Rules**: a message violates physics if any instantaneous residual exceeds ``margin`` times its
  99.99th percentile in benign traffic (with a floor), or a hard limit (speed > 60 m/s, |a| > 10 m/s^2).
* **Isolation forest** and an **autoencoder** on log-scaled window features; a message is anomalous if
  its score is beyond the 99.9th percentile of benign training scores.
* **Attack confidence** per vehicle: an exponentially weighted average of per-message evidence
  (any detector firing), ``c <- (1 - alpha) c + alpha e``. A vehicle is reported when ``c`` reaches
  ``alert_threshold``, calibrated on held-out *benign* traffic so that at most 2% of normal vehicles
  are ever flagged (0.5 corresponds to about three consecutive suspicious messages).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from .engine import INSTANT

FLOORS = {"r_disp": 1.0, "r_acc": 2.0, "jerk": 15.0, "r_lat": 1.0, "off_road": 1.0, "overlap": 99.0}
# overlap is not a rule: GNSS error makes benign footprints overlap in queues; it is left to the models


def _log(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(x))


class AutoEncoder(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 4))
        self.dec = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


@dataclass
class Detector:
    rule_thr: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    forest: IsolationForest
    if_thr: float
    ae: AutoEncoder
    ae_thr: float
    alpha: float = 0.25
    alert_threshold: float = 0.5
    conf: dict[int, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, inst: np.ndarray, win: np.ndarray, *, margin: float = 1.5, q: float = 0.999, seed: int = 0,
            epochs: int = 25) -> "Detector":
        thr = np.array([max(FLOORS[name], margin * np.quantile(inst[:, k], 0.9999)) for k, name in enumerate(INSTANT)])
        z = _log(win)
        mu, sd = z.mean(0), z.std(0) + 1e-6
        zs = (z - mu) / sd
        rng = np.random.default_rng(seed)
        sub = zs[rng.choice(len(zs), size=min(len(zs), 60000), replace=False)]
        forest = IsolationForest(n_estimators=200, random_state=seed).fit(sub)
        if_thr = float(np.quantile(-forest.score_samples(sub), q))
        torch.manual_seed(seed)
        ae = AutoEncoder(zs.shape[1])
        opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
        xt = torch.tensor(zs, dtype=torch.float32)
        for _ in range(epochs):
            for b in torch.randperm(len(xt)).split(1024):
                opt.zero_grad()
                loss = ((ae(xt[b]) - xt[b]) ** 2).mean()
                loss.backward()
                opt.step()
        with torch.no_grad():
            err = ((ae(xt) - xt) ** 2).mean(1).numpy()
        return cls(thr, mu, sd, forest, if_thr, ae, float(np.quantile(err, q)))

    # ---------------------------------------------------------- per message
    def rule_hits(self, inst: np.ndarray, speed: np.ndarray, accel: np.ndarray) -> np.ndarray:
        return (inst > self.rule_thr).any(axis=1) | (speed > 60.0) | (np.abs(accel) > 10.0)

    def if_scores(self, win: np.ndarray) -> np.ndarray:
        return -self.forest.score_samples((_log(win) - self.mu) / self.sd)

    def ae_scores(self, win: np.ndarray) -> np.ndarray:
        x = torch.tensor((_log(win) - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            return ((self.ae(x) - x) ** 2).mean(1).numpy()

    def evidence(self, inst: np.ndarray, win: np.ndarray, speed: np.ndarray, accel: np.ndarray) -> dict[str, np.ndarray]:
        return {"rules": self.rule_hits(inst, speed, accel), "iforest": self.if_scores(win) > self.if_thr,
                "autoencoder": self.ae_scores(win) > self.ae_thr}

    def update_confidence(self, ids: np.ndarray, e: np.ndarray) -> np.ndarray:
        out = np.empty(len(ids))
        for k, (vid, ev) in enumerate(zip(ids, e)):
            c = (1 - self.alpha) * self.conf.get(int(vid), 0.0) + self.alpha * float(ev)
            self.conf[int(vid)] = c
            out[k] = c
        return out


def ewma_by_vehicle(ids: np.ndarray, e: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Offline equivalent of :meth:`Detector.update_confidence` for rows in time order."""
    conf: dict[int, float] = {}
    out = np.empty(len(ids))
    for k, (vid, ev) in enumerate(zip(ids, e)):
        c = (1 - alpha) * conf.get(int(vid), 0.0) + alpha * float(ev)
        conf[int(vid)] = c
        out[k] = c
    return out


def save(det: Detector, road_x: tuple[float, float], path: str | Path) -> Path:
    """Persist a fitted detector with the road extent its engine needs. Pickle: load trusted files only."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump({"detector": det, "road_x": road_x}, fh)
    return p


def load(path: str | Path) -> tuple[Detector, tuple[float, float]]:
    with Path(path).open("rb") as fh:
        d = pickle.load(fh)  # noqa: S301 - model files are produced by `bsmguard study`
    det: Detector = d["detector"]
    det.conf = {}
    return det, tuple(d["road_x"])  # type: ignore[return-value]
