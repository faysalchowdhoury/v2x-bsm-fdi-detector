"""Scoring microservice: a roadside unit (or a message broker consumer) posts the BSMs it receives.

``POST /v1/bsm``        a batch of UPER-encoded ``BSMcoreData`` (hex). Messages are grouped by their
                        ``secMark`` into 100-ms slots and processed in time order, so context features
                        see every vehicle reported in the same slot. Returns per-message evidence and
                        the sender's attack confidence.
``GET  /v1/vehicles``   current confidence per temporary ID, highest first
``GET  /health``

State (per-vehicle history and confidence) lives in the process: run one replica per RSU, or shard by
RSU in front of a pool.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__
from ..data import LocalFrame
from ..detector import Detector, load
from ..engine import Engine
from ..j2735 import BSMCore, J2735Error


class Batch(BaseModel):
    messages: list[str] = Field(min_length=1, max_length=5000, description="hex-encoded UPER BSMcoreData")


def create_app(model_path: str | None = None, detector: Detector | None = None,
               road_x: tuple[float, float] | None = None) -> FastAPI:
    if detector is None:
        detector, road_x = load(model_path or os.environ.get("BSMGUARD_MODEL", "models/detector.pt"))
    assert road_x is not None
    det = detector
    eng = Engine(road_x=road_x)
    frame = LocalFrame()
    app = FastAPI(title="BSM false-data-injection detector", version=__version__)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "alert_threshold": det.alert_threshold, "tracked": len(eng.state)}

    @app.post("/v1/bsm")
    def score(batch: Batch) -> dict[str, object]:
        try:
            msgs = [BSMCore.decode(bytes.fromhex(h)) for h in batch.messages]
        except (ValueError, J2735Error) as exc:
            raise HTTPException(422, f"bad BSM: {exc}") from exc
        slots: dict[int, list[int]] = defaultdict(list)
        for k, m in enumerate(msgs):
            slots[round(m.sec_mark / 100)].append(k)
        results: list[dict[str, object] | None] = [None] * len(msgs)
        for slot in sorted(slots):
            ks = slots[slot]
            sub = [msgs[k] for k in ks]
            x, y = frame.from_geo(np.array([m.lat_deg for m in sub]), np.array([m.lon_deg for m in sub]))
            ids = np.array([m.temp_id for m in sub])
            v = np.array([m.speed_mps for m in sub])
            a = np.array([m.accel_mps2 for m in sub])
            ang = np.radians((np.array([m.heading_deg for m in sub]) - frame.bearing_deg + 180) % 360 - 180)
            inst, win = eng.process(slot / 10.0, ids, x, y, v, a, ang, np.array([m.length / 100 for m in sub]),
                                    np.array([m.width / 100 for m in sub]))
            ev = det.evidence(inst, win, v, a)
            e = ev["rules"] | ev["iforest"] | ev["autoencoder"]
            conf = det.update_confidence(ids, e)
            for j, k in enumerate(ks):
                results[k] = {"temp_id": f"{ids[j]:08x}", "rules": bool(ev["rules"][j]), "iforest": bool(ev["iforest"][j]),
                              "autoencoder": bool(ev["autoencoder"][j]), "confidence": round(float(conf[j]), 3),
                              "alert": bool(conf[j] >= det.alert_threshold)}
        return {"results": results}

    @app.get("/v1/vehicles")
    def vehicles(limit: int = 50) -> list[dict[str, object]]:
        ranked = sorted(det.conf.items(), key=lambda kv: -kv[1])[:limit]
        return [{"temp_id": f"{k:08x}", "confidence": round(c, 3), "alert": c >= det.alert_threshold} for k, c in ranked]

    return app
