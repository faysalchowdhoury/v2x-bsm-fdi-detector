"""Command line: ``bsmguard study`` (experiment on NGSIM I-80) and ``bsmguard serve`` (scoring API)."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bsmguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("study", help="train on benign NGSIM BSMs, inject attacks, compare detectors")
    st.add_argument("--raw", required=True, help="directory of NGSIM I-80 page_*.csv files")
    st.add_argument("--out", default="results/i80")
    st.add_argument("--share", type=float, default=0.1, help="share of eligible test vehicles that attack")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--save-model", default="models/detector.pt")
    sv = sub.add_parser("serve", help="run the scoring API")
    sv.add_argument("--model", default="models/detector.pt")
    sv.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    if a.cmd == "study":
        from .study import run

        s = run(a.raw, a.out, share=a.share, seed=a.seed, save_model=a.save_model)
        print(json.dumps({k: s["vehicle_level"]["all"][k] for k in ("benign_vehicles_flagged", "benign_message_alert_rate")}, indent=2))
        return 0
    import os

    import uvicorn

    os.environ["BSMGUARD_MODEL"] = a.model
    uvicorn.run("bsmguard.api.app:create_app", factory=True, host="0.0.0.0", port=a.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
