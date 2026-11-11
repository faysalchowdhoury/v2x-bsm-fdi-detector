"""Benign BSM streams from real NGSIM I-80 trajectories.

NGSIM gives 10-Hz positions of every vehicle on a 500-m section of I-80 (Emeryville, CA, 4:00-4:15 pm,
2005), extracted from video. Its raw speeds and accelerations are differentiated positions and are
famously noisy, so each trajectory is smoothed (Savitzky-Golay, 2.1-s window), and speed, acceleration
and heading are recomputed from the smoothed path, standing in for the CAN-bus values a real on-board
unit reports. Position then gets **GNSS error**: a slowly varying AR(1) bias (sd 1.5 m, 10-Hz
correlation 0.998) plus 5-cm white noise, which is what consecutive BSMs from one receiver look like.

Positions are placed on the globe with a local tangent plane anchored near the study site. The anchor
and road bearing are approximate; nothing downstream depends on the exact georeference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FT = 0.3048
EARTH_R = 6_378_137.0


@dataclass(frozen=True)
class LocalFrame:
    """Road frame (x lateral to the right, y along the road) <-> WGS-84, via a local tangent plane."""

    lat0: float = 37.8395
    lon0: float = -122.2985
    bearing_deg: float = 340.0  # direction of travel (degrees clockwise from north)

    def to_geo(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.radians(self.bearing_deg)
        east = y * np.sin(b) + x * np.cos(b)
        north = y * np.cos(b) - x * np.sin(b)
        lat = self.lat0 + np.degrees(north / EARTH_R)
        lon = self.lon0 + np.degrees(east / (EARTH_R * np.cos(np.radians(self.lat0))))
        return lat, lon

    def from_geo(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        north = np.radians(np.asarray(lat) - self.lat0) * EARTH_R
        east = np.radians(np.asarray(lon) - self.lon0) * EARTH_R * np.cos(np.radians(self.lat0))
        b = np.radians(self.bearing_deg)
        y = east * np.sin(b) + north * np.cos(b)
        x = east * np.cos(b) - north * np.sin(b)
        return x, y

    def heading_from_road(self, road_angle_rad: np.ndarray) -> np.ndarray:
        """Road-frame direction (angle of motion from +y toward +x) -> compass heading in degrees."""
        return (self.bearing_deg + np.degrees(road_angle_rad)) % 360.0


def load_ngsim(raw_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("page_*.csv"))
    if not files:
        raise FileNotFoundError(f"no NGSIM pages in {raw_dir}")
    cols = ["vehicle_id", "global_time", "local_x", "local_y", "v_length", "v_width", "lane_id"]
    df = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    df = df.drop_duplicates(["vehicle_id", "global_time"]).sort_values(["vehicle_id", "global_time"])
    df["t"] = (df["global_time"] - df["global_time"].min()) / 1000.0
    df["x"] = df["local_x"] * FT
    df["y"] = df["local_y"] * FT
    df["length"] = df["v_length"] * FT
    df["width"] = df["v_width"] * FT
    return df[["vehicle_id", "t", "x", "y", "length", "width", "lane_id"]].reset_index(drop=True)


def benign_stream(raw: pd.DataFrame, *, seed: int = 0, gnss_sd: float = 1.5, gnss_rho: float = 0.998,
                  white_sd: float = 0.05, window: int = 21, min_len: int = 40) -> pd.DataFrame:
    """One row per BSM with true road-frame state and the reported (noisy) state."""
    rng = np.random.default_rng(seed)
    parts = []
    for vid, g in raw.groupby("vehicle_id", sort=False):
        if len(g) < min_len:
            continue
        t = g["t"].to_numpy()
        if np.any(np.diff(t) > 0.15):  # gaps: keep the longest contiguous run
            cut = np.flatnonzero(np.diff(t) > 0.15) + 1
            seg = max(np.split(np.arange(len(t)), cut), key=len)
            g = g.iloc[seg]
            t = t[seg]
            if len(g) < min_len:
                continue
        x = savgol_filter(g["x"].to_numpy(), window, 3)
        y = savgol_filter(g["y"].to_numpy(), window, 3)
        vx = savgol_filter(g["x"].to_numpy(), window, 3, deriv=1, delta=0.1)
        vy = savgol_filter(g["y"].to_numpy(), window, 3, deriv=1, delta=0.1)
        speed = np.hypot(vx, vy)
        accel = savgol_filter(speed, window, 3, deriv=1, delta=0.1)
        ang = np.arctan2(vx, vy)  # 0 = straight along the road
        n = len(g)
        bias = np.empty((n, 2))
        bias[0] = rng.normal(0, gnss_sd, 2)
        innov = gnss_sd * np.sqrt(1 - gnss_rho**2)
        for k in range(1, n):
            bias[k] = gnss_rho * bias[k - 1] + rng.normal(0, innov, 2)
        noise = bias + rng.normal(0, white_sd, (n, 2))
        # the filter's derivatives are distorted within half a window of each end; drop those messages
        # (an early version kept them and they produced spurious acceleration and jerk alarms)
        keep = slice(window // 2, n - window // 2)
        parts.append(pd.DataFrame({
            "vehicle_id": vid, "t": np.round(t, 1), "x_true": x, "y_true": y,
            "x": x + noise[:, 0], "y": y + noise[:, 1], "speed": np.clip(speed, 0, None), "accel": accel,
            "angle": ang, "length": g["length"].to_numpy(), "width": g["width"].to_numpy(),
        }).iloc[keep])
    df = pd.concat(parts, ignore_index=True)
    df["attack"] = "none"
    return df.sort_values(["t", "vehicle_id"]).reset_index(drop=True)
