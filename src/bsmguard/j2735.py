"""SAE J2735 ``BSMcoreData`` with its unaligned PER (UPER) encoding.

Every field of ``BSMcoreData`` is a constrained INTEGER, fixed-size BIT/OCTET STRING or non-extensible
ENUMERATED, so UPER reduces to writing each value as ``value - lower_bound`` in
``ceil(log2(range))`` bits, in declaration order. The wire size is 290 bits (37 bytes).

Units (J2735 data elements): latitude/longitude 1/10 micro-degree, elevation 0.1 m, speed 0.02 m/s,
heading 0.0125 degree clockwise from north, accelerations 0.01 m/s^2, yaw rate 0.01 deg/s,
vehicle size in cm, ``secMark`` in milliseconds within the minute.

Scope: the core data frame only. Part II extensions and the ``MessageFrame`` wrapper (messageId 20)
are not encoded. "Unavailable" sentinel values are supported by the ranges but not interpreted.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass, fields

# (name, lower bound, upper bound) in J2735 declaration order
SPEC: tuple[tuple[str, int, int], ...] = (
    ("msg_cnt", 0, 127),
    ("temp_id", 0, 2**32 - 1),  # OCTET STRING (SIZE(4))
    ("sec_mark", 0, 65535),
    ("lat", -900000000, 900000001),
    ("lon", -1799999999, 1800000001),
    ("elev", -4096, 61439),
    ("acc_semi_major", 0, 255),
    ("acc_semi_minor", 0, 255),
    ("acc_orientation", 0, 65535),
    ("transmission", 0, 7),
    ("speed", 0, 8191),
    ("heading", 0, 28800),
    ("angle", -126, 127),
    ("accel_long", -2000, 2001),
    ("accel_lat", -2000, 2001),
    ("accel_vert", -127, 127),
    ("yaw_rate", -32767, 32767),
    ("wheel_brakes", 0, 31),  # BIT STRING (SIZE(5))
    ("traction", 0, 3),
    ("abs_", 0, 3),
    ("scs", 0, 3),
    ("brake_boost", 0, 2),
    ("aux_brakes", 0, 3),
    ("width", 0, 1023),
    ("length", 0, 4095),
)
BITS: tuple[int, ...] = tuple(max(1, math.ceil(math.log2(hi - lo + 1))) for _, lo, hi in SPEC)
TOTAL_BITS = sum(BITS)
NBYTES = (TOTAL_BITS + 7) // 8


class J2735Error(ValueError):
    """Value out of range or malformed encoding."""


@dataclass(frozen=True)
class BSMCore:
    msg_cnt: int
    temp_id: int
    sec_mark: int
    lat: int
    lon: int
    elev: int
    acc_semi_major: int
    acc_semi_minor: int
    acc_orientation: int
    transmission: int
    speed: int
    heading: int
    angle: int
    accel_long: int
    accel_lat: int
    accel_vert: int
    yaw_rate: int
    wheel_brakes: int
    traction: int
    abs_: int
    scs: int
    brake_boost: int
    aux_brakes: int
    width: int
    length: int

    # ------------------------------------------------------------- SI helpers
    @classmethod
    def from_si(cls, *, msg_cnt: int, temp_id: int, t_s: float, lat_deg: float, lon_deg: float, speed_mps: float,
                heading_deg: float, accel_mps2: float, width_m: float, length_m: float, elev_m: float = 10.0,
                pos_sigma_m: float = 1.5) -> "BSMCore":
        def clip(v: float, lo: int, hi: int) -> int:
            return int(min(max(round(v), lo), hi))

        return cls(msg_cnt % 128, temp_id & 0xFFFFFFFF, int(round(t_s * 1000)) % 60000,
                   clip(lat_deg * 1e7, -900000000, 900000000), clip(lon_deg * 1e7, -1799999999, 1800000000),
                   clip(elev_m * 10, -4095, 61439), clip(pos_sigma_m / 0.05, 0, 254), clip(pos_sigma_m / 0.05, 0, 254), 0,
                   2, clip(speed_mps / 0.02, 0, 8190), clip((heading_deg % 360.0) / 0.0125, 0, 28799), 127,
                   clip(accel_mps2 * 100, -2000, 2000), 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   clip(width_m * 100, 0, 1023), clip(length_m * 100, 0, 4095))

    @property
    def speed_mps(self) -> float:
        return self.speed * 0.02

    @property
    def heading_deg(self) -> float:
        return self.heading * 0.0125

    @property
    def accel_mps2(self) -> float:
        return self.accel_long * 0.01

    @property
    def lat_deg(self) -> float:
        return self.lat * 1e-7

    @property
    def lon_deg(self) -> float:
        return self.lon * 1e-7

    # ------------------------------------------------------------------ UPER
    def encode(self) -> bytes:
        acc = 0
        for (name, lo, hi), nb, v in zip(SPEC, BITS, astuple(self)):
            if not lo <= v <= hi:
                raise J2735Error(f"{name}={v} outside [{lo}, {hi}]")
            acc = (acc << nb) | (v - lo)
        pad = NBYTES * 8 - TOTAL_BITS
        return (acc << pad).to_bytes(NBYTES, "big")

    @classmethod
    def decode(cls, data: bytes) -> "BSMCore":
        if len(data) != NBYTES:
            raise J2735Error(f"BSMcoreData is {NBYTES} bytes, got {len(data)}")
        acc = int.from_bytes(data, "big") >> (NBYTES * 8 - TOTAL_BITS)
        vals = []
        for (name, lo, hi), nb in zip(reversed(SPEC), reversed(BITS)):
            v = (acc & ((1 << nb) - 1)) + lo
            acc >>= nb
            if v > hi:
                raise J2735Error(f"{name} decodes outside its range")
            vals.append(v)
        return cls(*reversed(vals))


assert [f.name for f in fields(BSMCore)] == [s[0] for s in SPEC]
