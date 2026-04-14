from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FDParams:
    """Fundamental diagram / CTM limits (per lane)."""

    v_ff_kmh: float
    w_kmh: float
    q_max_vph_lane: float
    k_jam_vpk_lane: float


@dataclass(frozen=True)
class DemandProfile:
    """Boundary and ramp demands in veh/h (converted to veh/dt inside the engine)."""

    upstream_vph: float
    onramp_vph: float
