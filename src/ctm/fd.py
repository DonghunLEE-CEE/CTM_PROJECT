from __future__ import annotations

from .types import FDParams


def q_max_total_vph(q_max_lane: float, n_lanes: int) -> float:
    return q_max_lane * n_lanes


def n_max_cell_veh(k_jam_lane: float, n_lanes: int, cell_len_km: float) -> float:
    """Jam accumulation (veh) in a cell of length L with per-lane jam density."""
    return k_jam_lane * n_lanes * cell_len_km


def send_veh(x_cell: float, q_max_lane: float, n_lanes: int, dt_h: float) -> float:
    """S = min(x, Q_max_total * dt). Q_max scales with lanes."""
    q_dt = q_max_lane * n_lanes * dt_h
    return float(min(x_cell, q_dt))


def receive_veh(
    x_cell: float,
    q_max_lane: float,
    n_lanes: int,
    w_kmh: float,
    cell_len_km: float,
    n_max: float,
    dt_h: float,
) -> float:
    """
    R = min(Q_max_total*dt, (w/L) * (N_max - x) * dt).
    Send/receive each include the q_max cap (CTM triple min via construction).
    """
    q_dt = q_max_lane * n_lanes * dt_h
    spare = max(0.0, n_max - x_cell)
    wave_term = (w_kmh / cell_len_km) * spare * dt_h
    return float(min(q_dt, wave_term))


def merge_proportional(s_main: float, s_ramp: float, r_cap: float) -> tuple[float, float]:
    """Proportional merge into receiving r_cap. Returns (f_main, f_ramp)."""
    d = s_main + s_ramp
    if d <= r_cap + 1e-12:
        return float(s_main), float(s_ramp)
    scale = r_cap / d if d > 0 else 0.0
    return float(scale * s_main), float(scale * s_ramp)


def density_lane_km(x_cell: float, n_lanes: int, cell_len_km: float) -> float:
    """Average per-lane density (veh/km/lane)."""
    denom = max(n_lanes * cell_len_km, 1e-12)
    return float(x_cell / denom)


def fd_speed_kmh(k_lane: float, fd: FDParams) -> float:
    """
    Piecewise analytical v(k) on a trapezoid/triangle with free-flow v_ff,
    backward wave w, jam k_jam, and capped q_max (per lane).
    """
    k = max(k_lane, 0.0)
    if k <= 1e-9:
        return float(fd.v_ff_kmh)
    k_free = fd.q_max_vph_lane / max(fd.v_ff_kmh, 1e-9)
    if k <= k_free + 1e-12:
        return float(fd.v_ff_kmh)
    if k >= fd.k_jam_vpk_lane - 1e-12:
        return 0.0
    v_cong = fd.w_kmh * (fd.k_jam_vpk_lane / k - 1.0)
    return float(max(0.0, min(fd.v_ff_kmh, v_cong)))


def fd_flow_lane(k_lane: float, fd: FDParams) -> float:
    """Per-lane flow q(k) consistent with fd_speed (veh/h/lane)."""
    v = fd_speed_kmh(k_lane, fd)
    return float(k_lane * v)
