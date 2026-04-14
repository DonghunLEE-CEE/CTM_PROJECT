from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fd import (
    density_lane_km,
    fd_flow_lane,
    fd_speed_kmh,
    merge_proportional,
    n_max_cell_veh,
    receive_veh,
    send_veh,
)
from .network import NetworkGeometry
from .types import FDParams


@dataclass(frozen=True)
class SimulationConfig:
    """
    peak_windows: (start_min, end_min) 닫힌 구간들의 튜플. t가 어느 구간에든 들어가면 peak 유량.
    """

    geometry: NetworkGeometry
    fd: FDParams
    n_lanes: int
    dt_minutes: float
    n_steps: int
    upstream_off_peak_vph: float
    onramp_off_peak_vph: float
    upstream_peak_vph: float
    onramp_peak_vph: float
    peak_windows: tuple[tuple[float, float], ...]
    off_split_beta: float
    accidents: tuple["AccidentSpec", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AccidentSpec:
    """단일 사고 이벤트: 지정 시간 동안 지정 셀의 레인 일부를 차단."""

    cell_idx: int
    start_min: float
    duration_min: float
    blocked_lanes: int


@dataclass(frozen=True)
class SimulationResult:
    """x, k_lane, v_kmh, ramp_queue: shape (n_steps+1, n_cells), up_queue: (n_steps+1,)."""

    x: np.ndarray
    k_lane: np.ndarray
    v_kmh: np.ndarray
    ramp_queue: np.ndarray
    up_queue: np.ndarray


def _dt_hours(dt_minutes: float) -> float:
    return float(dt_minutes) / 60.0


def demands_at_time(t_min: float, cfg: SimulationConfig) -> tuple[float, float]:
    """시각 t(분) 기준 upstream / on-ramp **희망 유입률** (veh/h)."""
    t = float(t_min)
    for lo, hi in cfg.peak_windows:
        if float(lo) <= t <= float(hi):
            return float(cfg.upstream_peak_vph), float(cfg.onramp_peak_vph)
    return float(cfg.upstream_off_peak_vph), float(cfg.onramp_off_peak_vph)


def parse_peak_windows(text: str) -> tuple[tuple[float, float], ...]:
    """'60-120, 180-220' 형식. 각 구간은 닫힌 구간 [lo, hi] (lo <= t <= hi)."""
    pairs: list[tuple[float, float]] = []
    for raw in text.replace(";", ",").split(","):
        part = raw.strip()
        if not part or "-" not in part:
            continue
        a, b = part.split("-", 1)
        lo, hi = float(a), float(b.strip())
        if lo > hi:
            lo, hi = hi, lo
        pairs.append((lo, hi))
    return tuple(pairs)


def _blocked_lanes_at_time(t_min: float, cfg: SimulationConfig, n_cells: int) -> np.ndarray:
    blocked = np.zeros(n_cells, dtype=int)
    for ac in cfg.accidents:
        c = int(ac.cell_idx)
        if not (0 <= c < n_cells):
            continue
        start = float(ac.start_min)
        end = start + max(float(ac.duration_min), 0.0)
        if start <= float(t_min) < end:
            blocked[c] += max(int(ac.blocked_lanes), 0)
    return blocked


def _masks(geom: NetworkGeometry, n: int) -> tuple[np.ndarray, np.ndarray]:
    is_on = np.zeros(n, dtype=bool)
    is_off = np.zeros(n, dtype=bool)
    for idx in geom.on_ramp_cells:
        if 0 <= idx < n:
            is_on[idx] = True
    for idx in geom.off_ramp_cells:
        if 0 <= idx < n:
            is_off[idx] = True
    return is_on, is_off


def ctm_step(
    x: np.ndarray,
    x_ramp: np.ndarray,
    x_up_queue: float,
    geom: NetworkGeometry,
    fd: FDParams,
    n_lanes: int,
    lanes_open: np.ndarray,
    dt_h: float,
    q_up_vph: float,
    q_ramp_vph: float,
    beta: float,
    is_on: np.ndarray,
    is_off: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    한 스텝 CTM.
    - 상류 큐 x_up_queue에 희망 유입 누적 후, 실제 셀 0 유입만큼 차감.
    - 각 on-ramp 큐 x_ramp[i]에 동일 ramp 희망 유입 누적, 병합 시 ramp send 사용.
    - off는 먼저 분리한 뒤, 남은 본선 공급과 ramp send를 R에 대해 merge (R에서 ramp_dt 차감 금지).
    """
    n = int(x.shape[0])
    L = float(geom.cell_len_km)
    q_lane = fd.q_max_vph_lane
    b = min(max(float(beta), 0.0), 1.0 - 1e-15)
    lanes_open = np.asarray(lanes_open, dtype=int)

    x = np.asarray(x, dtype=float).copy()
    x_ramp = np.asarray(x_ramp, dtype=float).copy()
    x_up_queue = float(x_up_queue)

    x_up_queue += q_up_vph * dt_h
    for i in range(n):
        if is_on[i]:
            x_ramp[i] += q_ramp_vph * dt_h

    s = np.zeros(n)
    r = np.zeros(n)
    for i in range(n):
        lanes_i = max(int(lanes_open[i]), 0)
        n_max_i = n_max_cell_veh(fd.k_jam_vpk_lane, lanes_i, L)
        s[i] = send_veh(float(x[i]), q_lane, lanes_i, dt_h)
        r[i] = receive_veh(
            float(x[i]),
            q_lane,
            lanes_i,
            fd.w_kmh,
            L,
            n_max_i,
            dt_h,
        )

    inflow = np.zeros(n)
    out_off = np.zeros(n)
    out_r = np.zeros(max(n - 1, 0))
    out_sink = np.zeros(n)

    s_bnd = send_veh(x_up_queue, q_lane, n_lanes, dt_h)
    if is_on[0]:
        s_r0 = send_veh(float(x_ramp[0]), q_lane, n_lanes, dt_h)
        fb, fr = merge_proportional(s_bnd, s_r0, r[0])
        inflow[0] = fb + fr
        x_up_queue -= fb
        x_ramp[0] -= fr
    else:
        fin = min(s_bnd, r[0])
        inflow[0] = fin
        x_up_queue -= fin

    for i in range(n - 1):
        si = float(s[i])
        if is_off[i] and is_on[i + 1]:
            s_r = send_veh(float(x_ramp[i + 1]), q_lane, n_lanes, dt_h)
            if (1.0 - b) <= 1e-14:
                out_off[i] = si
                fr = min(s_r, r[i + 1])
                inflow[i + 1] += fr
                x_ramp[i + 1] -= fr
                out_r[i] = 0.0
            else:
                main_try = (1.0 - b) * si
                fm, fr = merge_proportional(main_try, s_r, r[i + 1])
                out_r[i] = fm
                out_off[i] = b * fm / (1.0 - b)
                inflow[i + 1] += fm + fr
                x_ramp[i + 1] -= fr
        elif is_off[i]:
            y = min(si, r[i + 1] / (1.0 - b))
            out_off[i] = b * y
            out_r[i] = (1.0 - b) * y
            inflow[i + 1] += out_r[i]
        elif is_on[i + 1]:
            s_r = send_veh(float(x_ramp[i + 1]), q_lane, n_lanes, dt_h)
            fm, fr = merge_proportional(si, s_r, r[i + 1])
            out_r[i] = fm
            inflow[i + 1] += fm + fr
            x_ramp[i + 1] -= fr
        else:
            out_r[i] = min(si, r[i + 1])
            inflow[i + 1] += out_r[i]

    ln = n - 1
    if is_off[ln]:
        y_last = float(s[ln])
        out_off[ln] = b * y_last
        out_sink[ln] = (1.0 - b) * y_last
    else:
        out_sink[ln] = float(s[ln])

    x_new = x + inflow - out_off - out_sink
    if n > 1:
        x_new[:-1] -= out_r
    x_new = np.maximum(x_new, 0.0)
    x_ramp = np.maximum(x_ramp, 0.0)
    x_up_queue = max(x_up_queue, 0.0)
    return x_new, x_ramp, x_up_queue


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    geom = cfg.geometry
    n = geom.n_cells
    dt_h = _dt_hours(cfg.dt_minutes)
    is_on, is_off = _masks(geom, n)

    x = np.zeros(n, dtype=float)
    x_ramp = np.zeros(n, dtype=float)
    x_up_queue = 0.0
    xs: list[np.ndarray] = [x.copy()]
    xr_hist: list[np.ndarray] = [x_ramp.copy()]
    up_hist: list[float] = [x_up_queue]

    for k in range(cfg.n_steps):
        t_min = float(k) * float(cfg.dt_minutes)
        q_up, q_r = demands_at_time(t_min, cfg)
        blocked = _blocked_lanes_at_time(t_min, cfg, n)
        lanes_open = np.maximum(int(cfg.n_lanes) - blocked, 0)
        x, x_ramp, x_up_queue = ctm_step(
            x,
            x_ramp,
            x_up_queue,
            geom,
            cfg.fd,
            cfg.n_lanes,
            lanes_open,
            dt_h,
            q_up,
            q_r,
            cfg.off_split_beta,
            is_on,
            is_off,
        )
        xs.append(x.copy())
        xr_hist.append(x_ramp.copy())
        up_hist.append(float(x_up_queue))

    x_hist = np.stack(xs, axis=0)
    k_lane = np.zeros_like(x_hist)
    v_kmh = np.zeros_like(x_hist)
    for t in range(x_hist.shape[0]):
        for i in range(n):
            k_lane[t, i] = density_lane_km(float(x_hist[t, i]), cfg.n_lanes, geom.cell_len_km)
            v_kmh[t, i] = fd_speed_kmh(float(k_lane[t, i]), cfg.fd)

    return SimulationResult(
        x=x_hist,
        k_lane=k_lane,
        v_kmh=v_kmh,
        ramp_queue=np.stack(xr_hist, axis=0),
        up_queue=np.asarray(up_hist, dtype=float),
    )


def fd_scatter_points(k_lane: np.ndarray, v_kmh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return k_lane.reshape(-1), v_kmh.reshape(-1)


def fd_theory_curve(fd: FDParams, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = np.linspace(0.0, fd.k_jam_vpk_lane * 0.999, n)
    vs = np.array([fd_speed_kmh(float(k), fd) for k in ks])
    qs = np.array([fd_flow_lane(float(k), fd) for k in ks])
    return ks, vs, qs
