from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctm.network import (
    build_geometry,
    detector_indices,
    find_link,
    geometry_with_ramps,
    load_network_json,
    ramp_indices_for_length,
)
from ctm.simulation import (
    AccidentSpec,
    SimulationConfig,
    fd_scatter_points,
    fd_theory_curve,
    parse_peak_windows,
    run_simulation,
)
from ctm.types import FDParams


def _q_cap_total(q_max_lane: float, n_lanes: int) -> float:
    return float(q_max_lane * n_lanes)


def _ramp_layout_key(od_id: str, geom_base) -> str:
    return f"{od_id}_{geom_base.n_cells}_{round(float(geom_base.cell_len_km), 6)}"


def _sync_ramp_session(layout_key: str, geom_base) -> None:
    if st.session_state.get("_ramp_layout_key") != layout_key:
        st.session_state["_ramp_layout_key"] = layout_key
        st.session_state["custom_on_cells"] = set(geom_base.on_ramp_cells)
        st.session_state["custom_off_cells"] = set(geom_base.off_ramp_cells)
        st.session_state["custom_det_cells"] = set(
            detector_indices(
                geom_base.n_cells,
                geom_base.off_ramp_cells,
                geom_base.on_ramp_cells,
            )
        )


def _render_ramp_cell_buttons(n_cells: int, layout_key: str) -> None:
    """셀마다 ON / OFF 토글: 켜진 항목 다시 누르면 해제. ON과 OFF 동시 불가."""
    ncols = min(5, max(1, n_cells))
    for row0 in range(0, n_cells, ncols):
        cols = st.columns(ncols)
        for j in range(ncols):
            i = row0 + j
            if i >= n_cells:
                break
            with cols[j]:
                on_act = i in st.session_state["custom_on_cells"]
                off_act = i in st.session_state["custom_off_cells"]
                st.markdown(f"**Cell {i}**")
                b_on = st.button(
                    "ON",
                    key=f"rpon_{layout_key}_{i}",
                    type="primary" if on_act else "secondary",
                    use_container_width=True,
                )
                b_off = st.button(
                    "OFF",
                    key=f"rpoff_{layout_key}_{i}",
                    type="primary" if off_act else "secondary",
                    use_container_width=True,
                )
                if b_on:
                    s_on = set(st.session_state["custom_on_cells"])
                    s_off = set(st.session_state["custom_off_cells"])
                    if i in s_on:
                        s_on.discard(i)
                    else:
                        s_on.add(i)
                        s_off.discard(i)
                    st.session_state["custom_on_cells"] = s_on
                    st.session_state["custom_off_cells"] = s_off
                    st.rerun()
                if b_off:
                    s_on = set(st.session_state["custom_on_cells"])
                    s_off = set(st.session_state["custom_off_cells"])
                    if i in s_off:
                        s_off.discard(i)
                    else:
                        s_off.add(i)
                        s_on.discard(i)
                    st.session_state["custom_on_cells"] = s_on
                    st.session_state["custom_off_cells"] = s_off
                    st.rerun()


def _render_detector_cell_buttons(n_cells: int, layout_key: str) -> None:
    """Toggle detector cells."""
    ncols = min(5, max(1, n_cells))
    for row0 in range(0, n_cells, ncols):
        cols = st.columns(ncols)
        for j in range(ncols):
            i = row0 + j
            if i >= n_cells:
                break
            with cols[j]:
                active = i in st.session_state["custom_det_cells"]
                st.markdown(f"**Cell {i}**")
                clicked = st.button(
                    "Detector",
                    key=f"det_{layout_key}_{i}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                )
                if clicked:
                    s_det = set(st.session_state["custom_det_cells"])
                    if i in s_det:
                        s_det.discard(i)
                    else:
                        s_det.add(i)
                    st.session_state["custom_det_cells"] = s_det
                    st.rerun()


# 네트워크 도식: 한 줄에 전체 셀 균등 배치(가로 스크롤 없음)
_DIAGRAM_GAP_PX = 6
_DIAGRAM_CELL_H_PX = 92


def _network_diagram_html(
    n_cells: int,
    n_lanes: int,
    on_cells: frozenset[int],
    off_cells: frozenset[int],
    detector_cells: frozenset[int],
    accident_cells: frozenset[int],
) -> str:
    """예시 이미지 스타일에 가까운 선/도형 기반 네트워크 SVG."""
    if n_cells <= 0:
        return '<div style="padding:12px;color:#475569;">No cells to display.</div>'

    cell_w = 112.0
    cell_h = 64.0
    gap = float(_DIAGRAM_GAP_PX)
    left_pad = 24.0
    right_pad = 24.0
    top_pad = 54.0
    ramp_len = 42.0
    ramp_w = 28.0
    lane_lines = max(int(n_lanes) - 1, 0)
    svg_w = left_pad + right_pad + n_cells * cell_w + max(0.0, (n_cells - 1) * gap)
    svg_h = top_pad + cell_h + 96.0

    def cx_of(idx: int) -> float:
        return left_pad + idx * (cell_w + gap) + 0.5 * cell_w

    parts: list[str] = []
    parts.append(
        '<div style="background:#ffffff;border-radius:14px;padding:10px 14px 14px;'
        'border:1px solid #dbe4f0;box-sizing:border-box;width:100%;">'
    )
    parts.append(
        '<p style="margin:0 0 8px 0;color:#334155;font-size:13px;line-height:1.5;">'
        '<span style="color:#16a34a;font-weight:700;">ON</span> ramp · '
        '<span style="color:#0ea5e9;font-weight:700;">OFF</span> ramp · '
        '<span style="color:#ca8a04;font-weight:700;">●</span> detector · '
        '<span style="color:#dc2626;font-weight:700;">⚠</span> incident cell'
        '</p>'
    )
    parts.append(
        f'<svg viewBox="0 0 {svg_w:.1f} {svg_h:.1f}" width="100%" '
        'style="display:block;overflow:visible;">'
    )
    parts.append(
        '<defs>'
        '<linearGradient id="roadBand" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#f8fbff"/>'
        '<stop offset="100%" stop-color="#edf3fb"/>'
        '</linearGradient>'
        '</defs>'
    )

    y0 = top_pad
    # 도로 배경 밴드
    parts.append(
        f'<rect x="{left_pad - 8:.1f}" y="{y0 - 8:.1f}" width="{svg_w - left_pad - right_pad + 16:.1f}" '
        f'height="{cell_h + 16:.1f}" rx="10" fill="url(#roadBand)" stroke="#d7e3f4" stroke-width="1.2"/>'
    )

    # 셀 박스와 내부 차선
    for i in range(n_cells):
        x = left_pad + i * (cell_w + gap)
        is_acc = i in accident_cells
        stroke = "#dc2626" if is_acc else "#1f2937"
        stroke_w = 2.8 if is_acc else 2.1
        parts.append(
            f'<rect x="{x:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
            f'rx="3" fill="none" stroke="{stroke}" stroke-width="{stroke_w:.1f}"/>'
        )
        if lane_lines > 0:
            for ln in range(1, lane_lines + 1):
                yy = y0 + ln * (cell_h / (lane_lines + 1))
                parts.append(
                    f'<line x1="{x + 2:.1f}" y1="{yy:.1f}" x2="{x + cell_w - 2:.1f}" y2="{yy:.1f}" '
                    'stroke="#93b5de" stroke-width="1.7" stroke-dasharray="8 5"/>'
                )
        parts.append(
            f'<text x="{x + 0.5*cell_w:.1f}" y="{y0 + cell_h + 19:.1f}" text-anchor="middle" '
            'font-size="13" font-weight="700" fill="#0f172a">'
            f'{i}</text>'
        )
        if is_acc:
            parts.append(
                f'<text x="{x + 0.5*cell_w:.1f}" y="{y0 - 10:.1f}" text-anchor="middle" '
                'font-size="14" font-weight="700" fill="#dc2626">⚠</text>'
            )

    # 검지기
    for i in detector_cells:
        if not (0 <= i < n_cells):
            continue
        cx = cx_of(i)
        cy = y0 + 8
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="#eab308" stroke="#a16207" stroke-width="1"/>')

    # 램프(요청하신 그림 느낌의 사선 가지)
    for i in on_cells:
        if not (0 <= i < n_cells):
            continue
        cx = cx_of(i)
        join_x = cx - 8
        join_y = y0 + cell_h - 2
        p2x = join_x - ramp_len
        p2y = join_y + 26
        p3x = p2x - ramp_len
        p3y = p2y + 22
        parts.append(
            f'<polyline points="{join_x:.1f},{join_y:.1f} {p2x:.1f},{p2y:.1f} {p3x:.1f},{p3y:.1f}" '
            'fill="none" stroke="#16a34a" stroke-width="2.6"/>'
        )
        parts.append(
            f'<rect x="{p3x - ramp_w*0.5:.1f}" y="{p3y - ramp_w*0.35:.1f}" '
            f'width="{ramp_w:.1f}" height="{ramp_w*0.7:.1f}" transform="rotate(-32 {p3x:.1f} {p3y:.1f})" '
            'fill="none" stroke="#0f172a" stroke-width="1.7"/>'
        )

    for i in off_cells:
        if not (0 <= i < n_cells):
            continue
        cx = cx_of(i)
        join_x = cx + 8
        join_y = y0 + cell_h - 2
        p2x = join_x + ramp_len
        p2y = join_y + 26
        p3x = p2x + ramp_len
        p3y = p2y + 22
        parts.append(
            f'<polyline points="{join_x:.1f},{join_y:.1f} {p2x:.1f},{p2y:.1f} {p3x:.1f},{p3y:.1f}" '
            'fill="none" stroke="#0ea5e9" stroke-width="2.6"/>'
        )
        parts.append(
            f'<rect x="{p3x - ramp_w*0.5:.1f}" y="{p3y - ramp_w*0.35:.1f}" '
            f'width="{ramp_w:.1f}" height="{ramp_w*0.7:.1f}" transform="rotate(32 {p3x:.1f} {p3y:.1f})" '
            'fill="none" stroke="#0f172a" stroke-width="1.7"/>'
        )

    parts.append("</svg></div>")
    return "".join(parts)


def _network_diagram_iframe(html_fragment: str, n_cells: int, height: int | None = None) -> None:
    """Streamlit markdown은 flex/HTML이 깨지는 경우가 있어 iframe으로 렌더."""
    doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8"/>'
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        "<style>html,body{margin:0;padding:0;background:#ffffff;overflow-x:hidden;}"
        "body{padding:4px 6px 8px;box-sizing:border-box;width:100%;}"
        "svg{width:100% !important;height:auto;display:block;}</style></head><body>"
        f"{html_fragment}</body></html>"
    )
    est_h = height if height is not None else max(360, min(760, 240 + _DIAGRAM_CELL_H_PX + 72))
    components.html(doc, width=None, height=est_h, scrolling=False)


def _density_colorscale() -> list[list]:
    """밀도 0 → 흰색, 증가 시 진해지는 스케일 (고정 zmin/zmax와 함께 사용)."""
    return [
        [0.0, "rgb(255,255,255)"],
        [0.12, "rgb(230,240,255)"],
        [0.28, "rgb(160,200,255)"],
        [0.48, "rgb(90,140,220)"],
        [0.68, "rgb(40,90,180)"],
        [0.85, "rgb(20,50,130)"],
        [1.0, "rgb(8,20,80)"],
    ]


def _heatmap(
    z: np.ndarray,
    title: str,
    y_dt_min: float,
    cell_len_km: float,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    colorscale: str | list | None = "Viridis",
) -> go.Figure:
    """
    입력 z: (n_time, n_cell). Plotly Heatmap은 z의 열이 x, 행이 y에 대응하므로
    transpose로 (n_cell, n_time)로 만든 뒤, flipud로 행 순서를 맞춰
    **가로=시간, 세로=거리**, 0 km(상류)가 아래에 오게 함.
    """
    nt, nx = z.shape
    z_plot = np.flipud(z.T)  # shape (nx, nt): 행=거리(위=하류), 열=시간
    x_time = np.arange(nt, dtype=float) * y_dt_min
    L = float(cell_len_km)
    y_km = (nx - 1 - np.arange(nx, dtype=float)) * L
    hm_kw: dict = dict(
        z=z_plot,
        x=x_time,
        y=y_km,
        colorscale=colorscale if colorscale is not None else "Viridis",
        colorbar=dict(title=title),
    )
    if zmin is not None:
        hm_kw["zmin"] = float(zmin)
    if zmax is not None:
        hm_kw["zmax"] = float(zmax)
    fig = go.Figure(data=go.Heatmap(**hm_kw))
    ymax = max(float((nx - 1) * L), 1e-9)
    fig.update_layout(
        title=title,
        xaxis_title="Time (min)",
        yaxis_title="Distance (km, 0 = upstream at bottom)",
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.update_yaxes(autorange=False, range=[0.0, ymax])
    return fig


def main() -> None:
    st.set_page_config(page_title="CTM Simulator", layout="wide")
    st.title("CTM Simulator")

    od_path = ROOT / "data" / "od_network.json"
    links = load_network_json(od_path)
    od_ids = [ln.od_id for ln in links]

    if "q_off_up" not in st.session_state and "q_up" in st.session_state:
        st.session_state["q_off_up"] = float(st.session_state["q_up"])
        st.session_state["q_off_r"] = float(st.session_state.get("q_ramp", 750.0))
    st.session_state.setdefault("q_off_up", 4500.0)
    st.session_state.setdefault("q_off_r", 750.0)
    st.session_state.setdefault("q_pk_up", 6000.0)
    st.session_state.setdefault("q_pk_r", 1200.0)
    st.session_state.setdefault("peak_windows_text", "60-120")

    with st.sidebar:
        st.header("Parameters (per lane)")
        dt_min = st.number_input("dt (min)", min_value=0.5, max_value=60.0, value=5.0, step=0.5)
        v_ff = st.number_input("v_ff (km/h)", min_value=10.0, value=100.0, step=5.0)
        w = st.number_input("w (km/h)", min_value=1.0, value=15.0, step=1.0)
        q_max = st.number_input("q_max (veh/h/lane)", min_value=100.0, value=2000.0, step=100.0)
        k_jam = st.number_input("k_jam (veh/km/lane)", min_value=10.0, value=200.0, step=5.0)
        n_lanes = st.slider("Number of lanes", min_value=1, max_value=8, value=3, step=1)
        beta = st.slider("Off-ramp split ratio β", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
        t_horizon_min = st.number_input("Simulation horizon (min)", min_value=10.0, value=1000.0, step=10.0)

        q_cap = _q_cap_total(q_max, n_lanes)
        st.caption(f"Mainline capacity cap: q_max × lanes ≈ {q_cap:.0f} veh/h")

        st.divider()
        st.subheader("Demand (veh/h)")
        up_max_off = max(q_cap * 0.999, 1.0)
        ramp_max_off = max(q_cap * 0.999, 1.0)
        up_max_pk = max(q_cap * 2.5, q_cap + 100.0)
        ramp_max_pk = max(q_cap * 2.5, q_cap + 100.0)
        if st.button("Reset demand defaults (Off 4500/750, Peak 6000/1200, window 60-120 min)"):
            st.session_state["q_off_up"] = 4500.0
            st.session_state["q_off_r"] = 750.0
            st.session_state["q_pk_up"] = 6000.0
            st.session_state["q_pk_r"] = 1200.0
            st.session_state["peak_windows_text"] = "60-120"
            st.rerun()

        st.markdown("**Off-peak**")
        q_off_up = st.slider(
            "Off-peak upstream",
            0.0,
            float(up_max_off),
            float(min(st.session_state["q_off_up"], up_max_off)),
            50.0,
            key="q_off_up",
        )
        q_off_r = st.slider(
            "Off-peak on-ramp",
            0.0,
            float(ramp_max_off),
            float(min(st.session_state["q_off_r"], ramp_max_off)),
            50.0,
            key="q_off_r",
        )
        if q_off_up >= q_cap - 1e-6 or q_off_r >= q_cap - 1e-6:
            st.warning("Set off-peak values below q_max × lanes.")

        st.markdown("**Peak**")
        q_pk_up = st.slider(
            "Peak upstream",
            0.0,
            float(up_max_pk),
            float(min(st.session_state["q_pk_up"], up_max_pk)),
            50.0,
            key="q_pk_up",
        )
        q_pk_r = st.slider(
            "Peak on-ramp",
            0.0,
            float(ramp_max_pk),
            float(min(st.session_state["q_pk_r"], ramp_max_pk)),
            50.0,
            key="q_pk_r",
        )

        st.markdown("**Peak windows (min, closed interval)** · step start `t = k·dt`")
        st.session_state.setdefault("peak_windows_text", "60-120")
        peak_windows_text = st.text_input(
            "Window list (comma-separated, each as start-end)",
            key="peak_windows_text",
            help="Example: 60-120 or 60-120, 180-220",
        )
        peak_windows = parse_peak_windows(peak_windows_text)
        if not peak_windows:
            st.info("No peak window: off-peak demand is used for all times.")
        else:
            parts = [f"{lo:g}–{hi:g}" for lo, hi in peak_windows]
            st.caption(f"Peak active for **t ∈** " + " **∪** ".join(parts) + " min")

    fd = FDParams(v_ff_kmh=v_ff, w_kmh=w, q_max_vph_lane=q_max, k_jam_vpk_lane=k_jam)

    accident_events: tuple[AccidentSpec, ...] = tuple()
    accident_cells_for_diagram: frozenset[int] = frozenset()
    col_net, col_run = st.columns((1, 2))
    with col_net:
        st.subheader("Network")
        od_id = st.selectbox("OD", od_ids, index=0)
        link = find_link(links, od_id)
        st.caption(f"{link.name} · distance {link.distance_km:g} km")

        geom_base = build_geometry(link, v_ff_kmh=v_ff, dt_hours=dt_min / 60.0)
        rk = _ramp_layout_key(od_id, geom_base)
        _sync_ramp_session(rk, geom_base)
        st.session_state.setdefault(
            "custom_det_cells",
            set(detector_indices(geom_base.n_cells, geom_base.off_ramp_cells, geom_base.on_ramp_cells)),
        )
        geom = geometry_with_ramps(
            geom_base,
            tuple(st.session_state["custom_on_cells"]),
            tuple(st.session_state["custom_off_cells"]),
        )

        st.metric("Cell length (km)", f"{geom.cell_len_km:.4f}")
        st.metric("Number of cells", geom.n_cells)
        st.caption(
            f"OFF-ramp cells: {list(geom.off_ramp_cells)} · ON-ramp cells: {list(geom.on_ramp_cells)}"
        )

        with st.expander("Ramp locations", expanded=False):
            if st.button("Reset ramps by 10-cell rule", key=f"ramp_auto_reset_{rk}"):
                off_a, on_a = ramp_indices_for_length(geom_base.n_cells)
                st.session_state["custom_on_cells"] = set(on_a)
                st.session_state["custom_off_cells"] = set(off_a)
                st.rerun()
            _render_ramp_cell_buttons(geom.n_cells, rk)

        with st.expander("Detector locations", expanded=False):
            if st.button("Reset detectors to default", key=f"det_reset_{rk}"):
                st.session_state["custom_det_cells"] = set(
                    detector_indices(
                        geom.n_cells,
                        st.session_state["custom_off_cells"],
                        st.session_state["custom_on_cells"],
                    )
                )
                st.rerun()
            _render_detector_cell_buttons(geom.n_cells, rk)

        st.subheader("Incidents")
        with st.expander("Incident events", expanded=False):
            use_accident = st.checkbox("Enable incidents", value=False, key=f"use_accident_{rk}")
            if use_accident:
                n_acc = st.number_input(
                    "Number of incidents",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    key=f"acc_count_{rk}",
                )
                events: list[AccidentSpec] = []
                for j in range(int(n_acc)):
                    st.markdown(f"**Incident #{j + 1}**")
                    acc_cell = st.selectbox(
                        f"Incident cell #{j + 1}",
                        options=list(range(geom.n_cells)),
                        index=min(geom.n_cells - 1, geom.n_cells // 2),
                        format_func=lambda i: f"Cell {i}",
                        key=f"acc_cell_{rk}_{j}",
                    )
                    acc_start = st.number_input(
                        f"Start time #{j + 1} (min)",
                        min_value=0.0,
                        max_value=float(t_horizon_min),
                        value=min(60.0, float(t_horizon_min)),
                        step=1.0,
                        key=f"acc_start_{rk}_{j}",
                    )
                    acc_duration = st.number_input(
                        f"Duration #{j + 1} (min)",
                        min_value=0.0,
                        max_value=float(t_horizon_min),
                        value=30.0,
                        step=1.0,
                        key=f"acc_dur_{rk}_{j}",
                    )
                    acc_blocked = st.slider(
                        f"Blocked lanes #{j + 1}",
                        min_value=1,
                        max_value=max(int(n_lanes), 1),
                        value=min(1, int(n_lanes)),
                        step=1,
                        key=f"acc_blocked_{rk}_{j}",
                    )
                    events.append(
                        AccidentSpec(
                            cell_idx=int(acc_cell),
                            start_min=float(acc_start),
                            duration_min=float(acc_duration),
                            blocked_lanes=int(acc_blocked),
                        )
                    )
                accident_events = tuple(events)
                accident_cells_for_diagram = frozenset(ev.cell_idx for ev in accident_events)
                st.success(f"{len(accident_events)} incident(s) configured.")
                for j, ev in enumerate(accident_events, start=1):
                    st.caption(
                        f"#{j}: Cell {ev.cell_idx}, start {ev.start_min:g} min, "
                        f"duration {ev.duration_min:g} min, blocked lanes {ev.blocked_lanes}"
                    )

        on_set = frozenset(geom.on_ramp_cells)
        off_set = frozenset(geom.off_ramp_cells)
        det_set = frozenset(st.session_state.get("custom_det_cells", set()))

    n_steps = max(int(t_horizon_min / dt_min), 1)
    cfg = SimulationConfig(
        geometry=geom,
        fd=fd,
        n_lanes=int(n_lanes),
        dt_minutes=float(dt_min),
        n_steps=n_steps,
        upstream_off_peak_vph=float(q_off_up),
        onramp_off_peak_vph=float(q_off_r),
        upstream_peak_vph=float(q_pk_up),
        onramp_peak_vph=float(q_pk_r),
        peak_windows=peak_windows,
        off_split_beta=float(beta),
        accidents=accident_events,
    )

    with col_run:
        st.subheader("Run")
        run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)
        if accident_events:
            st.caption(f"Incidents active: {len(accident_events)}")
            for j, a in enumerate(accident_events, start=1):
                st.caption(
                    f"#{j} Cell {a.cell_idx}, {a.start_min:g}~{(a.start_min + a.duration_min):g} min, "
                    f"blocked lanes {a.blocked_lanes}"
                )
        else:
            st.caption("No incidents")

    st.subheader("Network diagram")
    _network_diagram_iframe(
        _network_diagram_html(
            geom.n_cells,
            int(n_lanes),
            on_set,
            off_set,
            det_set,
            accident_cells_for_diagram,
        ),
        geom.n_cells,
    )

    if run_clicked:
        st.session_state["last_run_result"] = run_simulation(cfg)
        st.session_state["last_run_cfg"] = cfg
        st.session_state["last_run_meta"] = {
            "od_id": od_id,
            "n_lanes": int(n_lanes),
            "q_max": float(q_max),
            "v_ff": float(v_ff),
            "w": float(w),
            "beta": float(beta),
            "t_horizon_min": float(t_horizon_min),
            "dt_min": float(dt_min),
            "peak_windows_text": str(peak_windows_text),
            "incidents": len(accident_events),
        }

    if "last_run_result" not in st.session_state:
        st.info("Set parameters and click 'Run Simulation'.")
        return

    res = st.session_state["last_run_result"]
    fd_state = st.session_state["last_run_cfg"].fd
    run_meta = st.session_state.get("last_run_meta", {})

    k = res.k_lane
    v = res.v_kmh
    st.plotly_chart(
        _heatmap(
            k,
            "Density k (veh/km/lane)",
            dt_min,
            geom.cell_len_km,
            zmin=0.0,
            zmax=float(fd_state.k_jam_vpk_lane),
            colorscale=_density_colorscale(),
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        _heatmap(
            v,
            "Speed v (km/h)",
            dt_min,
            geom.cell_len_km,
            zmin=0.0,
            zmax=float(fd_state.v_ff_kmh),
            colorscale="Viridis",
        ),
        use_container_width=True,
    )

    st.subheader("FD (Full simulation)")
    ks_th, vs_th, qs_th = fd_theory_curve(fd_state, n=200)
    k_flat, v_flat = fd_scatter_points(k, v)
    q_flat = k_flat * v_flat

    fig_fd = go.Figure()
    fig_fd.add_trace(
        go.Scatter(
            x=k_flat,
            y=v_flat,
            mode="markers",
            marker=dict(size=4, opacity=0.25),
            name="Simulation points",
        )
    )
    fig_fd.add_trace(
        go.Scatter(x=ks_th, y=vs_th, mode="lines", name="Theory v(k)")
    )
    fig_fd.update_layout(
        xaxis_title="k (veh/km/lane)",
        yaxis_title="v (km/h)",
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_fd, use_container_width=True)

    fig_qk = go.Figure()
    fig_qk.add_trace(
        go.Scatter(
            x=k_flat,
            y=q_flat,
            mode="markers",
            marker=dict(size=4, opacity=0.25),
            name="Simulation points q=kv",
        )
    )
    fig_qk.add_trace(go.Scatter(x=ks_th, y=qs_th, mode="lines", name="Theory q(k)"))
    fig_qk.update_layout(
        xaxis_title="k (veh/km/lane)",
        yaxis_title="q (veh/h/lane)",
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_qk, use_container_width=True)

    st.subheader("FD (Detector cells only)")
    det_idx = sorted(int(i) for i in det_set if 0 <= int(i) < geom.n_cells)
    if len(det_idx) == 0:
        st.info("No detector cell selected.")
    else:
        k_det = k[:, det_idx]
        v_det = v[:, det_idx]
        k_det_flat = k_det.reshape(-1)
        v_det_flat = v_det.reshape(-1)
        q_det_flat = k_det_flat * v_det_flat

        fig_fd_det = go.Figure()
        fig_fd_det.add_trace(
            go.Scatter(
                x=k_det_flat,
                y=v_det_flat,
                mode="markers",
                marker=dict(size=4, opacity=0.3),
                name="Detector points",
            )
        )
        fig_fd_det.add_trace(
            go.Scatter(x=ks_th, y=vs_th, mode="lines", name="Theory v(k)")
        )
        fig_fd_det.update_layout(
            xaxis_title="k (veh/km/lane)",
            yaxis_title="v (km/h)",
            height=420,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_fd_det, use_container_width=True)

        fig_qk_det = go.Figure()
        fig_qk_det.add_trace(
            go.Scatter(
                x=k_det_flat,
                y=q_det_flat,
                mode="markers",
                marker=dict(size=4, opacity=0.3),
                name="Detector points q=kv",
            )
        )
        fig_qk_det.add_trace(
            go.Scatter(x=ks_th, y=qs_th, mode="lines", name="Theory q(k)")
        )
        fig_qk_det.update_layout(
            xaxis_title="k (veh/km/lane)",
            yaxis_title="q (veh/h/lane)",
            height=420,
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_qk_det, use_container_width=True)

    st.subheader("FD analysis (congested subset)")
    k_free = fd_state.q_max_vph_lane / max(fd_state.v_ff_kmh, 1e-9)
    mask = (v_flat < 0.9 * fd_state.v_ff_kmh) | (k_flat > k_free * 1.05)
    k_c = k_flat[mask]
    v_c = v_flat[mask]
    q_c = k_c * v_c
    fig_fda = go.Figure()
    fig_fda.add_trace(
        go.Scatter(
            x=k_c,
            y=v_c,
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name="Congested v-k",
        )
    )
    fig_fda.add_trace(go.Scatter(x=ks_th, y=vs_th, mode="lines", name="Theory v(k)"))
    fig_fda.update_layout(
        title="v-k (filter: v < 0.9*v_ff or k > 1.05*k_free)",
        xaxis_title="k (veh/km/lane)",
        yaxis_title="v (km/h)",
        height=420,
    )
    st.plotly_chart(fig_fda, use_container_width=True)

    fig_fda2 = go.Figure()
    fig_fda2.add_trace(
        go.Scatter(
            x=k_c,
            y=q_c,
            mode="markers",
            marker=dict(size=5, opacity=0.35),
            name="Congested q-k",
        )
    )
    fig_fda2.add_trace(go.Scatter(x=ks_th, y=qs_th, mode="lines", name="Theory q(k)"))
    fig_fda2.update_layout(
        title="q-k (same filter)",
        xaxis_title="k (veh/km/lane)",
        yaxis_title="q (veh/h/lane)",
        height=420,
    )
    st.plotly_chart(fig_fda2, use_container_width=True)

    # Whole-simulation summary metrics (exclude empty cells where k=0 / x=0)
    occupied_mask_all = res.x > 1e-9
    nonff_mask_all = occupied_mask_all & (v < 0.99 * float(fd_state.v_ff_kmh))
    congested_mask_all = occupied_mask_all & (
        (v < 0.8 * float(fd_state.v_ff_kmh)) | (k > 1.05 * k_free)
    )

    occupied_cells = int(np.sum(occupied_mask_all))
    total_cells = int(geom.n_cells)
    congested_cells = int(np.sum(congested_mask_all))
    congestion_ratio_cells = (congested_cells / occupied_cells) if occupied_cells > 0 else 0.0

    veh_total_occ = float(np.sum(res.x[occupied_mask_all])) if occupied_cells > 0 else 0.0
    veh_nonff = float(np.sum(res.x[nonff_mask_all])) if np.any(nonff_mask_all) else 0.0
    delay_vehicle_ratio = (veh_nonff / veh_total_occ) if veh_total_occ > 1e-12 else 0.0
    if np.any(nonff_mask_all) and veh_nonff > 1e-12:
        v_nonff = v[nonff_mask_all]
        w_nonff = res.x[nonff_mask_all]
        mean_nonff = float(np.average(v_nonff, weights=w_nonff))
        var_nonff = float(np.average((v_nonff - mean_nonff) ** 2, weights=w_nonff))
    else:
        mean_nonff = 0.0
        var_nonff = 0.0

    st.subheader("Whole-simulation performance summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Congested cell ratio", f"{100.0 * congestion_ratio_cells:.1f}%")
    m2.metric("Delayed vehicle ratio", f"{100.0 * delay_vehicle_ratio:.1f}%")
    m3.metric("Mean speed (non-FF)", f"{mean_nonff:.1f} km/h")
    m4.metric("Speed variance (non-FF)", f"{var_nonff:.1f}")

    if np.any(occupied_mask_all):
        v_occ = np.where(occupied_mask_all, v, np.inf)
        flat_idx = int(np.argmin(v_occ))
        t_idx, c_idx = np.unravel_index(flat_idx, v_occ.shape)
        worst_speed = float(v[t_idx, c_idx])
        worst_k = float(k[t_idx, c_idx])
        t_min_worst = float(t_idx) * float(dt_min)
        incident_active = any(
            (int(ac.cell_idx) == int(c_idx))
            and (float(ac.start_min) <= t_min_worst < float(ac.start_min) + float(ac.duration_min))
            for ac in cfg.accidents
        )
    else:
        t_idx, c_idx = 0, 0
        worst_speed = 0.0
        worst_k = 0.0
        t_min_worst = 0.0
        incident_active = False

    st.subheader("Most congested point")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Cell", f"{c_idx}")
    w2.metric("Time", f"{t_min_worst:.1f} min")
    w3.metric("Min speed", f"{worst_speed:.1f} km/h")
    w4.metric("Incident active", "Yes" if incident_active else "No")
    st.caption(f"At worst point: k={worst_k:.2f} veh/km/lane, step={t_idx}")

    c_left, c_right = st.columns((2, 3))
    with c_left:
        st.markdown("**Run configuration**")
        st.markdown(
            f"- OD: `{run_meta.get('od_id', od_id)}`\n"
            f"- Cells: `{total_cells}`\n"
            f"- Cell length: `{geom.cell_len_km:.4f} km`\n"
            f"- Lanes: `{run_meta.get('n_lanes', int(n_lanes))}`\n"
            f"- q_max: `{run_meta.get('q_max', float(q_max)):.0f} veh/h/lane`\n"
            f"- v_ff: `{run_meta.get('v_ff', float(v_ff)):.1f} km/h`, w: `{run_meta.get('w', float(w)):.1f} km/h`\n"
            f"- beta: `{run_meta.get('beta', float(beta)):.2f}`\n"
            f"- Horizon: `{run_meta.get('t_horizon_min', float(t_horizon_min)):.0f} min`, dt: `{run_meta.get('dt_min', float(dt_min)):.1f} min`\n"
            f"- Peak windows: `{run_meta.get('peak_windows_text', peak_windows_text) if peak_windows_text else 'none'}`\n"
            f"- Incidents: `{run_meta.get('incidents', len(accident_events))}`"
        )
    with c_right:
        st.markdown("**Network snapshot**")
        _network_diagram_iframe(
            _network_diagram_html(
                geom.n_cells,
                int(n_lanes),
                on_set,
                off_set,
                det_set,
                accident_cells_for_diagram,
            ),
            geom.n_cells,
            height=260,
        )


if __name__ == "__main__":
    main()
