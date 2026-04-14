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


def _render_ramp_cell_buttons(n_cells: int, layout_key: str) -> None:
    """셀마다 ON / OFF 토글: 켜진 항목 다시 누르면 해제. ON과 OFF 동시 불가."""
    ncols = min(8, max(1, n_cells))
    for row0 in range(0, n_cells, ncols):
        cols = st.columns(ncols)
        for j in range(ncols):
            i = row0 + j
            if i >= n_cells:
                break
            with cols[j]:
                on_act = i in st.session_state["custom_on_cells"]
                off_act = i in st.session_state["custom_off_cells"]
                st.caption(f"셀 {i}")
                b_on = st.button(
                    "ON",
                    key=f"rpon_{layout_key}_{i}",
                    type="primary" if on_act else "secondary",
                )
                b_off = st.button(
                    "OFF",
                    key=f"rpoff_{layout_key}_{i}",
                    type="primary" if off_act else "secondary",
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


# 네트워크 도식: 한 줄에 전체 셀 균등 배치(가로 스크롤 없음)
_DIAGRAM_GAP_PX = 6
_DIAGRAM_CELL_H_PX = 92


def _network_diagram_html(
    n_cells: int,
    on_cells: frozenset[int],
    off_cells: frozenset[int],
    detector_cells: frozenset[int],
    accident_cells: frozenset[int],
) -> str:
    """가로 셀 도식(흰 배경): 한 화면에 균등 너비로 전체 표시, 가로 스크롤 없음."""
    gap = _DIAGRAM_GAP_PX
    cell_h = _DIAGRAM_CELL_H_PX
    parts: list[str] = []
    parts.append(
        '<div style="background:#ffffff;border-radius:12px;padding:12px 14px 16px;'
        'border:1px solid #e2e8f0;box-sizing:border-box;width:100%;">'
    )
    parts.append(
        '<p style="margin:0 0 10px 0;color:#334155;font-size:13px;line-height:1.55;">'
        '<span style="color:#15803d;font-weight:700;">━</span> 녹색 테두리 = ON 램프 · '
        '<span style="color:#0e7490;font-weight:700;">━</span> 청록 테두리 = OFF 램프 · '
        '<span style="color:#ca8a04;font-weight:700;">●</span> 노란 점 = 검지기 · '
        '<span style="color:#dc2626;font-weight:700;">▣</span> 붉은 음영 = 사고 (선택 시)</p>'
    )
    parts.append(
        f'<div style="display:flex;flex-direction:row;flex-wrap:nowrap;gap:{gap}px;'
        'width:100%;box-sizing:border-box;min-width:0;">'
    )
    for i in range(n_cells):
        is_on = i in on_cells
        is_off = i in off_cells
        is_det = i in detector_cells
        is_acc = i in accident_cells
        bg = "rgba(254,202,202,0.95)" if is_acc else "#f8fafc"
        if is_on and is_off:
            border = "3px solid #a855f7"
        elif is_on:
            border = "3px solid #16a34a"
        elif is_off:
            border = "3px solid #0891b2"
        else:
            border = "1px solid #cbd5e1"
        det_html = ""
        if is_det:
            det_html = (
                '<div style="position:absolute;top:6px;left:50%;transform:translateX(-50%);'
                'width:10px;height:10px;background:#eab308;border-radius:50%;'
                'box-shadow:0 0 0 1px rgba(0,0,0,0.15);"></div>'
            )
        parts.append(
            f'<div style="position:relative;flex:1 1 0;min-width:0;height:{cell_h}px;'
            f"border-radius:10px;background:{bg};border:{border};box-sizing:border-box;\">"
            f"{det_html}"
            '<div style="position:absolute;bottom:8px;left:0;right:0;text-align:center;'
            'font-size:clamp(11px,2.6vw,15px);font-weight:700;color:#0f172a;'
            'font-family:system-ui,sans-serif;overflow:hidden;text-overflow:ellipsis;">'
            f"{i}</div></div>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def _network_diagram_iframe(html_fragment: str, n_cells: int) -> None:
    """Streamlit markdown은 flex/HTML이 깨지는 경우가 있어 iframe으로 렌더."""
    doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8"/>'
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        "<style>html,body{margin:0;padding:0;background:#ffffff;overflow-x:hidden;}"
        "body{padding:4px 6px 8px;box-sizing:border-box;width:100%;}</style></head><body>"
        f"{html_fragment}</body></html>"
    )
    est_h = max(240, min(560, 150 + _DIAGRAM_CELL_H_PX + 36))
    # width=None → 컨테이너 너비에 맞춤(가로 스크롤 없이 한 줄)
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
        yaxis_title="거리 (km, 0 = 상류·아래)",
        height=420,
        margin=dict(l=60, r=20, t=50, b=50),
    )
    fig.update_yaxes(autorange=False, range=[0.0, ymax])
    return fig


def main() -> None:
    st.set_page_config(page_title="CTM 시뮬레이터", layout="wide")
    st.title("CTM 웹 시뮬레이터 (Daganzo식)")

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
        st.header("상수 (per lane)")
        dt_min = st.number_input("dt (분)", min_value=0.5, max_value=60.0, value=5.0, step=0.5)
        v_ff = st.number_input("v_ff (km/h)", min_value=10.0, value=100.0, step=5.0)
        w = st.number_input("w (km/h)", min_value=1.0, value=15.0, step=1.0)
        q_max = st.number_input("q_max (veh/h/lane)", min_value=100.0, value=2000.0, step=100.0)
        k_jam = st.number_input("k_jam (veh/km/lane)", min_value=10.0, value=200.0, step=5.0)
        n_lanes = st.slider("레인 수", min_value=1, max_value=8, value=3, step=1)
        beta = st.slider("Off-ramp 분기율 β", min_value=0.0, max_value=0.99, value=0.1, step=0.01)
        t_horizon_min = st.number_input("시뮬 시간 (분)", min_value=10.0, value=180.0, step=10.0)

        q_cap = _q_cap_total(q_max, n_lanes)
        st.caption(f"총 도로 용량 상한(참고): q_max×레인 ≈ {q_cap:.0f} veh/h")

        st.divider()
        st.subheader("수요 (veh/h)")
        up_max_off = max(q_cap * 0.999, 1.0)
        ramp_max_off = max(q_cap * 0.999, 1.0)
        up_max_pk = max(q_cap * 2.5, q_cap + 100.0)
        ramp_max_pk = max(q_cap * 2.5, q_cap + 100.0)
        st.caption(
            "Off-peak·Peak 유량을 각각 설정합니다. 시뮬은 **끊김 없이 누적**되며, "
            "아래 Peak 구간(닫힌 구간) 중 **어느 하나에라도** 속할 때만 Peak 유량이 적용됩니다."
        )
        if st.button("수요 기본값 (Off 4500/750, Peak 6000/1200, 구간 60–120분)"):
            st.session_state["q_off_up"] = 4500.0
            st.session_state["q_off_r"] = 750.0
            st.session_state["q_pk_up"] = 6000.0
            st.session_state["q_pk_r"] = 1200.0
            st.session_state["peak_windows_text"] = "60-120"
            st.rerun()

        st.markdown("**Off-peak** (그 외 시간)")
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
            st.warning("Off-peak는 q_max×레인 **미만**이 되도록 조절하세요.")

        st.markdown("**Peak** (지정 구간만)")
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

        st.markdown("**Peak 구간 (분, 닫힌 구간)** · 스텝 시작 `t = k·dt`")
        st.session_state.setdefault("peak_windows_text", "60-120")
        peak_windows_text = st.text_input(
            "구간 목록 (쉼표로 구분, 각 구간은 시작-끝)",
            key="peak_windows_text",
            help="예: 60-120 또는 60-120, 180-220 (여러 피크)",
        )
        peak_windows = parse_peak_windows(peak_windows_text)
        if not peak_windows:
            st.info("구간이 비어 있으면 전 시간 Off-peak 유량만 사용됩니다.")
        else:
            parts = [f"{lo:g}–{hi:g}" for lo, hi in peak_windows]
            st.caption(f"Peak 적용: **t ∈** " + " **∪** ".join(parts) + " 분 · 그 외 Off-peak (상태 연속)")

    fd = FDParams(v_ff_kmh=v_ff, w_kmh=w, q_max_vph_lane=q_max, k_jam_vpk_lane=k_jam)

    accident_events: tuple[AccidentSpec, ...] = tuple()
    accident_cells_for_diagram: frozenset[int] = frozenset()
    col_net, col_run = st.columns((1, 2))
    with col_net:
        st.subheader("네트워크")
        od_id = st.selectbox("OD 선택", od_ids, index=0)
        link = find_link(links, od_id)
        st.caption(f"{link.name} · 거리 {link.distance_km:g} km")

        geom_base = build_geometry(link, v_ff_kmh=v_ff, dt_hours=dt_min / 60.0)
        rk = _ramp_layout_key(od_id, geom_base)
        _sync_ramp_session(rk, geom_base)
        geom = geometry_with_ramps(
            geom_base,
            tuple(st.session_state["custom_on_cells"]),
            tuple(st.session_state["custom_off_cells"]),
        )

        st.metric("셀 길이 (km)", f"{geom.cell_len_km:.4f}")
        st.metric("셀 개수", geom.n_cells)
        st.caption(
            f"OFF 램프 셀: {list(geom.off_ramp_cells)} · ON 램프 셀: {list(geom.on_ramp_cells)}"
        )

        with st.expander("램프 위치 (셀별 ON / OFF, 같은 버튼 다시 누르면 해제)", expanded=False):
            st.caption(
                "한 셀에는 ON 또는 OFF만 가능합니다. ON을 켜면 해당 셀의 OFF는 꺼지고, 그 반대도 같습니다."
            )
            if st.button("10셀 자동 규칙으로 램프 초기화", key=f"ramp_auto_reset_{rk}"):
                off_a, on_a = ramp_indices_for_length(geom_base.n_cells)
                st.session_state["custom_on_cells"] = set(on_a)
                st.session_state["custom_off_cells"] = set(off_a)
                st.rerun()
            _render_ramp_cell_buttons(geom.n_cells, rk)

        st.subheader("사고 설정 + 네트워크 도식")
        with st.expander("사고 이벤트 설정 (접기/펼치기)", expanded=False):
            st.caption("아래 사고 설정은 **시뮬레이션에 실제 반영**됩니다. (해당 셀 유효 레인 감소)")
            use_accident = st.checkbox("사고 이벤트 적용(시뮬 반영)", value=False, key=f"use_accident_{rk}")
            if use_accident:
                n_acc = st.number_input(
                    "사고 건수",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    key=f"acc_count_{rk}",
                )
                events: list[AccidentSpec] = []
                for j in range(int(n_acc)):
                    st.markdown(f"**사고 #{j + 1}**")
                    acc_cell = st.selectbox(
                        f"사고 위치 셀 #{j + 1}",
                        options=list(range(geom.n_cells)),
                        index=min(geom.n_cells - 1, geom.n_cells // 2),
                        format_func=lambda i: f"셀 {i}",
                        key=f"acc_cell_{rk}_{j}",
                    )
                    acc_start = st.number_input(
                        f"사고 시작 시각 #{j + 1} (분)",
                        min_value=0.0,
                        max_value=float(t_horizon_min),
                        value=min(60.0, float(t_horizon_min)),
                        step=1.0,
                        key=f"acc_start_{rk}_{j}",
                    )
                    acc_duration = st.number_input(
                        f"사고 지속 시간 #{j + 1} (분)",
                        min_value=0.0,
                        max_value=float(t_horizon_min),
                        value=30.0,
                        step=1.0,
                        key=f"acc_dur_{rk}_{j}",
                    )
                    acc_blocked = st.slider(
                        f"막힌 레인 수 #{j + 1}",
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
                st.success(f"적용 예정 사고 {len(accident_events)}건")
                for j, ev in enumerate(accident_events, start=1):
                    st.caption(
                        f"#{j}: 셀 {ev.cell_idx}, 시작 {ev.start_min:g}분, "
                        f"지속 {ev.duration_min:g}분, 차단 {ev.blocked_lanes}개 차로"
                    )

        on_set = frozenset(geom.on_ramp_cells)
        off_set = frozenset(geom.off_ramp_cells)
        det_set = frozenset(
            detector_indices(geom.n_cells, geom.off_ramp_cells, geom.on_ramp_cells)
        )
        _network_diagram_iframe(
            _network_diagram_html(
                geom.n_cells,
                on_set,
                off_set,
                det_set,
                accident_cells_for_diagram,
            ),
            geom.n_cells,
        )

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
        st.subheader("실행")
        st.caption("사이드바·OD 설정이 바뀌면 매 렌더마다 CTM을 다시 계산합니다.")
        if accident_events:
            st.caption(f"사고 반영 중: 총 {len(accident_events)}건")
            for j, a in enumerate(accident_events, start=1):
                st.caption(
                    f"#{j} 셀 {a.cell_idx}, {a.start_min:g}~{(a.start_min + a.duration_min):g}분, "
                    f"차단 차로 {a.blocked_lanes}"
                )
        else:
            st.caption("사고 이벤트 없음")

    res = run_simulation(cfg)
    fd_state = fd

    k = res.k_lane
    v = res.v_kmh
    st.plotly_chart(
        _heatmap(
            k,
            "밀도 k (veh/km/lane)",
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
            "속도 v (km/h)",
            dt_min,
            geom.cell_len_km,
            zmin=0.0,
            zmax=float(fd_state.v_ff_kmh),
            colorscale="Viridis",
        ),
        use_container_width=True,
    )

    st.subheader("FD (전체 시뮬레이션)")
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
            name="시뮬 산점",
        )
    )
    fig_fd.add_trace(
        go.Scatter(x=ks_th, y=vs_th, mode="lines", name="이론 v(k)")
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
            name="시뮬 산점 q=kv",
        )
    )
    fig_qk.add_trace(go.Scatter(x=ks_th, y=qs_th, mode="lines", name="이론 q(k)"))
    fig_qk.update_layout(
        xaxis_title="k (veh/km/lane)",
        yaxis_title="q (veh/h/lane)",
        height=420,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_qk, use_container_width=True)

    st.subheader("FD analysis (혼잡 구간)")
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
            name="혼잡 v-k",
        )
    )
    fig_fda.add_trace(go.Scatter(x=ks_th, y=vs_th, mode="lines", name="이론 v(k)"))
    fig_fda.update_layout(
        title="v–k (혼잡 필터: v<0.9·v_ff 또는 k>1.05·k_free)",
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
            name="혼잡 q-k",
        )
    )
    fig_fda2.add_trace(go.Scatter(x=ks_th, y=qs_th, mode="lines", name="이론 q(k)"))
    fig_fda2.update_layout(
        title="q–k (동일 필터)",
        xaxis_title="k (veh/km/lane)",
        yaxis_title="q (veh/h/lane)",
        height=420,
    )
    st.plotly_chart(fig_fda2, use_container_width=True)


if __name__ == "__main__":
    main()
