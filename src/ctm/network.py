from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class LinkSpec:
    od_id: str
    name: str
    origin: str
    destination: str
    distance_km: float


@dataclass(frozen=True)
class NetworkGeometry:
    """Discretization and ramp placement for one link."""

    link: LinkSpec
    cell_len_km: float
    n_cells: int
    off_ramp_cells: tuple[int, ...]
    on_ramp_cells: tuple[int, ...]


def load_network_json(path: str | Path) -> list[LinkSpec]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    links = raw.get("links", [])
    out: list[LinkSpec] = []
    for item in links:
        out.append(
            LinkSpec(
                od_id=str(item["od_id"]),
                name=str(item.get("name", item["od_id"])),
                origin=str(item.get("origin", "")),
                destination=str(item.get("destination", "")),
                distance_km=float(item["distance_km"]),
            )
        )
    return out


def ramp_indices_for_length(n_cells: int) -> tuple[list[int], list[int]]:
    """
    Each 10-cell block (1-based indexing within the block):
    - 3rd cell -> off-ramp
    - 7th cell -> on-ramp
    Only include indices that exist (0-based cell index).
    """
    off: list[int] = []
    on: list[int] = []
    for block_start in range(0, n_cells, 10):
        third = block_start + 2  # 1-based 3rd -> 0-based +2
        seventh = block_start + 6
        if third < n_cells:
            off.append(third)
        if seventh < n_cells:
            on.append(seventh)
    return off, on


def build_geometry(link: LinkSpec, v_ff_kmh: float, dt_hours: float) -> NetworkGeometry:
    if link.distance_km <= 0:
        raise ValueError("distance_km must be positive")
    if v_ff_kmh <= 0 or dt_hours <= 0:
        raise ValueError("v_ff and dt must be positive")

    cell_len_km = v_ff_kmh * dt_hours
    n_cells = int(link.distance_km // cell_len_km)
    if n_cells < 1:
        raise ValueError(
            f"Link too short for one cell: distance={link.distance_km} km, "
            f"cell_len={cell_len_km:.4f} km"
        )

    off_idx, on_idx = ramp_indices_for_length(n_cells)
    return NetworkGeometry(
        link=link,
        cell_len_km=cell_len_km,
        n_cells=n_cells,
        off_ramp_cells=tuple(off_idx),
        on_ramp_cells=tuple(on_idx),
    )


def geometry_with_ramps(
    base: NetworkGeometry,
    on_ramp_cells: Iterable[int],
    off_ramp_cells: Iterable[int],
) -> NetworkGeometry:
    """동일 링크·이산화에 대해 on/off 램프 셀 인덱스만 교체(범위 밖 인덱스는 제외)."""
    n = base.n_cells
    on_t = tuple(sorted(int(i) for i in on_ramp_cells if 0 <= int(i) < n))
    off_t = tuple(sorted(int(i) for i in off_ramp_cells if 0 <= int(i) < n))
    return replace(base, on_ramp_cells=on_t, off_ramp_cells=off_t)


def find_link(links: Iterable[LinkSpec], od_id: str) -> LinkSpec:
    for ln in links:
        if ln.od_id == od_id:
            return ln
    raise KeyError(f"Unknown od_id: {od_id}")


def detector_indices(
    n_cells: int,
    off_ramp_cells: Iterable[int],
    on_ramp_cells: Iterable[int],
) -> tuple[int, ...]:
    """
    검지기 위치: 구간 양끝 + 모든 on/off 램프 셀 (중복 제거, 정렬).
    시뮬 로직과 무관하게 도식용 기본 배치.
    """
    if n_cells <= 0:
        return ()
    s: set[int] = {0, n_cells - 1}
    s.update(int(i) for i in off_ramp_cells)
    s.update(int(i) for i in on_ramp_cells)
    s = {i for i in s if 0 <= i < n_cells}
    return tuple(sorted(s))
