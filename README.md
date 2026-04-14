# CTM 웹 시뮬레이터

Daganzo식 CTM(Cell Transmission Model)을 Python으로 계산하고 Streamlit으로 파라미터 입력·결과 시각화를 제공합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## 단위

- `dt`: 시간 스텝 (시간 단위로 내부 변환, UI에서는 분 단위 입력 가능)
- `v_ff`, `w`: km/h
- `q_max`: veh/h/lane
- `k_jam`: veh/km/lane
- OD 거리: km (`data/od_network.json`의 `distance_km`)
- 셀 길이: `cell_len_km = v_ff * dt_h`
- 셀 개수: `floor(distance_km / cell_len_km)`

FD 상수는 **per lane**이며, Send/Receive/q_cap 한계는 **레인 수 `n_lanes`를 곱한 총도로 기준**으로 스케일합니다.

## OD JSON 스키마

`data/od_network.json`:

- `links`: 배열
  - `od_id`: 문자열 식별자
  - `name`: 표시용 이름
  - `origin`, `destination`: 선택 표시용
  - `distance_km`: 링크 길이 (필수, 시뮬에 사용)

## 램프 규칙

10셀마다 블록을 나누고, 각 블록의 **1-based 3번째 셀**에 off-ramp, **7번째 셀**에 on-ramp를 둡니다(셀이 실제로 존재할 때만).

## 기본 파라미터 (앱 기본값과 동일)

- `dt` = 5분, `v_ff` = 100 km/h, `w` = 15 km/h, `q_max` = 2000 veh/h/lane, `k_jam` = 200 veh/km/lane
- off-ramp 분기율 β = 0.1
- Off-peak: upstream 4500 veh/h, on-ramp 750 veh/h
- Peak: upstream 6000 veh/h, on-ramp 1200 veh/h
- Peak 적용 시각(예): 60–120분 — 스텝 시작 시각 `t = k·dt`가 이 구간에 있을 때만 Peak 유량, 그 외에는 Off-peak(차량 상태는 연속 누적).

## 확장

사고(차단) 기능은 추후 `simulation`에 인접 셀 유량 제한 훅으로 추가할 수 있습니다.
