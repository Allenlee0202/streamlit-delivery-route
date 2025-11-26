# app.py
# 물류 데이터 기반 동적 다익스트라 택배 경로 분석 (Streamlit용)
# - 모든 간선 거리: 10km 고정
# - 평균 속도를 10분의 1로 줄인 v' 사용 → 각 구간 이동시간 >= 1h
# - 시간 변수는 실수형(시간 단위), 표시할 때만 반올림해서 출력
# - 가중치 w = (거리 / v') * 10  +  (혼잡도 × 교통량 × 연료) / 1000

import streamlit as st
import pandas as pd
import heapq
import math
import re
from datetime import datetime

# ==========================================
# 전역 설정
# ==========================================

FUEL_PER_HOUR = 1.0      # 시간당 연료 소비량 (고정 상수)
DEFAULT_SPEED = 80.0     # 속도 정보 없을 때 기본값 (km/h)
INF = float("inf")

# 모든 간선(구간)의 거리를 10km로 고정
EDGE_DISTANCE_KM = 10.0

# --- 엑셀 열 이름 (네 파일에 맞게 필요하면 수정) ---
SEGMENT_COL = "콘존명"      # ex) "구서IC-영락IC"
TIME_COL = "측정시각"       # datetime 이거나 0~23 정수 또는 문자열
VOLUME_COL = "평균교통량"
SPEED_COL = "평균속도"
CONG_COL = "혼잡빈도수"


# ==========================================
# 1. 측정시각 파싱 유틸
# ==========================================

def parse_hour_cell(x):
    """
    셀 하나를 받아서 hour(0~23)로 최대한 뽑아내는 함수.
    - datetime 객체: .hour
    - 숫자: int(x) % 24
    - 문자열:
        * datetime 파싱 시도
        * 안 되면, 문자열 안 첫 번째 정수(0~23)를 hour로 사용
    - 실패하면 None 반환
    """
    if pd.isna(x):
        return None

    # datetime 타입
    if isinstance(x, (datetime, pd.Timestamp)):
        return int(x.hour) % 24

    # 숫자형
    if isinstance(x, (int, float)):
        if math.isnan(x):
            return None
        return int(x) % 24

    # 문자열
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None

        # 1) datetime 파싱 (예: '2024-01-01 03:00', '03:00' 등)
        dt = pd.to_datetime(s, errors="coerce")
        if not pd.isna(dt):
            return int(dt.hour) % 24

        # 2) 문자열 안에서 1~2자리 숫자 찾기
        m = re.search(r"(\d{1,2})", s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return h

        return None

    return None


# ==========================================
# 2. 엑셀 전처리 (안 돌아가는 일 없게)
# ==========================================

def preprocess_excel(df: pd.DataFrame):
    """
    업로드된 엑셀에서:
    - TIME_COL(측정시각)에서 hour(0~23) 추출 (실패해도 강제로 hour 생성)
    - 필수 열 존재 여부 확인 (없으면 최대한 만들어서라도 진행)
    - 수치형 열(평균교통량, 평균속도, 혼잡빈도수) 숫자로 변환 (이상한 값은 0)
    - SEGMENT_COL이 비어 있는 행만 제거
    """
    df = df.copy()

    # 1) 측정시각 처리
    if TIME_COL in df.columns:
        # 각 셀에서 hour 파싱 시도
        hours = df[TIME_COL].apply(parse_hour_cell)
        valid_count = hours.notna().sum()

        if valid_count == 0:
            # 전부 파싱 실패 → index % 24 로 대체
            st.warning(
                "측정시각을 hour(0~23)로 해석하지 못했습니다. "
                "행 번호(index) % 24 값을 시간대(hour)로 사용합니다."
            )
            df["hour"] = [int(i % 24) for i in range(len(df))]
        else:
            # 파싱된 값은 그대로, 나머지는 0시로 채우기
            st.info(
                f"측정시각 {len(df)}행 중 {valid_count}행에서 시간 정보를 추출했습니다. "
                "추출되지 않은 행은 0시로 처리했습니다."
            )
            df["hour"] = hours.fillna(0).astype(int) % 24
    else:
        # TIME_COL 자체가 없음 → 그냥 index % 24 로 시간 생성
        st.warning(
            f"엑셀에 '{TIME_COL}' 열이 없습니다. "
            "열이 없어도 실행되도록, 행 번호(index) % 24를 시간(hour)로 사용합니다."
        )
        df["hour"] = [int(i % 24) for i in range(len(df))]

    # 2) 필수 열이 없으면 최대한 만들어 준다.
    # 콘존명 없으면 더 이상 진행 불가능 → 에러
    if SEGMENT_COL not in df.columns:
        st.error(f"엑셀에 '{SEGMENT_COL}' 열이 없습니다. 콘존명(예: 구서IC-영락IC)을 포함해야 합니다.")
        return None

    # 나머지 수치형 열이 없으면 0으로 채운 열을 새로 만든다.
    for col in [VOLUME_COL, SPEED_COL, CONG_COL]:
        if col not in df.columns:
            st.warning(f"엑셀에 '{col}' 열이 없어 0으로 채운 열을 생성했습니다.")
            df[col] = 0

    # 3) 수치형 열을 숫자로 강제 변환 (이상한 값은 NaN → 0)
    for col in [VOLUME_COL, SPEED_COL, CONG_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4) 콘존명이 비어 있는 행만 제거
    df[SEGMENT_COL] = df[SEGMENT_COL].astype(str)
    df = df[df[SEGMENT_COL].str.strip() != ""]
    if len(df) == 0:
        st.error("콘존명이 비어 있거나 잘못되어, 유효한 행이 없습니다.")
        return None

    return df


# ==========================================
# 3. 기타 유틸 함수
# ==========================================

def extract_nodes_from_segments(df: pd.DataFrame):
    """
    콘존명(예: '구서IC-영락IC')에서 모든 지점 이름을 뽑는다.
    """
    nodes = set()
    for seg in df[SEGMENT_COL].unique():
        if not isinstance(seg, str):
            continue
        seg = seg.replace("–", "-")
        parts = seg.split("-")
        for p in parts:
            name = p.strip()
            if name:
                nodes.add(name)
    return sorted(nodes)


# ==========================================
# 4. 그래프 및 시간대별 데이터 생성
# ==========================================

def build_graph_and_time_data(df: pd.DataFrame):
    """
    엑셀에서 다음 정보를 만든다.
    - graph : {노드: [이웃노드 리스트]}
    - segment_to_nodes : {"구서IC-영락IC": ("구서IC","영락IC"), ...}
    - distance_for_segment : {segment: 거리(여기서는 모두 10km 고정)}
    - congestion_by_hour : {(segment, hour): 혼잡도 합}
    - throughput_by_hour : {(segment, hour): 교통량 합}
    - speed_by_hour : {(segment, hour): 0이 아닌 속도 평균}
    """
    graph = {}
    segment_to_nodes = {}
    distance_for_segment = {}

    # (1) 콘존명 파싱해서 노드와 간선 만들기
    for seg in df[SEGMENT_COL].unique():
        if not isinstance(seg, str):
            continue
        seg_clean = seg.replace("–", "-")
        parts = [p.strip() for p in seg_clean.split("-")]
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]

        # 양방향 그래프
        graph.setdefault(a, [])
        graph.setdefault(b, [])
        if b not in graph[a]:
            graph[a].append(b)
        if a not in graph[b]:
            graph[b].append(a)

        segment_to_nodes[seg_clean] = (a, b)
        # ★ 모든 간선 거리 10km로 고정
        distance_for_segment[seg_clean] = EDGE_DISTANCE_KM

    if not graph:
        st.error("콘존명에서 유효한 구간을 하나도 찾지 못했습니다. 콘존명 형식을 확인하세요. (예: 구서IC-영락IC)")
        return None, None, None, None, None, None

    # (2) 시간대별 집계
    congestion_by_hour = {}
    throughput_by_hour = {}
    speed_by_hour = {}

    grouped = df.groupby([SEGMENT_COL, "hour"])

    for (seg, h), sub in grouped:
        if not isinstance(seg, str):
            continue
        seg_clean = seg.replace("–", "-")
        hour_int = int(h)

        cong_series = pd.to_numeric(sub[CONG_COL], errors="coerce").fillna(0)
        vol_series = pd.to_numeric(sub[VOLUME_COL], errors="coerce").fillna(0)
        speed_series = pd.to_numeric(sub[SPEED_COL], errors="coerce").fillna(0)

        C_e = cong_series.sum()
        tp = vol_series.sum()
        speeds = speed_series[speed_series > 0]
        if len(speeds) > 0:
            v = float(speeds.mean())
        else:
            v = DEFAULT_SPEED

        congestion_by_hour[(seg_clean, hour_int)] = float(C_e)
        throughput_by_hour[(seg_clean, hour_int)] = float(tp)
        speed_by_hour[(seg_clean, hour_int)] = v

    return graph, segment_to_nodes, distance_for_segment, \
        congestion_by_hour, throughput_by_hour, speed_by_hour


# ==========================================
# 5. 시간대별 가중치 계산 (새로운 정의)
# ==========================================

def edge_weight(segment: str, time_float: float,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour):
    """
    간선(segment)을 현재 시간 time_float(실수 시간, 단위: h)에 통과할 때의 가중치 w_e를 계산.

    - 시간대(hour bucket)는 int(time_float) % 24 로 결정
    - d_e : 거리 (여기서는 항상 10km)
    - v_raw : 평균속도
    - v' = v_raw / 10  (10분의 1배 속도)
    - 시간 비용(time_cost) = d_e / v'
    - 교통 비용(traffic_cost) = C_e(h) * (tp(h) * FUEL_PER_HOUR)
    - 최종 가중치 w = time_cost * 10 + traffic_cost / 1000

    반환:
    - w      : 간선 가중치
    - v_eff  : v' (실제 계산에 사용한 유효 속도)
    - d_e    : 거리(10km)
    - t_edge : 이 구간을 지나는 데 걸리는 시간 (시간 단위, = time_cost)
    """
    hour_bucket = int(time_float) % 24

    d_e = distance_for_segment.get(segment, EDGE_DISTANCE_KM)
    C_e = congestion_by_hour.get((segment, hour_bucket), 0.0)
    tp = throughput_by_hour.get((segment, hour_bucket), 0.0)
    v_raw = speed_by_hour.get((segment, hour_bucket), DEFAULT_SPEED)

    # 평균 속도를 10분의 1배로 줄인 유효 속도
    v_eff = v_raw / 10.0
    if v_eff <= 0:
        v_eff = 1e-6

    # 시간 비용 (이 값이 t_edge와 동일)
    time_cost = d_e / v_eff   # 단위: 시간(h) → 항상 1h 이상이 되도록 설계

    # 교통 비용
    traffic_cost = C_e * (tp * FUEL_PER_HOUR)

    # 최종 가중치
    w = time_cost * 10.0 + traffic_cost / 1000.0

    t_edge = time_cost
    return w, v_eff, d_e, t_edge


# ==========================================
# 6. 시간 의존 다익스트라 알고리즘 (시간=실수)
# ==========================================

def dijkstra_with_time(start: str, end: str, start_hour: int,
                       graph,
                       segment_to_nodes,
                       distance_for_segment,
                       congestion_by_hour,
                       throughput_by_hour,
                       speed_by_hour):
    """
    상태를 (노드, 실수 시간 t)으로 가지는 다익스트라 알고리즘.
    - 내부적으로는 hour bucket = int(t) % 24 를 사용해 24개 슬롯으로 관리.
    - d[node][bucket] : 해당 시간대(bucket)에 그 노드에 도착했을 때 최소 비용
    - time_at[node][bucket] : 그 상태에서의 실제 시간 t (실수)
    - prev[node][bucket] : (이전 노드, 이전 시간 t_prev, 사용한 segment)
    """
    if start not in graph or end not in graph:
        return None

    t0 = float(start_hour)  # 시작 시간(실수)
    b0 = int(t0) % 24       # 시작 시간대(bucket)

    d = {node: [INF] * 24 for node in graph}
    time_at = {node: [None] * 24 for node in graph}
    prev = {node: [None] * 24 for node in graph}

    d[start][b0] = 0.0
    time_at[start][b0] = t0

    # 우선순위 큐: (비용, 노드, 현재시간 t)
    pq = [(0.0, start, t0)]

    while pq:
        cost, node, t = heapq.heappop(pq)
        bucket = int(t) % 24

        # 이미 더 좋은 경로가 있으면 스킵
        if cost > d[node][bucket]:
            continue
        if time_at[node][bucket] is None or abs(time_at[node][bucket] - t) > 1e-9:
            # 이 bucket 상태는 더 좋은 시간으로 이미 업데이트된 상태
            continue

        for nxt in graph[node]:
            seg1 = f"{node}-{nxt}"
            seg2 = f"{nxt}-{node}"
            if seg1 in segment_to_nodes:
                seg = seg1
            elif seg2 in segment_to_nodes:
                seg = seg2
            else:
                continue

            # 현재 시간 t에서 이 간선을 지날 때의 가중치와 소요 시간
            w, v_eff, d_e, t_edge = edge_weight(
                seg, t,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour
            )

            t_new = t + t_edge           # 실수 시간
            bucket_new = int(t_new) % 24 # 새 시간대

            new_cost = cost + w
            if new_cost < d[nxt][bucket_new]:
                d[nxt][bucket_new] = new_cost
                time_at[nxt][bucket_new] = t_new
                prev[nxt][bucket_new] = (node, t, seg)
                heapq.heappush(pq, (new_cost, nxt, t_new))

    # end 노드의 0~23시 중 최소 비용 찾기
    best_bucket = None
    best_cost = INF
    best_time = None
    for b in range(24):
        if d[end][b] < best_cost and time_at[end][b] is not None:
            best_cost = d[end][b]
            best_bucket = b
            best_time = time_at[end][b]

    if best_cost == INF or best_bucket is None:
        return None

    # 경로 역추적 (실제 시간 t 사용)
    path_info = []
    node = end
    bucket = best_bucket
    t = best_time

    while prev[node][bucket] is not None:
        prev_node, t_prev, seg = prev[node][bucket]

        # prev에서 저장한 t_prev 를 사용해 다시 간선 정보 계산
        w, v_eff, d_e, t_edge = edge_weight(
            seg, t_prev,
            distance_for_segment,
            congestion_by_hour,
            throughput_by_hour,
            speed_by_hour
        )

        # 이 구간의 start_time = t_prev, end_time = t_prev + t_edge
        t_start = t_prev
        t_end = t_prev + t_edge

        C_e = congestion_by_hour.get((seg.replace("–", "-"),
                                      int(t_prev) % 24), 0.0)
        tp = throughput_by_hour.get((seg.replace("–", "-"),
                                     int(t_prev) % 24), 0.0)

        path_info.append({
            "from": prev_node,
            "to": node,
            "segment": seg,
            "start_time": t_start,   # 실수 시간(h)
            "end_time": t_end,       # 실수 시간(h)
            "distance": d_e,
            "speed": v_eff,          # 10분의 1로 줄인 유효 속도
            "congestion": C_e,
            "throughput": tp,
            "weight": w
        })

        node = prev_node
        t = t_prev
        bucket = int(t) % 24

    path_info.reverse()
    return best_cost, best_time, path_info


# ==========================================
# 7. 출력용 보조 함수 (시간 포맷 등)
# ==========================================

def format_time_hhmm(t: float) -> str:
    """
    실수 시간 t (단위: 시간)를 "HH시 MM분" 형식 문자열로 변환.
    24시간 주기로 표시.
    """
    if t is None or math.isnan(t):
        return "-"
    if t < 0:
        t = 0
    # 하루(24h) 주기
    t_mod = t % 24.0
    hour = int(t_mod)
    minute = int(round((t_mod - hour) * 60))
    if minute == 60:
        minute = 0
        hour = (hour + 1) % 24
    return f"{hour}시 {minute:02d}분"


def path_to_string(path_info):
    """A → B → C 형태 문자열."""
    if not path_info:
        return ""
    nodes = [path_info[0]["from"]]
    for seg in path_info:
        nodes.append(seg["to"])
    return " → ".join(nodes)


def path_to_dataframe(path_info):
    """구간별 정보를 DataFrame으로 변환."""
    rows = []
    for seg in path_info:
        rows.append({
            "구간": f"{seg['from']}-{seg['to']}",
            "출발 시각": format_time_hhmm(seg["start_time"]),
            "도착 시각": format_time_hhmm(seg["end_time"]),
            "거리(km)": round(seg["distance"], 2),
            "유효 속도 v'(km/h)": round(seg["speed"], 2),
            "혼잡도": round(seg["congestion"], 2),
            "교통량": round(seg["throughput"], 2),
            "간선 비용": round(seg["weight"], 2),
        })
    return pd.DataFrame(rows)


def hour_str_from_time(t: float) -> str:
    """최종 도착 시간 t(실수)를 'HH시 MM분'으로."""
    return format_time_hhmm(t)


# ==========================================
# 8. Streamlit 메인 앱
# ==========================================

def main():
    st.set_page_config(
        page_title="동적 다익스트라 택배 경로 분석",
        layout="wide"
    )

    st.title("🚚 물류 데이터 기반 동적 다익스트라 택배 경로 분석")

    st.markdown("""
    이 웹앱은 **고속도로 교통 데이터**와 **시간 의존 다익스트라 알고리즘**을 이용하여  
    택배 배송에 가장 효율적인 **출발 시각**과 **경로**를 찾는 도구입니다.

    변경된 조건:
    - 모든 구간(간선)의 거리는 10km로 가정합니다.
    - 각 구간의 평균 속도를 10분의 1로 줄인 유효 속도 v'를 사용하여,
      **각 구간 이동 시간이 1시간 이상**이 되도록 했습니다.
    - 내부 계산에서는 시간을 실수형으로 저장하고,
      화면에는 `HH시 MM분` 형식으로 반올림해서 표시합니다.
    - 간선 비용(가중치)은  
      **시간 비용(거리/속도')×10 + 교통 비용(혼잡도×교통량×연료)/1000** 으로 정의합니다.

    사용 순서:
    1. 왼쪽에서 엑셀 파일을 업로드한다.
    2. 출발 지점과 도착 지점을 선택한다.
    3. 출발 시각(필수)과 비교용 출발 시각(선택)을 지정한다.
    4. `최적 경로 계산하기` 버튼을 누르면 결과가 아래에 나타난다.
    """)

    # ---------- 사이드바 ----------
    st.sidebar.header("📂 데이터 & 설정")

    uploaded = st.sidebar.file_uploader(
        "교통 데이터 엑셀 업로드 (.xlsx)",
        type=["xlsx"],
        help="콘존명, 측정시각, 평균교통량, 평균속도, 혼잡빈도수 열을 포함하면 가장 좋지만, 없어도 돌아가게 해두었습니다."
    )

    if uploaded is None:
        st.warning("엑셀 파일을 먼저 업로드하세요.")
        st.info("예시 열 이름: 콘존명 / 측정시각 / 평균교통량 / 평균속도 / 혼잡빈도수")
        return

    try:
        df_raw = pd.read_excel(uploaded, engine="openpyxl")
    except Exception as e:
        st.error(f"엑셀을 읽는 중 오류가 발생했습니다: {e}")
        return

    df = preprocess_excel(df_raw)
    if df is None or len(df) == 0:
        return

    with st.expander("업로드된 데이터 미리보기"):
        st.dataframe(df.head(20), use_container_width=True)

    graph, segment_to_nodes, distance_for_segment, \
        congestion_by_hour, throughput_by_hour, speed_by_hour = \
        build_graph_and_time_data(df)

    if graph is None:
        return

    nodes = extract_nodes_from_segments(df)
    if len(nodes) < 2:
        st.error("인식된 지점이 너무 적습니다. 콘존명 형식을 확인하세요.")
        return

    st.sidebar.success(f"인식된 지점 수: {len(nodes)}개")

    st.sidebar.subheader("🧭 경로 설정")
    start_node = st.sidebar.selectbox("출발 지점", nodes, index=0)
    end_node = st.sidebar.selectbox("도착 지점", nodes, index=min(1, len(nodes)-1))

    st.sidebar.subheader("⏰ 출발 시각 (정수 시간)")
    start_hour_1 = st.sidebar.slider("첫 번째 출발 시각", 0, 23, 9)

    compare_mode = st.sidebar.checkbox("두 번째 출발 시각과 비교하기")
    if compare_mode:
        start_hour_2 = st.sidebar.slider("두 번째 출발 시각", 0, 23, 14)
    else:
        start_hour_2 = None

    run_btn = st.sidebar.button("🚀 최적 경로 계산하기")

    if not run_btn:
        st.info("왼쪽에서 설정을 마친 뒤 **🚀 최적 경로 계산하기** 버튼을 눌러주세요.")
        return

    # ---------- 첫 번째 시나리오 ----------
    st.header("📈 분석 결과")

    with st.spinner(f"{start_hour_1}시 출발 경로 계산 중..."):
        result1 = dijkstra_with_time(
            start_node, end_node, start_hour_1,
            graph, segment_to_nodes, distance_for_segment,
            congestion_by_hour, throughput_by_hour, speed_by_hour
        )

    if result1 is None:
        st.error(f"{start_hour_1}시 출발로는 {start_node}에서 {end_node}까지 경로를 찾을 수 없습니다.")
        return

    cost1, arrival_time1, path1 = result1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("출발 지점", start_node)
    with col2:
        st.metric("도착 지점", end_node)
    with col3:
        st.metric("예상 도착 시각", hour_str_from_time(arrival_time1))
    with col4:
        st.metric("총 비용", f"{cost1:.2f}")

    st.subheader(f"✅ {start_hour_1}시 출발 최적 경로")
    st.markdown("**경로:** " + path_to_string(path1))

    df_path1 = path_to_dataframe(path1)
    st.dataframe(df_path1, use_container_width=True)

    total_dist1 = sum(seg["distance"] for seg in path1)
    st.info(f"총 이동 거리: {total_dist1:.2f} km (모든 간선 10km 가정)")

    # ---------- 두 번째 시나리오 비교 ----------
    if compare_mode and start_hour_2 is not None:
        st.markdown("---")
        st.subheader("🔄 출발 시각 비교")

        with st.spinner(f"{start_hour_2}시 출발 경로 계산 중..."):
            result2 = dijkstra_with_time(
                start_node, end_node, start_hour_2,
                graph, segment_to_nodes, distance_for_segment,
                congestion_by_hour, throughput_by_hour, speed_by_hour
            )

        if result2 is None:
            st.error(f"{start_hour_2}시 출발 경로는 존재하지 않습니다.")
        else:
            cost2, arrival_time2, path2 = result2
            total_dist2 = sum(seg["distance"] for seg in path2)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {start_hour_1}시 출발")
                st.metric("총 비용", f"{cost1:.2f}")
                st.metric("도착 시각", hour_str_from_time(arrival_time1))
                st.metric("총 거리", f"{total_dist1:.2f} km")
            with c2:
                st.markdown(f"#### {start_hour_2}시 출발")
                st.metric("총 비용", f"{cost2:.2f}",
                          delta=f"{cost2-cost1:.2f}" if cost2 != cost1 else None)
                st.metric("도착 시각", hour_str_from_time(arrival_time2))
                st.metric("총 거리", f"{total_dist2:.2f} km")

            st.markdown("#### 💡 해석")
            if cost1 < cost2:
                st.success(
                    f"{start_hour_1}시 출발이 {start_hour_2}시 출발보다 "
                    f"비용이 {cost2-cost1:.2f}만큼 더 낮아 **더 효율적**입니다."
                )
            elif cost2 < cost1:
                st.success(
                    f"{start_hour_2}시 출발이 {start_hour_1}시 출발보다 "
                    f"비용이 {cost1-cost2:.2f}만큼 더 낮아 **더 효율적**입니다."
                )
            else:
                st.info("두 시간대의 총 비용이 동일합니다.")

    st.markdown("---")
    st.markdown("""
    ### 📘 해석 가이드
    - **간선 비용**  
      = 시간 비용(거리/속도')×10 + 교통 비용(혼잡도×교통량×연료)/1000  
      로 정의했습니다.
    - 모든 간선의 거리를 10km로 고정하고, 평균 속도를 10분의 1로 줄인 v'를 사용했기 때문에  
      **각 구간을 지날 때마다 실제 시간(실수형)이 1시간 이상 증가**합니다.
    - 테이블에는 내부 실수 시간을 `HH시 MM분`으로 반올림해 표시하므로  
      **각 구간마다 출발·도착 시각이 달라지는 동적 경로 탐색의 특징**을 잘 보여줄 수 있습니다.
    """)


if __name__ == "__main__":
    main()
