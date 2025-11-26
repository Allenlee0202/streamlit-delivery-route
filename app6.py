# app.py
# 물류 데이터 기반 동적 다익스트라 택배 경로 분석 (Streamlit용)

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

# --- 엑셀 열 이름 (네 파일에 맞게 필요하면 수정) ---
SEGMENT_COL = "콘존명"      # ex) "구서IC-영락IC"
TIME_COL = "측정시각"       # datetime 이거나 0~23 정수 또는 문자열
VOLUME_COL = "평균교통량"
SPEED_COL = "평균속도"
CONG_COL = "혼잡빈도수"

# --- 두 지점 사이 거리 (km) : 필요하면 나중에 직접 채우기 ---
DISTANCE_MAP = {
    # ("구서IC", "영락IC"): 10.5,
    # ("영락IC", "부산TG"): 15.2,
    # ("부산TG", "노포IC"):  8.3,
}


# ==========================================
# 1. 엑셀 전처리 (측정시각 파싱 강화)
# ==========================================

def parse_hour_cell(x):
    """
    셀 하나를 받아서 hour(0~23)로 최대한 뽑아내는 함수.
    - datetime 객체: .hour
    - 숫자: int(x) % 24
    - 문자열:
        * 먼저 datetime 파싱 시도
        * 안 되면, 안에 들어 있는 '숫자 1~2자리' 중 첫 번째를 hour로 사용
          예) '0~1', '3시~4시', '03:00-04:00' -> 0,3,3
    - 실패하면 None 반환
    """
    if pd.isna(x):
        return None

    # 이미 datetime 타입인 경우
    if isinstance(x, (datetime, pd.Timestamp)):
        return int(x.hour) % 24

    # 숫자형인 경우
    if isinstance(x, (int, float)):
        if math.isnan(x):
            return None
        return int(x) % 24

    # 문자열인 경우
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None

        # 1) datetime으로 한 번 더 시도 (예: '2024-01-01 03:00', '03:00' 등)
        dt = pd.to_datetime(s, errors="coerce")
        if not pd.isna(dt):
            return int(dt.hour) % 24

        # 2) 문자열 안에서 숫자 1~2자리 찾기 (첫 번째 것 사용)
        #    예: '0~1', '3시~4시', '03:00-04:00'
        m = re.search(r'(\d{1,2})', s)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return h

        return None

    # 그 외 타입은 처리하지 않음
    return None


def preprocess_excel(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    업로드된 엑셀에서:
    - TIME_COL(측정시각)에서 hour(0~23) 추출 (가능한 모든 형식 지원)
    - 필수 열 존재 여부 확인
    - 수치형 열(평균교통량, 평균속도, 혼잡빈도수) 숫자로 변환
    - 결측치 제거
    """
    df = df.copy()

    # 1) 측정시각 열 존재 확인
    if TIME_COL not in df.columns:
        st.error(f"엑셀에 '{TIME_COL}' 열이 없습니다. 열 이름을 확인하세요.")
        return None

    # 2) 각 셀을 개별적으로 hour로 파싱
    df["hour"] = df[TIME_COL].apply(parse_hour_cell)

    # hour에서 None/NaN 인 행 제거
    df = df.dropna(subset=["hour"])
    if len(df) == 0:
        st.error(
            "측정시각을 hour(0~23)로 변환할 수 있는 행이 없습니다.\n"
            "예: '0', '3', '15', '0~1', '3시~4시', '03:00-04:00' 형식처럼 "
            "시간 정보가 포함되도록 데이터를 수정해 주세요."
        )
        return None

    df["hour"] = df["hour"].astype(int)

    # 3) 필수 열 존재 여부 확인
    required = [SEGMENT_COL, VOLUME_COL, SPEED_COL, CONG_COL, "hour"]
    for col in required:
        if col not in df.columns:
            st.error(f"엑셀에 '{col}' 열이 없습니다. 열 이름을 확인하세요.")
            return None

    # 4) 수치형 열을 전부 숫자로 강제 변환 (문자 → NaN)
    for col in [VOLUME_COL, SPEED_COL, CONG_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) 필수 열에 결측치 있으면 제거
    df = df.dropna(subset=[SEGMENT_COL, VOLUME_COL, SPEED_COL, CONG_COL, "hour"])

    if len(df) == 0:
        st.error("전처리 후 남은 유효한 데이터가 없습니다. 값이 모두 비어 있거나 형식이 잘못되었을 수 있습니다.")
        return None

    return df


# ==========================================
# 1-2. 기타 유틸 함수
# ==========================================

def extract_nodes_from_segments(df: pd.DataFrame) -> list[str]:
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


def get_distance(a: str, b: str) -> float:
    """
    두 지점 사이 거리를 DISTANCE_MAP에서 찾는다.
    없으면 기본값 1.0km 사용.
    """
    if (a, b) in DISTANCE_MAP:
        return DISTANCE_MAP[(a, b)]
    if (b, a) in DISTANCE_MAP:
        return DISTANCE_MAP[(b, a)]
    return 1.0


# ==========================================
# 2. 그래프 및 시간대별 데이터 생성
# ==========================================

def build_graph_and_time_data(df: pd.DataFrame):
    """
    엑셀에서 다음 정보를 만든다.
    - graph : {노드: [이웃노드 리스트]}
    - segment_to_nodes : {"구서IC-영락IC": ("구서IC","영락IC"), ...}
    - distance_for_segment : {segment: 거리}
    - congestion_by_hour : {(segment, hour): 혼잡도 합}
    - throughput_by_hour : {(segment, hour): 교통량 합}
    - speed_by_hour : {(segment, hour): 0이 아닌 속도 평균}
    """
    graph: dict[str, list[str]] = {}
    segment_to_nodes: dict[str, tuple[str, str]] = {}
    distance_for_segment: dict[str, float] = {}

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
        distance_for_segment[seg_clean] = get_distance(a, b)

    # 그래프가 비어 있으면 오류
    if not graph:
        st.error("콘존명에서 유효한 구간을 하나도 찾지 못했습니다. 콘존명 형식을 확인하세요. (예: 구서IC-영락IC)")
        return None, None, None, None, None, None

    # (2) 시간대별 집계
    congestion_by_hour: dict[tuple[str, int], float] = {}
    throughput_by_hour: dict[tuple[str, int], float] = {}
    speed_by_hour: dict[tuple[str, int], float] = {}

    grouped = df.groupby([SEGMENT_COL, "hour"])

    for (seg, h), sub in grouped:
        if not isinstance(seg, str):
            continue
        seg_clean = seg.replace("–", "-")
        hour_int = int(h)

        # 혼잡도 합
        cong_series = pd.to_numeric(sub[CONG_COL], errors="coerce")
        C_e = cong_series.sum(skipna=True)
        congestion_by_hour[(seg_clean, hour_int)] = float(C_e)

        # 교통량 합
        vol_series = pd.to_numeric(sub[VOLUME_COL], errors="coerce")
        tp = vol_series.sum(skipna=True)
        throughput_by_hour[(seg_clean, hour_int)] = float(tp)

        # 속도: 0보다 큰 값만 평균
        speed_series = pd.to_numeric(sub[SPEED_COL], errors="coerce")
        speeds = speed_series[speed_series > 0]
        if len(speeds) > 0:
            v = float(speeds.mean())
        else:
            v = DEFAULT_SPEED
        speed_by_hour[(seg_clean, hour_int)] = v

    return graph, segment_to_nodes, distance_for_segment, \
        congestion_by_hour, throughput_by_hour, speed_by_hour


# ==========================================
# 3. 시간대별 가중치 계산
# ==========================================

def edge_weight(segment: str, hour: int,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour) -> tuple[float, float]:
    """
    간선(segment)을 hour 시각에 통과할 때의 가중치 w_e(h)를 계산.
    - d_e : 거리
    - C_e(h) : 혼잡빈도수 합
    - tp(h) : 교통량 합
    - v(h) : 속도
    - time_cost = d_e / v(h)
    - traffic_cost = C_e(h) * (tp(h)*FUEL_PER_HOUR)
    - w = d_e + time_cost + traffic_cost
    """
    h = hour % 24

    d_e = distance_for_segment.get(segment, 1.0)
    C_e = congestion_by_hour.get((segment, h), 0.0)
    tp = throughput_by_hour.get((segment, h), 0.0)
    v = speed_by_hour.get((segment, h), DEFAULT_SPEED)

    time_cost = d_e / max(v, 1e-6)
    traffic_cost = C_e * (tp * FUEL_PER_HOUR)
    w = d_e + time_cost + traffic_cost
    return w, v


# ==========================================
# 4. 시간 의존 다익스트라 알고리즘
# ==========================================

def dijkstra_with_time(start: str, end: str, start_hour: int,
                       graph,
                       segment_to_nodes,
                       distance_for_segment,
                       congestion_by_hour,
                       throughput_by_hour,
                       speed_by_hour):
    """
    상태를 (노드, 시각)으로 확장한 다익스트라 알고리즘.
    - d[node][hour] : 해당 시각에 그 노드에 도착했을 때 최소 비용
    - prev[node][hour] : 어디서 왔는지 (이전 노드, 이전 시각, 사용한 segment)
    """
    if start not in graph or end not in graph:
        return None

    h0 = start_hour % 24

    d: dict[str, list[float]] = {node: [INF] * 24 for node in graph}
    prev: dict[str, list[tuple[str, int, str] | None]] = {node: [None] * 24 for node in graph}
    d[start][h0] = 0.0

    pq: list[tuple[float, str, int]] = [(0.0, start, h0)]   # (비용, 노드, 시각)

    while pq:
        cost, node, h = heapq.heappop(pq)
        if cost > d[node][h]:
            continue

        for nxt in graph[node]:
            # segment 이름 찾아서 (양방향 둘 다 검사)
            seg1 = f"{node}-{nxt}"
            seg2 = f"{nxt}-{node}"
            if seg1 in segment_to_nodes:
                seg = seg1
            elif seg2 in segment_to_nodes:
                seg = seg2
            else:
                continue

            w, v = edge_weight(
                seg, h,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour
            )

            d_e = distance_for_segment.get(seg, 1.0)
            t_hours = d_e / max(v, 1e-6)   # 이동 시간(시간 단위)
            h2 = int((h + t_hours) % 24)

            new_cost = cost + w
            if new_cost < d[nxt][h2]:
                d[nxt][h2] = new_cost
                prev[nxt][h2] = (node, h, seg)
                heapq.heappush(pq, (new_cost, nxt, h2))

    # end 노드의 0~23시 중 최소 비용 찾기
    best_h = None
    best_cost = INF
    for h in range(24):
        if d[end][h] < best_cost:
            best_cost = d[end][h]
            best_h = h

    if best_cost == INF or best_h is None:
        return None

    # 경로 역추적
    path_info: list[dict] = []
    node = end
    h = best_h
    while prev[node][h] is not None:
        prev_node, prev_h, seg = prev[node][h]

        d_e = distance_for_segment.get(seg, 1.0)
        w, v = edge_weight(
            seg, prev_h,
            distance_for_segment,
            congestion_by_hour,
            throughput_by_hour,
            speed_by_hour
        )
        C_e = congestion_by_hour.get((seg, prev_h), 0.0)
        tp = throughput_by_hour.get((seg, prev_h), 0.0)

        path_info.append({
            "from": prev_node,
            "to": node,
            "segment": seg,
            "start_hour": prev_h,
            "end_hour": h,
            "distance": d_e,
            "speed": v,
            "congestion": C_e,
            "throughput": tp,
            "weight": w
        })

        node = prev_node
        h = prev_h

    path_info.reverse()
    return best_cost, best_h, path_info


# ==========================================
# 5. 출력용 보조 함수
# ==========================================

def path_to_string(path_info: list[dict]) -> str:
    """A → B → C 형태 문자열."""
    if not path_info:
        return ""
    nodes = [path_info[0]["from"]]
    for seg in path_info:
        nodes.append(seg["to"])
    return " → ".join(nodes)


def path_to_dataframe(path_info: list[dict]) -> pd.DataFrame:
    """구간별 정보를 DataFrame으로 변환."""
    rows = []
    for seg in path_info:
        rows.append({
            "구간": f"{seg['from']}-{seg['to']}",
            "출발 시각": f"{seg['start_hour']}시",
            "도착 시각": f"{seg['end_hour']}시",
            "거리(km)": round(seg["distance"], 2),
            "속도(km/h)": round(seg["speed"], 2),
            "혼잡도": round(seg["congestion"], 2),
            "교통량": round(seg["throughput"], 2),
            "간선 비용": round(seg["weight"], 2),
        })
    return pd.DataFrame(rows)


def hour_str(h: int) -> str:
    return f"{h}시"


# ==========================================
# 6. Streamlit 메인 앱
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
        help="콘존명, 측정시각, 평균교통량, 평균속도, 혼잡빈도수 열을 포함해야 합니다."
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

    st.sidebar.subheader("⏰ 출발 시각")
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

    cost1, arrival1, path1 = result1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("출발 지점", start_node)
    with col2:
        st.metric("도착 지점", end_node)
    with col3:
        st.metric("예상 도착 시각", hour_str(arrival1))
    with col4:
        st.metric("총 비용", f"{cost1:.2f}")

    st.subheader(f"✅ {start_hour_1}시 출발 최적 경로")
    st.markdown("**경로:** " + path_to_string(path1))

    df_path1 = path_to_dataframe(path1)
    st.dataframe(df_path1, use_container_width=True)

    total_dist1 = sum(seg["distance"] for seg in path1)
    st.info(f"총 이동 거리: {total_dist1:.2f} km")

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
            cost2, arrival2, path2 = result2
            total_dist2 = sum(seg["distance"] for seg in path2)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"#### {start_hour_1}시 출발")
                st.metric("총 비용", f"{cost1:.2f}")
                st.metric("도착 시각", hour_str(arrival1))
                st.metric("총 거리", f"{total_dist1:.2f} km")
            with c2:
                st.markdown(f"#### {start_hour_2}시 출발")
                st.metric("총 비용", f"{cost2:.2f}",
                          delta=f"{cost2-cost1:.2f}" if cost2 != cost1 else None)
                st.metric("도착 시각", hour_str(arrival2))
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
    - **간선 비용** = 거리 + 시간 비용(거리/속도) + 교통 비용(혼잡도 × 교통량 × 연료)
    - 비용이 작을수록, 해당 경로와 시간대가 더 효율적인 배송 조건을 의미합니다.
    - 실제 수행평가 보고서에서는
      1) 데이터 수집 과정,
      2) 시간대별 혼잡도·속도 패턴,
      3) 다익스트라 알고리즘 설계,
      4) 두 출발 시각 비교 결과
      순서로 정리하면 좋습니다.
    """)


if __name__ == "__main__":
    main()