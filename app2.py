# app.py
# 물류 데이터 기반 동적 다익스트라 택배 경로 분석 (자동화.csv를 직접 읽어서 실행)

import streamlit as st
import pandas as pd
import heapq
from typing import Dict, List, Tuple

# ============================================================================
# 전역 설정 및 상수
# ============================================================================

FUEL_PER_HOUR = 1.0   # 시간당 연료 소비량 가정값
DEFAULT_SPEED = 80.0  # 속도 정보가 없을 때 기본값 (km/h)
INF = float('inf')

DATA_CSV_PATH = "자동화.csv"  # 같은 폴더에 있는 CSV 파일 이름

# ============================================================================
# 1. 데이터 전처리 및 그래프 구축
# ============================================================================

def parse_hour(hhmm: str) -> int:
    """
    집계시분 문자열('3:15', '10:05' 등)에서 시(hour)만 정수로 추출.
    """
    try:
        h = int(str(hhmm).split(":")[0])
        return h % 24
    except Exception:
        return 0


def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    자동화.csv를 읽어서 시간대(hour) 컬럼을 추가하고,
    필요한 컬럼만 남긴 DataFrame을 반환.
    """
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)

    # 필수 컬럼 체크
    required_cols = ["집계시분", "평균교통량", "평균속도", "혼잡빈도수", "콘존명"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 열 '{col}' 이(가) CSV에 없습니다.")

    # 시간대(0~23시) 추출
    df["hour"] = df["집계시분"].map(parse_hour)

    # 결측 제거
    df = df.dropna(subset=["hour", "평균교통량", "평균속도", "혼잡빈도수", "콘존명"])

    return df


def build_graph_and_time_data(df: pd.DataFrame):
    """
    자동화.csv 데이터로부터
    - graph: {node: [neighbor_nodes]}
    - segments: 전체 segment 리스트
    - segment_to_nodes: {segment: (node_a, node_b)}
    - distance_for_segment: {segment: distance}
    - congestion_by_hour: {(segment, hour): C_e(h)}
    - throughput_by_hour: {(segment, hour): throughput(h)}
    - speed_by_hour: {(segment, hour): v(h)}
    를 계산한다.

    여기서는 모든 구간의 거리를 1.0km로 가정한다.
    """
    graph: Dict[str, List[str]] = {}
    segments: List[str] = []
    segment_to_nodes: Dict[str, Tuple[str, str]] = {}
    distance_for_segment: Dict[str, float] = {}

    # 1) 콘존명별로 노드 쌍 추출 및 그래프 구조 생성
    unique_segments = df["콘존명"].unique().tolist()

    for seg in unique_segments:
        s = str(seg)

        # '-' 또는 '~' 로 구간 나누기
        if "-" in s:
            parts = s.split("-")
        elif "~" in s:
            parts = s.split("~")
        else:
            continue

        if len(parts) < 2:
            continue

        a = parts[0].strip()
        b = parts[1].strip()
        if not a or not b:
            continue

        # 양방향 그래프 구성
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []

        if b not in graph[a]:
            graph[a].append(b)
        if a not in graph[b]:
            graph[b].append(a)

        segments.append(s)
        segment_to_nodes[s] = (a, b)

        # 거리: 현재는 모두 1km로 가정
        distance_for_segment[s] = 1.0

    # 2) 시간대별 집계: 혼잡도 합, 교통량 합, 속도 평균(0이 아닌 값만)
    congestion_by_hour: Dict[Tuple[str, int], float] = {}
    throughput_by_hour: Dict[Tuple[str, int], float] = {}
    speed_by_hour: Dict[Tuple[str, int], float] = {}

    grouped = df.groupby(["콘존명", "hour"])

    for (seg, hour), sub in grouped:
        # 혼잡빈도수 합
        C_e_h = float(sub["혼잡빈도수"].sum())
        # 평균교통량 합
        tp_h = float(sub["평균교통량"].sum())
        # 평균속도: 0이 아닌 값들의 평균
        speeds = sub["평균속도"]
        speeds_nonzero = speeds[speeds > 0]
        if len(speeds_nonzero) > 0:
            v_h = float(speeds_nonzero.mean())
        else:
            v_h = DEFAULT_SPEED

        congestion_by_hour[(seg, int(hour))] = C_e_h
        throughput_by_hour[(seg, int(hour))] = tp_h
        speed_by_hour[(seg, int(hour))] = v_h

    return (
        graph,
        segments,
        segment_to_nodes,
        distance_for_segment,
        congestion_by_hour,
        throughput_by_hour,
        speed_by_hour,
    )


# ============================================================================
# 2. 간선 가중치 계산 함수
# ============================================================================

def edge_weight(segment: str, hour: int,
                distance_for_segment: Dict[str, float],
                congestion_by_hour: Dict[Tuple[str, int], float],
                throughput_by_hour: Dict[Tuple[str, int], float],
                speed_by_hour: Dict[Tuple[str, int], float]) -> Tuple[float, float]:
    """
    시간 의존적 간선 가중치 w_e(h)를 계산한다.

    w_e(h) = d_e + time_cost + traffic_cost
      - d_e : 구간 거리 (현재 1km)
      - time_cost   = d_e / v(h)
      - traffic_cost = C_e(h) * (throughput(h) * fuel_per_hour)

    반환값: (w_e(h), v(h))
    """
    hour = hour % 24

    d_e = distance_for_segment.get(segment, 1.0)
    C_e_h = congestion_by_hour.get((segment, hour), 0.0)
    tp_h = throughput_by_hour.get((segment, hour), 0.0)
    v_h = speed_by_hour.get((segment, hour), DEFAULT_SPEED)

    # 시간 비용
    time_cost = d_e / max(v_h, 1e-6)
    # 교통 비용
    traffic_cost = C_e_h * (tp_h * FUEL_PER_HOUR)

    w = d_e + time_cost + traffic_cost
    return w, v_h


# ============================================================================
# 3. 시간 의존 다익스트라 알고리즘
# ============================================================================

def dijkstra_with_time(start: str, end: str, start_hour: int,
                       graph: Dict[str, List[str]],
                       segment_to_nodes: Dict[str, Tuple[str, str]],
                       distance_for_segment: Dict[str, float],
                       congestion_by_hour: Dict[Tuple[str, int], float],
                       throughput_by_hour: Dict[Tuple[str, int], float],
                       speed_by_hour: Dict[Tuple[str, int], float]):
    """
    상태: (node, hour)
    - 출발 시각 start_hour에서 시작해 각 구간을 통과하면서 시간이 변함
    - 가중치는 통과 시간대에 따라 동적으로 계산
    """
    if start not in graph or end not in graph:
        return None

    start_hour = start_hour % 24

    # d[node][hour] = 그 상태까지의 최소 비용
    d: Dict[str, List[float]] = {node: [INF] * 24 for node in graph}
    d[start][start_hour] = 0.0

    # prev[node][hour] = (이전 node, 이전 hour, 사용한 segment)
    prev: Dict[str, List] = {node: [None] * 24 for node in graph}

    # 우선순위 큐
    pq: List[Tuple[float, str, int]] = [(0.0, start, start_hour)]

    while pq:
        cost, node, h = heapq.heappop(pq)
        if cost > d[node][h]:
            continue

        for next_node in graph.get(node, []):
            seg1 = f"{node}-{next_node}"
            seg2 = f"{next_node}-{node}"
            seg3 = f"{node}~{next_node}"
            seg4 = f"{next_node}~{node}"

            segment = None
            if seg1 in segment_to_nodes:
                segment = seg1
            elif seg2 in segment_to_nodes:
                segment = seg2
            elif seg3 in segment_to_nodes:
                segment = seg3
            elif seg4 in segment_to_nodes:
                segment = seg4
            else:
                continue

            w, v_h = edge_weight(
                segment, h,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour,
            )

            d_e = distance_for_segment.get(segment, 1.0)
            t_hours = d_e / max(v_h, 1e-6)
            h_prime = int((h + t_hours) % 24)

            new_cost = cost + w
            if new_cost < d[next_node][h_prime]:
                d[next_node][h_prime] = new_cost
                prev[next_node][h_prime] = (node, h, segment)
                heapq.heappush(pq, (new_cost, next_node, h_prime))

    # 도착 노드의 모든 시각 중 최소 비용 찾기
    min_cost = INF
    best_hour = -1
    for h in range(24):
        if d[end][h] < min_cost:
            min_cost = d[end][h]
            best_hour = h

    if min_cost == INF:
        return None

    # 역추적
    path_info: List[Dict] = []
    curr_node = end
    curr_hour = best_hour

    while prev[curr_node][curr_hour] is not None:
        prev_node, prev_hour, segment = prev[curr_node][curr_hour]
        d_e = distance_for_segment.get(segment, 1.0)
        w, v_h = edge_weight(
            segment, prev_hour,
            distance_for_segment,
            congestion_by_hour,
            throughput_by_hour,
            speed_by_hour,
        )
        C_e = congestion_by_hour.get((segment, prev_hour), 0.0)
        tp = throughput_by_hour.get((segment, prev_hour), 0.0)

        path_info.append({
            "from": prev_node,
            "to": curr_node,
            "segment": segment,
            "start_hour": prev_hour,
            "end_hour": curr_hour,
            "distance": d_e,
            "speed": v_h,
            "congestion": C_e,
            "throughput": tp,
            "weight": w,
        })

        curr_node = prev_node
        curr_hour = prev_hour

    path_info.reverse()
    return min_cost, best_hour, path_info


# ============================================================================
# 4. UI 유틸 함수
# ============================================================================

def draw_path_summary(path_info: List[Dict]) -> str:
    if not path_info:
        return ""
    nodes = [path_info[0]["from"]]
    for seg in path_info:
        nodes.append(seg["to"])
    return " → ".join(nodes)


def draw_path_table(path_info: List[Dict]) -> pd.DataFrame:
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


def format_hour(hour: int) -> str:
    return f"{hour}시"


# ============================================================================
# 5. 메인 Streamlit 앱
# ============================================================================

def main():
    st.set_page_config(
        page_title="물류 데이터 기반 동적 다익스트라 택배 경로 분석",
        layout="wide",
    )

    st.title("🚚 물류 데이터 기반 동적 다익스트라 택배 경로 분석")

    st.markdown("""
    ### 프로젝트 개요
    이 앱은 **고속도로 교통 데이터(자동화.csv)**를 활용하여
    택배 배송에 가장 효율적인 **출발 시간**과 **경로**를 분석합니다.

    - **시간 의존적 다익스트라 알고리즘**: 시간대별로 변하는 교통 상황 반영  
    - **동적 가중치**: 거리, 속도, 혼잡도, 교통량을 종합적으로 고려  
    - **비교 분석**: 두 개의 출발 시각을 비교하여 최적 시간대 도출  

    ※ 사용자는 파일 업로드 없이, 미리 포함된 데이터로 바로 분석할 수 있습니다.
    """)

    st.sidebar.header("📋 분석 설정")
    st.sidebar.markdown("""
    **사용 방법:**
    1. 출발/도착 지점을 선택합니다.
    2. 출발 시각을 설정합니다.
    3. (선택) 두 번째 출발 시각을 비교할 수 있습니다.
    4. **최적 경로 계산하기** 버튼을 누릅니다.
    """)

    # 데이터 로드
    try:
        df_raw = load_and_preprocess_data(DATA_CSV_PATH)
    except Exception as e:
        st.error(f"데이터 로드/전처리 중 오류 발생: {e}")
        st.stop()

    with st.spinner("그래프 구조 및 시간대별 데이터 생성 중..."):
        (graph,
         segments,
         segment_to_nodes,
         distance_for_segment,
         congestion_by_hour,
         throughput_by_hour,
         speed_by_hour) = build_graph_and_time_data(df_raw)

    # 노드 목록
    nodes = sorted(list(graph.keys()))
    if len(nodes) < 2:
        st.error("그래프에 충분한 노드가 없습니다.")
        st.stop()

    st.sidebar.success(f"📍 총 {len(nodes)}개 지점 인식")

    # 출발/도착 지점 설정
    st.sidebar.subheader("🎯 경로 설정")
    start_node = st.sidebar.selectbox("출발 지점", nodes, index=0)
    end_node = st.sidebar.selectbox("도착 지점", nodes, index=1)

    if start_node == end_node:
        st.sidebar.warning("출발지와 도착지가 같습니다. 다른 지점을 선택하세요.")

    # 출발 시각 설정
    st.sidebar.subheader("⏰ 출발 시각")
    start_hour_1 = st.sidebar.slider(
        "첫 번째 출발 시각", 0, 23, 9, 1
    )

    compare_mode = st.sidebar.checkbox("두 번째 출발 시각과 비교", value=False)
    start_hour_2 = None
    if compare_mode:
        start_hour_2 = st.sidebar.slider(
            "두 번째 출발 시각", 0, 23, 14, 1
        )

    # 버튼
    analyze_button = st.sidebar.button("🚀 최적 경로 계산하기", type="primary")

    if not analyze_button:
        st.info("👈 왼쪽에서 설정을 완료한 후 버튼을 눌러주세요.")
        with st.expander("📊 원시 데이터 일부 보기"):
            st.dataframe(df_raw.head(50))
        return

    # ========== 분석 실행 ==========
    st.markdown("---")
    st.header("📈 분석 결과")

    # 첫 번째 시나리오
    with st.spinner(f"{start_hour_1}시 출발 경로 계산 중..."):
        result_1 = dijkstra_with_time(
            start_node, end_node, start_hour_1,
            graph,
            segment_to_nodes,
            distance_for_segment,
            congestion_by_hour,
            throughput_by_hour,
            speed_by_hour,
        )

    if result_1 is None:
        st.error(f"❌ {start_hour_1}시 출발: 경로를 찾을 수 없습니다.")
        return

    cost_1, arrival_hour_1, path_1 = result_1

    st.subheader(f"✅ 출발 시각: {start_hour_1}시")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("출발 지점", start_node)
    with col2:
        st.metric("도착 지점", end_node)
    with col3:
        st.metric("예상 도착 시각", format_hour(arrival_hour_1))
    with col4:
        st.metric("총 비용", f"{cost_1:.2f}")

    path_summary_1 = draw_path_summary(path_1)
    st.markdown(f"**최적 경로:** {path_summary_1}")

    st.subheader("📋 구간별 상세 정보")
    path_df_1 = draw_path_table(path_1)
    st.dataframe(path_df_1, use_container_width=True)

    total_distance_1 = sum(seg["distance"] for seg in path_1)
    st.info(f"총 이동 거리: {total_distance_1:.2f} km")

    # 두 번째 시나리오 비교
    if compare_mode and start_hour_2 is not None:
        st.markdown("---")
        st.subheader("🔄 출발 시각 비교")

        with st.spinner(f"{start_hour_2}시 출발 경로 계산 중..."):
            result_2 = dijkstra_with_time(
                start_node, end_node, start_hour_2,
                graph,
                segment_to_nodes,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour,
            )

        if result_2 is None:
            st.error(f"❌ {start_hour_2}시 출발: 경로를 찾을 수 없습니다.")
        else:
            cost_2, arrival_hour_2, path_2 = result_2
            total_distance_2 = sum(seg["distance"] for seg in path_2)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### {start_hour_1}시 출발")
                st.metric("총 비용", f"{cost_1:.2f}")
                st.metric("도착 시각", format_hour(arrival_hour_1))
                st.metric("총 거리", f"{total_distance_1:.2f} km")
            with col_b:
                st.markdown(f"### {start_hour_2}시 출발")
                st.metric(
                    "총 비용", f"{cost_2:.2f}",
                    delta=f"{cost_2 - cost_1:.2f}" if cost_2 != cost_1 else None
                )
                st.metric("도착 시각", format_hour(arrival_hour_2))
                st.metric("총 거리", f"{total_distance_2:.2f} km")

            st.markdown("### 💡 분석 결론")
            if cost_1 < cost_2:
                st.success(
                    f"**{start_hour_1}시 출발**이 **{start_hour_2}시 출발**보다 "
                    f"**{cost_2 - cost_1:.2f}만큼 더 효율적**입니다."
                )
            elif cost_2 < cost_1:
                st.success(
                    f"**{start_hour_2}시 출발**이 **{start_hour_1}시 출발**보다 "
                    f"**{cost_1 - cost_2:.2f}만큼 더 효율적**입니다."
                )
            else:
                st.info("두 시각의 비용이 동일합니다.")

    st.markdown("---")
    st.markdown("""
    ### 📖 해석 가이드
    - **간선 비용** = 거리 + 시간 비용 + 교통 비용  
    - **시간 비용** = 거리 / 속도  
    - **교통 비용** = 혼잡빈도수 × 교통량 × 연료 소비량(1로 가정)  
    - 비용이 낮을수록 더 효율적인 경로입니다.

    ※ 실제 거리 값이 아닌, '단위 구간'으로 이상화되어 있으므로  
    **상대적인 시간대/경로 비교**에 초점을 두고 해석해야 합니다.
    """)


if __name__ == "__main__":
    main()