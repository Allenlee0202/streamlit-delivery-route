# app.py
# 물류 데이터 기반 동적 다익스트라 택배 경로 분석 Streamlit 앱
# (자동화.csv를 직접 읽어서 실행하는 버전)

import streamlit as st
import pandas as pd
import heapq
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 전역 설정 및 상수
# ============================================================================

FUEL_PER_HOUR = 1.0  # 시간당 연료 소비량 가정값
DEFAULT_SPEED = 80.0  # 속도 정보가 없을 때 기본값 (km/h)
INF = float('inf')

# 엑셀/CSV 열 이름 매핑
SEGMENT_COL = "콘존명"
TIME_COL = "측정시각"       # 우리는 집계시분에서 시(hour)만 뽑아서 여기에 넣을 거야
VOLUME_COL = "평균교통량"
SPEED_COL = "평균속도"
CONG_COL = "혼잡빈도수"

CSV_PATH = "자동화.csv"      # 같은 폴더에 있는 CSV 파일 이름

# 거리 정보 딕셔너리 (필요하면 직접 채우기)
# 예시: (노드A, 노드B) -> 거리(km)
DISTANCE_MAP = {
    ("구서IC", "영락IC"): 10.5,
    ("영락IC", "부산TG"): 15.2,
    ("부산TG", "노포IC"): 8.3,
    ("노포IC", "서부산IC"): 12.0,
    ("서부산IC", "김해IC"): 9.5,
    ("김해IC", "동김해IC"): 6.8,
    ("동김해IC", "장유IC"): 5.2,
    # 필요시 추가
}


# ============================================================================
# 0. CSV 전처리: 집계시분 → 시간대(hour) 추출
# ============================================================================

def parse_hour_from_string(val) -> int:
    """
    '집계시분' 컬럼에서 '3:15', '12:30' 같은 값이 들어있다고 가정하고
    앞의 시만 정수(0~23)로 추출.
    """
    try:
        s = str(val)
        h = int(s.split(":")[0])
        return h % 24
    except Exception:
        return 0


def load_csv_and_convert_to_excel_like(csv_path: str) -> pd.DataFrame:
    """
    자동화.csv를 읽어서, 기존 엑셀 기반 코드가 기대하던 형태로 맞춰준다.
    - 집계시분 → TIME_COL('측정시각')에 시간대 정수 저장
    - 나머지 열 이름은 그대로 사용
    """
    # 인코딩은 환경에 따라 다를 수 있어서 두 번 시도
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="cp949")

    required_cols = ["집계시분", SEGMENT_COL, VOLUME_COL, SPEED_COL, CONG_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"필수 열 '{col}' 이(가) CSV에 없습니다. 현재 열 목록: {df.columns.tolist()}")

    # 집계시분 → TIME_COL(측정시각) (정수 시각)
    df[TIME_COL] = df["집계시분"].map(parse_hour_from_string)

    return df


# ============================================================================
# 1. 데이터 전처리 함수 (기존 엑셀용 로직 재사용)
# ============================================================================

def preprocess_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    업로드된 엑셀/CSV 데이터를 전처리합니다.
    - 시간대 추출 (0~23 정수) → df['hour'] 컬럼 생성
    - 결측치 제거
    """
    df = df.copy()
    
    # 시간대 정보 추출
    if TIME_COL in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[TIME_COL]):
            df["hour"] = df[TIME_COL].dt.hour
        else:
            # 이미 정수형이거나 변환 가능한 경우
            df["hour"] = df[TIME_COL].astype(int) % 24
    else:
        st.error(f"'{TIME_COL}' 열을 찾을 수 없습니다.")
        return None
    
    # 필수 열 확인
    required_cols = [SEGMENT_COL, VOLUME_COL, SPEED_COL, CONG_COL]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"'{col}' 열을 찾을 수 없습니다.")
            return None
    
    # 결측치 제거
    df = df.dropna(subset=required_cols + ["hour"])
    
    return df


def extract_nodes_from_segments(df: pd.DataFrame) -> List[str]:
    """
    콘존명 열에서 모든 노드(지점) 이름을 추출합니다.
    예: "구서IC-영락IC" -> ["구서IC", "영락IC"]
    """
    nodes = set()
    for segment in df[SEGMENT_COL].unique():
        s = str(segment)
        # "-" 또는 "–" 또는 "~" 기준으로 split
        s = s.replace("–", "-").replace("~", "-")
        parts = s.split("-")
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                nodes.add(cleaned)
    return sorted(list(nodes))


def get_distance(node_a: str, node_b: str) -> float:
    """
    두 노드 사이의 거리를 반환합니다.
    DISTANCE_MAP에서 찾으며, 없으면 기본값 1.0을 반환합니다.
    """
    if (node_a, node_b) in DISTANCE_MAP:
        return DISTANCE_MAP[(node_a, node_b)]
    if (node_b, node_a) in DISTANCE_MAP:
        return DISTANCE_MAP[(node_b, node_a)]
    return 1.0  # 정보 없는 경우 기본값


# ============================================================================
# 2. 그래프 구조 생성 및 시간대별 값 계산
# ============================================================================

def build_graph_and_time_data(df: pd.DataFrame):
    """
    데이터로부터 다음을 생성합니다:
    - graph: {node: [neighbor_nodes]}
    - segments: 전체 segment 리스트
    - segment_to_nodes: {segment: (node_a, node_b)}
    - distance_for_segment: {segment: distance}
    - congestion_by_hour: {(segment, hour): C_e(h)}
    - throughput_by_hour: {(segment, hour): throughput(h)}
    - speed_by_hour: {(segment, hour): v(h)}
    """
    graph: Dict[str, List[str]] = {}
    segments: List[str] = []
    segment_to_nodes: Dict[str, Tuple[str, str]] = {}
    distance_for_segment: Dict[str, float] = {}
    
    # 1) 콘존명별로 노드 쌍 추출 및 그래프 구조 생성
    for segment in df[SEGMENT_COL].unique():
        s = str(segment)
        # "–", "~" 등을 "-"로 통일
        s2 = s.replace("–", "-").replace("~", "-")
        parts = s2.split("-")
        if len(parts) < 2:
            continue  # 형식이 맞지 않으면 무시
        
        node_a, node_b = parts[0].strip(), parts[1].strip()
        if not node_a or not node_b:
            continue
        
        # 양방향 간선 추가 (실제 운행방향에 따라 조정 가능)
        if node_a not in graph:
            graph[node_a] = []
        if node_b not in graph:
            graph[node_b] = []
        
        if node_b not in graph[node_a]:
            graph[node_a].append(node_b)
        if node_a not in graph[node_b]:
            graph[node_b].append(node_a)
        
        segments.append(s)
        segment_to_nodes[s] = (node_a, node_b)
        
        # 거리 정보
        d = get_distance(node_a, node_b)
        distance_for_segment[s] = d
    
    # 2) 시간대별 집계
    congestion_by_hour: Dict[Tuple[str, int], float] = {}
    throughput_by_hour: Dict[Tuple[str, int], float] = {}
    speed_by_hour: Dict[Tuple[str, int], float] = {}
    
    grouped = df.groupby([SEGMENT_COL, "hour"])
    
    for (segment, hour), subdf in grouped:
        # 혼잡빈도수 합
        C_e_h = subdf[CONG_COL].sum()
        congestion_by_hour[(segment, int(hour))] = float(C_e_h)
        
        # 평균교통량 합
        throughput_h = subdf[VOLUME_COL].sum()
        throughput_by_hour[(segment, int(hour))] = float(throughput_h)
        
        # 평균속도: 0이 아닌 값들의 평균
        valid_speeds = subdf[SPEED_COL][subdf[SPEED_COL] > 0]
        if len(valid_speeds) > 0:
            v_h = float(valid_speeds.mean())
        else:
            v_h = DEFAULT_SPEED
        speed_by_hour[(segment, int(hour))] = v_h
    
    return (graph, segments, segment_to_nodes, distance_for_segment,
            congestion_by_hour, throughput_by_hour, speed_by_hour)


# ============================================================================
# 3. 간선 가중치 계산 함수
# ============================================================================

def edge_weight_from_excel(segment: str, hour: int,
                           distance_for_segment: Dict,
                           congestion_by_hour: Dict,
                           throughput_by_hour: Dict,
                           speed_by_hour: Dict) -> Tuple[float, float]:
    """
    시간 의존적 간선 가중치 w_e(h)를 계산합니다.
    
    w_e(h) = d_e + time_cost + traffic_cost
    - time_cost = d_e / v(h)
    - traffic_cost = C_e(h) * (throughput(h) * fuel_per_hour)
    
    반환값: (가중치, 속도)
    """
    hour = hour % 24
    
    # 거리
    d_e = distance_for_segment.get(segment, 1.0)
    
    # 혼잡빈도수
    C_e_h = congestion_by_hour.get((segment, hour), 0.0)
    
    # 평균교통량
    tp_h = throughput_by_hour.get((segment, hour), 0.0)
    
    # 평균속도
    v_h = speed_by_hour.get((segment, hour), DEFAULT_SPEED)
    
    # 시간 비용 계산
    time_cost = d_e / max(v_h, 1e-6)
    
    # 교통 비용 계산
    traffic_cost = C_e_h * (tp_h * FUEL_PER_HOUR)
    
    # 최종 가중치
    w = d_e + time_cost + traffic_cost
    
    return w, v_h


# ============================================================================
# 4. 시간 의존 다익스트라 알고리즘
# ============================================================================

def dijkstra_with_time(start: str, end: str, start_hour: int,
                       graph: Dict,
                       segment_to_nodes: Dict,
                       distance_for_segment: Dict,
                       congestion_by_hour: Dict,
                       throughput_by_hour: Dict,
                       speed_by_hour: Dict):
    """
    시간 의존적 다익스트라 알고리즘을 수행합니다.
    
    상태: (node, hour)
    - 출발 시각에서 시작해 각 구간을 통과하면서 시간이 변합니다.
    - 가중치는 통과하는 시각에 따라 동적으로 계산됩니다.
    
    반환값:
    - 경로가 없으면 None
    - 있으면 (총비용, 도착시각, 경로정보 리스트)
      경로정보: [{'from', 'to', 'segment', 'start_hour', 'end_hour', ...}, ...]
    """
    if start not in graph or end not in graph:
        return None

    start_hour = start_hour % 24
    
    # 거리 배열 초기화: d[node][hour]
    d = {node: [INF] * 24 for node in graph}
    d[start][start_hour] = 0.0
    
    # 이전 상태 추적: prev[node][hour] = (prev_node, prev_hour, segment_used)
    prev = {node: [None] * 24 for node in graph}
    
    # 우선순위 큐: (cost, node, hour)
    pq = [(0.0, start, start_hour)]
    
    while pq:
        cost, node, h = heapq.heappop(pq)
        
        # 이미 처리된 상태면 스킵
        if cost > d[node][h]:
            continue
        
        # 이웃 노드 탐색
        if node not in graph:
            continue
        
        for next_node in graph[node]:
            # segment 이름 찾기 (원데이터 이름 그대로 사용)
            seg_candidates = [
                f"{node}-{next_node}",
                f"{next_node}-{node}",
                f"{node}–{next_node}",
                f"{next_node}–{node}",
                f"{node}~{next_node}",
                f"{next_node}~{node}",
            ]
            
            segment = None
            for s in seg_candidates:
                if s in segment_to_nodes:
                    segment = s
                    break
            if segment is None:
                continue
            
            # 현재 시각 h에서 해당 segment의 가중치 계산
            w, v_h = edge_weight_from_excel(
                segment, h,
                distance_for_segment,
                congestion_by_hour,
                throughput_by_hour,
                speed_by_hour
            )
            
            # 이동 시간 계산 (시간 단위)
            d_e = distance_for_segment.get(segment, 1.0)
            t_hours = d_e / max(v_h, 1e-6)
            
            # 도착 시각 계산 (24시간 기준)
            h_prime = int((h + t_hours) % 24)
            
            # 새로운 비용
            new_cost = cost + w
            
            # 갱신 조건
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
        return None  # 경로 없음
    
    # 경로 역추적
    path_info = []
    curr_node = end
    curr_hour = best_hour
    
    while prev[curr_node][curr_hour] is not None:
        prev_node, prev_hour, segment = prev[curr_node][curr_hour]
        
        # 해당 구간 정보 수집
        d_e = distance_for_segment.get(segment, 1.0)
        w, v_h = edge_weight_from_excel(
            segment, prev_hour,
            distance_for_segment,
            congestion_by_hour,
            throughput_by_hour,
            speed_by_hour
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
            "weight": w
        })
        
        curr_node = prev_node
        curr_hour = prev_hour
    
    path_info.reverse()
    
    return min_cost, best_hour, path_info


# ============================================================================
# 5. UI 관련 함수
# ============================================================================

def draw_path_summary(path_info: List[Dict]) -> str:
    """
    경로를 A → B → C 형태의 문자열로 반환합니다.
    """
    if not path_info:
        return ""
    
    nodes = [path_info[0]["from"]]
    for seg in path_info:
        nodes.append(seg["to"])
    
    return " → ".join(nodes)


def draw_path_table(path_info: List[Dict]) -> pd.DataFrame:
    """
    구간별 상세 정보를 DataFrame으로 반환합니다.
    """
    rows = []
    for seg in path_info:
        rows.append({
            "구간": f"{seg['from']}-{seg['to']}",
            "출발 시각": f"{seg['start_hour']}시",
            "도착 시각": f"{seg['end_hour']}시",
            "거리(km)": round(seg['distance'], 2),
            "속도(km/h)": round(seg['speed'], 2),
            "혼잡도": round(seg['congestion'], 2),
            "교통량": round(seg['throughput'], 2),
            "간선 비용": round(seg['weight'], 2)
        })
    
    return pd.DataFrame(rows)


def format_hour(hour: int) -> str:
    """시각을 문자열로 포맷합니다."""
    return f"{hour}시"


# ============================================================================
# 6. 메인 Streamlit 앱
# ============================================================================

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="물류 데이터 기반 동적 다익스트라 택배 경로 분석",
        layout="wide"
    )
    
    st.title("🚚 물류 데이터 기반 동적 다익스트라 택배 경로 분석")
    
    st.markdown("""
    ### 프로젝트 개요
    이 앱은 **자동화.csv에 들어 있는 고속도로 교통 데이터**를 활용하여  
    택배 배송에 가장 효율적인 **출발 시간**과 **경로**를 분석합니다.
    
    - **시간 의존적 다익스트라 알고리즘**: 시간대별로 변하는 교통 상황을 반영  
    - **동적 가중치**: 거리, 속도, 혼잡도, 교통량을 종합적으로 고려  
    - **비교 분석**: 두 개의 출발 시각을 비교하여 최적 시간대 도출  
    
    ※ 사용자는 파일 업로드 없이, 같은 폴더의 `자동화.csv`를 자동으로 읽어 분석합니다.
    ---
    """)
    
    st.sidebar.header("📋 분석 설정")
    st.sidebar.markdown("""
    **사용 방법:**
    1. 출발/도착 지점을 선택합니다.
    2. 출발 시각을 설정합니다.
    3. (선택) 두 번째 출발 시각을 비교합니다.
    4. **최적 경로 계산하기** 버튼을 누릅니다.
    """)

    # CSV 데이터 로드
    try:
        df_raw = load_csv_and_convert_to_excel_like(CSV_PATH)
    except Exception as e:
        st.error(f"`{CSV_PATH}` 파일을 읽는 중 오류 발생: {e}")
        return
    
    # 전처리
    df = preprocess_excel(df_raw)
    if df is None or len(df) == 0:
        st.error("데이터 전처리 실패 또는 유효한 데이터가 없습니다.")
        return
    
    # 그래프 및 시간대별 데이터 구축
    with st.spinner("그래프 구조 및 시간대별 데이터 생성 중..."):
        (graph, segments, segment_to_nodes, distance_for_segment,
         congestion_by_hour, throughput_by_hour, speed_by_hour) = \
            build_graph_and_time_data(df)
    
    # 노드 리스트 추출
    nodes = extract_nodes_from_segments(df)
    
    if len(nodes) < 2:
        st.error("그래프에 충분한 노드가 없습니다. 데이터를 확인하세요.")
        return
    
    st.sidebar.success(f"📍 총 {len(nodes)}개 지점 인식")
    
    # 출발지/도착지 선택
    st.sidebar.subheader("🎯 경로 설정")
    start_node = st.sidebar.selectbox("출발 지점", nodes, index=0)
    end_node = st.sidebar.selectbox("도착 지점", nodes, index=min(1, len(nodes)-1))
    
    if start_node == end_node:
        st.sidebar.warning("출발지와 도착지가 같습니다.")
    
    # 출발 시각 선택
    st.sidebar.subheader("⏰ 출발 시각")
    start_hour_1 = st.sidebar.slider(
        "첫 번째 출발 시각",
        min_value=0, max_value=23, value=9, step=1,
        help="0시부터 23시까지 선택 가능"
    )
    
    compare_mode = st.sidebar.checkbox("두 번째 출발 시각과 비교", value=False)
    start_hour_2 = None
    if compare_mode:
        start_hour_2 = st.sidebar.slider(
            "두 번째 출발 시각",
            min_value=0, max_value=23, value=14, step=1
        )
    
    # 분석 실행 버튼
    analyze_button = st.sidebar.button("🚀 최적 경로 계산하기", type="primary")
    
    # 메인 영역
    if not analyze_button:
        st.info("👈 왼쪽 사이드바에서 설정을 완료한 후 '최적 경로 계산하기' 버튼을 눌러주세요.")
        
        # 데이터 미리보기
        with st.expander("📊 원본 데이터 미리보기 (자동화.csv → 전처리 후)"):
            st.dataframe(df.head(30))
        return
    
    # 분석 수행
    st.markdown("---")
    st.header("📈 분석 결과")
    
    # 첫 번째 시나리오 분석
    with st.spinner(f"{start_hour_1}시 출발 경로 계산 중..."):
        result_1 = dijkstra_with_time(
            start_node, end_node, start_hour_1,
            graph, segment_to_nodes, distance_for_segment,
            congestion_by_hour, throughput_by_hour, speed_by_hour
        )
    
    if result_1 is None:
        st.error(f"❌ {start_hour_1}시 출발: 경로를 찾을 수 없습니다. 출발지와 도착지가 연결되어 있는지 확인하세요.")
        return
    
    cost_1, arrival_hour_1, path_1 = result_1
    
    # 결과 출력 - 시나리오 1
    st.subheader(f"✅ 출발 시각: {start_hour_1}시")
    
    # 카드 형식 요약
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("출발 지점", start_node)
    with col2:
        st.metric("도착 지점", end_node)
    with col3:
        st.metric("예상 도착 시각", format_hour(arrival_hour_1))
    with col4:
        st.metric("총 비용", f"{cost_1:.2f}")
    
    # 경로 요약
    path_summary_1 = draw_path_summary(path_1)
    st.markdown(f"**최적 경로:** {path_summary_1}")
    
    # 구간별 상세 정보
    st.subheader("📋 구간별 상세 정보")
    path_df_1 = draw_path_table(path_1)
    st.dataframe(path_df_1, use_container_width=True)
    
    # 총 거리 계산
    total_distance_1 = sum([seg['distance'] for seg in path_1])
    st.info(f"총 이동 거리: {total_distance_1:.2f} km")
    
    # 비교 모드
    if compare_mode and start_hour_2 is not None:
        st.markdown("---")
        st.subheader("🔄 출발 시각 비교")
        
        with st.spinner(f"{start_hour_2}시 출발 경로 계산 중..."):
            result_2 = dijkstra_with_time(
                start_node, end_node, start_hour_2,
                graph, segment_to_nodes, distance_for_segment,
                congestion_by_hour, throughput_by_hour, speed_by_hour
            )
        
        if result_2 is None:
            st.error(f"❌ {start_hour_2}시 출발: 경로를 찾을 수 없습니다.")
        else:
            cost_2, arrival_hour_2, path_2 = result_2
            
            # 비교 표
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"### {start_hour_1}시 출발")
                st.metric("총 비용", f"{cost_1:.2f}")
                st.metric("도착 시각", format_hour(arrival_hour_1))
                st.metric("총 거리", f"{total_distance_1:.2f} km")
            
            with col_b:
                st.markdown(f"### {start_hour_2}시 출발")
                total_distance_2 = sum([seg['distance'] for seg in path_2])
                st.metric("총 비용", f"{cost_2:.2f}", 
                         delta=f"{cost_2 - cost_1:.2f}" if cost_2 != cost_1 else None)
                st.metric("도착 시각", format_hour(arrival_hour_2))
                st.metric("총 거리", f"{total_distance_2:.2f} km")
            
            # 결론
            st.markdown("### 💡 분석 결론")
            if cost_1 < cost_2:
                st.success(f"**{start_hour_1}시 출발**이 **{start_hour_2}시 출발**보다 "
                          f"**{cost_2 - cost_1:.2f}만큼 더 효율적**입니다. "
                          f"교통량과 혼잡도가 상대적으로 낮은 시간대입니다.")
            elif cost_2 < cost_1:
                st.success(f"**{start_hour_2}시 출발**이 **{start_hour_1}시 출발**보다 "
                          f"**{cost_1 - cost_2:.2f}만큼 더 효율적**입니다. "
                          f"교통량과 혼잡도가 상대적으로 낮은 시간대입니다.")
            else:
                st.info("두 시각의 비용이 동일합니다.")
    
    # 하단 안내
    st.markdown("---")
    st.markdown("""
    ### 📖 해석 가이드
    - **간선 비용**: 거리 + 시간 비용 + 교통 비용의 합  
    - **시간 비용**: 해당 구간을 통과하는 데 걸리는 시간 (거리/속도)  
    - **교통 비용**: 혼잡빈도수 × 교통량 × 연료 소비량  
    - 비용이 낮을수록 효율적인 경로입니다.
    
    **주의**: 현재 분석은 `자동화.csv`에 포함된 과거 데이터를 기반으로 하며,  
    실시간 교통 상황과는 다를 수 있습니다.
    """)


if __name__ == "__main__":
    main()
