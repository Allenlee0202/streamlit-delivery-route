# app.py
# 물류 데이터 기반 동적 다익스트라 택배 경로 분석 Streamlit 앱
# (사용자가 CSV 파일을 업로드하는 버전)

import streamlit as st
import pandas as pd
import heapq
from typing import Dict, List, Tuple

# ============================================================================
# 전역 설정 및 상수
# ============================================================================

FUEL_PER_HOUR = 1.0  # 시간당 연료 소비량 가정값
DEFAULT_SPEED = 80.0  # 속도 정보가 없을 때 기본값 (km/h)
INF = float('inf')

# CSV / 데이터 열 이름 매핑
SEGMENT_COL = "콘존명"      # 예: "구서IC-영락IC"
TIME_COL = "측정시각"       # 우리가 집계시분에서 시(hour)만 뽑아서 만들 컬럼
VOLUME_COL = "평균교통량"
SPEED_COL = "평균속도"
CONG_COL = "혼잡빈도수"
AGG_TIME_COL = "집계시분"   # CSV에 실제로 들어있는 시간 정보 컬럼

# 거리 정보 딕셔너리 (원하는 만큼 채우기)
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
    '집계시분' 컬럼에서 '3:15', '03:00', '12:30' 같은 값이 들어있다고 가정하고
    앞의 '시' 부분만 정수(0~23)로 추출한다.
    """
    try:
        s = str(val)
        # 콜론(:) 기준으로 앞부분
        h = int(s.split(":")[0])
        return h % 24
    except Exception:
        return 0


def load_csv_to_dataframe(uploaded_file) -> pd.DataFrame:
    """
    업로드된 CSV 파일을 읽어서, 기존 엑셀 기반 코드가 기대하던 형태로 맞춰준다.
    - 집계시분 → TI
