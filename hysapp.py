import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def init_session_state():
    """세션 상태 초기화: 스핀 격자, 핀닝 필드, 자계, 히스토리 등"""
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = True
        st.session_state["grid_size"] = 20
        st.session_state["spins"] = np.random.choice([-1, 1], size=(20, 20))
        st.session_state["pinning"] = np.random.uniform(-1, 1, size=(20, 20))
        st.session_state["H_ext"] = 0.0
        st.session_state["H_dir"] = 1
        st.session_state["H_history"] = []
        st.session_state["M_history"] = []
        st.session_state["running"] = False

def reset_simulation(grid_size):
    """시뮬레이션 초기화: 새로운 격자 생성"""
    st.session_state["grid_size"] = grid_size
    st.session_state["spins"] = np.random.choice([-1, 1], size=(grid_size, grid_size))
    st.session_state["pinning"] = np.random.uniform(-1, 1, size=(grid_size, grid_size))
    st.session_state["H_ext"] = 0.0
    st.session_state["H_dir"] = 1
    st.session_state["H_history"] = []
    st.session_state["M_history"] = []

def update_field_and_spins(field_speed, H_max, steps_per_frame, J, temperature, disorder_strength, flip_threshold):
    """외부 자계 업데이트 및 스핀 격자 업데이트"""
    dH = 0.02
    delta_H = field_speed * dH
    
    if field_speed > 0:
        H_ext = st.session_state["H_ext"]
        H_dir = st.session_state["H_dir"]
        
        H_ext += H_dir * delta_H
        
        if H_ext > H_max:
            H_ext = H_max
            H_dir = -1
        elif H_ext < -H_max:
            H_ext = -H_max
            H_dir = 1
        
        st.session_state["H_ext"] = H_ext
        st.session_state["H_dir"] = H_dir
    
    spins = st.session_state["spins"]
    pinning = st.session_state["pinning"]
    H_ext = st.session_state["H_ext"]
    grid_size = st.session_state["grid_size"]
    
    for _ in range(steps_per_frame):
        indices = np.random.randint(0, grid_size, size=(grid_size * grid_size, 2))
        
        for idx in range(len(indices)):
            i, j = indices[idx]
            
            up = spins[(i - 1) % grid_size, j]
            down = spins[(i + 1) % grid_size, j]
            left = spins[i, (j - 1) % grid_size]
            right = spins[i, (j + 1) % grid_size]
            
            neighbor_sum = up + down + left + right
            neighbor_mean = neighbor_sum / 4.0
            
            h_eff = H_ext + J * neighbor_mean + disorder_strength * pinning[i, j]
            
            s = spins[i, j]
            
            if temperature == 0:
                if h_eff * s < 0 and abs(h_eff) > flip_threshold:
                    spins[i, j] *= -1
            else:
                p_flip = 1.0 / (1.0 + np.exp(2 * s * h_eff / temperature))
                if np.random.rand() < p_flip:
                    spins[i, j] *= -1
    
    st.session_state["spins"] = spins
    
    M = spins.mean()
    st.session_state["H_history"].append(H_ext)
    st.session_state["M_history"].append(M)
    
    if len(st.session_state["H_history"]) > 5000:
        st.session_state["H_history"] = st.session_state["H_history"][-5000:]
        st.session_state["M_history"] = st.session_state["M_history"][-5000:]

def plot_domain(spins):
    """자기 영역(자구) 시각화"""
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    grid_size = spins.shape[0]
    
    cmap = plt.cm.RdBu_r
    im = ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, origin='lower', interpolation='nearest')
    
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)
    
    U = spins
    V = np.zeros_like(spins)
    
    colors = np.where(spins > 0, 'red', 'blue')
    colors_flat = colors.flatten()
    
    ax.quiver(X, Y, U, V, color=colors_flat, scale=grid_size*0.8, width=0.003, headwidth=3, headlength=4)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("자기 영역(자구) 배열", fontsize=14, pad=10)
    
    return fig

def plot_hysteresis(H_history, M_history, H_max):
    """H-M 히스테리시스 곡선 시각화"""
    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    if len(H_history) > 0:
        ax.plot(H_history, M_history, '-', linewidth=1.5, color='blue', alpha=0.7)
        
        if len(H_history) > 0:
            ax.scatter(H_history[-1], M_history[-1], color='red', s=50, zorder=5, edgecolors='black', linewidths=1)
    
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax.set_xlim(-H_max * 1.1, H_max * 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    ax.set_xlabel("외부 자계 H", fontsize=12)
    ax.set_ylabel("자화 M", fontsize=12)
    ax.set_title("H–M 히스테리시스 곡선", fontsize=14, pad=10)
    
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    """메인 Streamlit 앱"""
    st.set_page_config(page_title="자기 히스테리시스 시뮬레이터", layout="wide")
    
    init_session_state()
    
    st.title("자기 히스테리시스 & 자기 영역(자구) 시뮬레이터")
    
    st.markdown("""
    **왼쪽**: 강자성체 내부의 자기 영역(자구) 배열  
    **오른쪽**: 외부 자계에 따른 H–M 히스테리시스 루프  
    외부 자계를 변화시키면서 자구의 정렬과 히스테리시스 현상을 관찰하세요.
    """)
    
    st.sidebar.title("시뮬레이션 설정")
    
    st.sidebar.markdown("### 격자 및 자계 설정")
    grid_size = st.sidebar.slider("자기 영역 격자 크기 (N×N)", 10, 40, st.session_state["grid_size"], 1)
    H_max = st.sidebar.slider("최대 자계 크기 H_max", 0.5, 3.0, 1.5, 0.1)
    field_speed = st.sidebar.slider("외부 자계 변화 속도", 0.0, 1.0, 0.2, 0.05)
    steps_per_frame = st.sidebar.slider("한 프레임당 스핀 업데이트 횟수", 1, 30, 5, 1)
    
    st.sidebar.markdown("### 물리 파라미터")
    J = st.sidebar.slider("이웃 결합 강도 J", 0.0, 3.0, 1.0, 0.1)
    temperature = st.sidebar.slider("유효 온도 (열요동)", 0.0, 2.0, 0.2, 0.05)
    disorder_strength = st.sidebar.slider("결함 / 핀닝 세기", 0.0, 1.0, 0.1, 0.05)
    flip_threshold = st.sidebar.slider("온도 0일 때 뒤집기 임계값", 0.0, 0.5, 0.1, 0.01)
    
    st.sidebar.markdown("### 제어")
    
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    with col_btn1:
        if st.button("초기화", use_container_width=True):
            reset_simulation(grid_size)
            st.rerun()
    
    with col_btn2:
        if st.button("한 스텝 진행", use_container_width=True):
            update_field_and_spins(field_speed, H_max, steps_per_frame, J, temperature, disorder_strength, flip_threshold)
            st.rerun()
    
    if st.session_state["running"]:
        button_label = "실행 중지"
    else:
        button_label = "실행"
    
    if st.sidebar.button(button_label, use_container_width=True):
        st.session_state["running"] = not st.session_state["running"]
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**현재 외부 자계**: {st.session_state['H_ext']:.3f}")
    st.sidebar.markdown(f"**현재 자화**: {st.session_state['spins'].mean():.3f}")
    st.sidebar.markdown(f"**히스토리 길이**: {len(st.session_state['H_history'])}")
    
    if grid_size != st.session_state["grid_size"]:
        reset_simulation(grid_size)
        st.rerun()
    
    if st.session_state["running"]:
        update_field_and_spins(field_speed, H_max, steps_per_frame, J, temperature, disorder_strength, flip_threshold)
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_domain = plot_domain(st.session_state["spins"])
        st.pyplot(fig_domain)
        plt.close(fig_domain)
    
    with col2:
        fig_hysteresis = plot_hysteresis(
            st.session_state["H_history"],
            st.session_state["M_history"],
            H_max
        )
        st.pyplot(fig_hysteresis)
        plt.close(fig_hysteresis)

if __name__ == "__main__":
    main()