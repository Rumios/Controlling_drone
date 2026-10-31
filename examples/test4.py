"""CF2X 직접 모터 RPM 기반 이동/자세 성능 측정 통합 실험 (240 Hz).

navigation 모드: 계획된 시간별 궤적 없이 (0,0,100)m -> (10,10,100)m 이동 및 정지.
progressive_tune 모드(선택): 무오버슈트 검증 기준선에서 최대 제한값 하나씩 증가하며 자동 평가.
settling_tune 모드(선택): XY 튜닝 통과값 기반으로 Yaw/고도 정착 보정 및 성능 한계 탐색.
combo_search 모드(기본): 이전 702회 측정 기반 10개 복합 파라미터 후보를 전 시나리오에서 비교.
all_axes_batch 모드(선택): Roll/Pitch/Yaw 단독·동시 2052회 물리 한계/정착 측정.
목표점까지 남은 거리와 실제 접근 속도로 목표 속도를 생성한다.
Roll/Pitch/고도 PD, Yaw 예측 제동, RPM 믹싱은 기존 구현을 보존한다.
주의: 장애물 회피/이동 표적 추적/실기체 안전보증/수학적 최소시간 최적화는 포함하지 않는다.
GUI free는 마우스 시점 자유, follow는 자세/거리 조절 유지 + 자동 중심점 추적.
"""
import csv
import time
from pathlib import Path

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# 기본값: 702회 결과 주변 10개 파라미터 조합 비교. 일반 비행은 "navigation".
EXPERIMENT_MODE = "yaw_coupling_ab"   # "yaw_coupling_ab" / "combo_search" / "settling_tune" / "navigation" / "progressive_tune" / "all_axes_batch" / "yaw_batch"
TUNE_PROFILE = "full"          # "smoke"(간단 확인), "standard", "full"(다수 반복)

# Yaw 물리 한계 측정/오버슈트 최적화 일괄 시험 (소프트 제한 우회).
# CtrlAviary/CF2X, 실제 모터 0~MAX_RPM 제약은 반드시 유지한다.
# 32 calibration + 80 target = 112 cases, 240Hz all-step CSV.
YAW_BATCH_GUI = False           # 수십~수백 시험은 GUI 없이 실행 권장
YAW_BATCH_REAL_TIME = False     # True는 GUI 관찰용, 대량 시험에는 False
YAW_BATCH_START_ALT_M = 100.0
YAW_BATCH_WARMUP_S = 0.125
YAW_BATCH_PULSE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
YAW_BATCH_PULSE_DURATIONS_S = (0.0125, 0.025, 0.05, 0.10)
YAW_BATCH_DIRECTIONS = (+1, -1)
YAW_BATCH_COAST_S = 0.025       # 가속 종료 후 관성 회전 관측
YAW_BATCH_BRAKE_TIMEOUT_S = 1.0
YAW_BATCH_BRAKE_STOP_RATE_DEG_S = 1.0
YAW_BATCH_TARGET_ANGLES_DEG = (15.0, 30.0, 60.0, 90.0, 150.0)
YAW_BATCH_TARGET_TORQUE_FRACTIONS = (0.65, 1.0)
YAW_BATCH_BRAKE_MODEL_SCALES = (0.65, 0.80, 0.95, 1.05)
# 1.05는 계측 감속보다 낙관적으로 가정하는 stress test; 안전 운용 추천값 아님.
YAW_BATCH_TARGET_TIMEOUT_S = 3.50
YAW_BATCH_ARRIVAL_ANGLE_DEG = 0.15
YAW_BATCH_ARRIVAL_RATE_DEG_S = 0.25
YAW_BATCH_SETTLE_HOLD_S = 0.25
YAW_BATCH_POST_SETTLE_OBSERVE_S = 0.20
YAW_BATCH_OVERSHOOT_TOL_DEG = 0.10
YAW_BATCH_BRAKE_DELAY_S = 0.0125   # 240Hz 3틱 보수적 여유
YAW_BATCH_BRAKE_MARGIN_DEG = 0.20
YAW_BATCH_TERMINAL_START_ANGLE_DEG = 2.5
YAW_BATCH_TERMINAL_START_RATE_DEG_S = 80.0
YAW_BATCH_TERMINAL_KP = 900.0   # [deg/s²]/deg (초기 실험값)
YAW_BATCH_TERMINAL_KD = 60.0    # [deg/s²]/(deg/s) (초기 실험값)
# 아래는 모션 최적화용 상한이 아니라 비정상 상태 시험 중단 조건.
YAW_BATCH_ABORT_RATE_DEG_S = 3500.0
YAW_BATCH_ABORT_TILT_DEG = 25.0
YAW_BATCH_ABORT_Z_ERROR_M = 1.0
YAW_BATCH_ABORT_XY_DRIFT_M = 2.0

# ======================= 1. 실험 설정 =======================
SIM_FREQ = 240
DURATION = 20.0
INITIAL_EVAL_END = 5.0
INITIAL_RPY_DEG = np.array([0.0, 0.0, 0.0])  # 처음부터 수평 호버링
TARGET_RPY_DEG = np.array([0.0, 0.0, 0.0])
INITIAL_ALTITUDE = 100.0
TARGET_ALTITUDE = 100.0
# 마지막 대각선 실험과 동일한 시작점/고정 목표점.
START_XY_M = np.array([0.0, 0.0])
TARGET_XY_M = np.array([10.0, 10.0])
START_HOVER_S = 1.0            # 비교를 위한 출발 전 정지; 경로의 시간 제약은 아님
SAVE_NAVIGATION_PLOT = True
# 정밀 최종 정착용으로 기존 검증 수준의 XY PD 이득 유지
KP_XY = 0.8
KD_XY = 1.8
# 목표점 접근 목표: 높은 속도 + 가용 가속도 안에서 안전한 제동을 함께 고려.
# 미검증 실험값. 이 속도가 전역 최소시간 해임을 의미하지 않음.
MAX_APPROACH_SPEED_M_S = 3.6875   # [m/s] 접근 목표 속도 상한
APPROACH_ACCEL_M_S2 = 2.70    # [m/s²] 진행 방향 가속
BRAKE_ACCEL_M_S2 = 2.05       # [m/s²] 진행 방향 제동 계획 및 명령 상한
MAX_XY_ACCEL = 3.0           # [m/s²] 수평 가속도 명령 벡터 크기 제한
MAX_XY_TILT_DEG = 25.0       # [deg] 목표 기울기 한계
SPEED_TRACK_GAIN = 2.0       # [1/s] 진행 방향 속도 오차 -> 가속도
LATERAL_DAMP_GAIN = 2.0      # [1/s] 목표 방향 외의 속도 제거
BRAKE_RESPONSE_S = 0.18      # [s] 제동 전환/자세 응답 지연을 고려한 경험적 여유
BRAKE_DISTANCE_MARGIN_M = 0.12  # [m] 제동 계산에서 추가 여유
TERMINAL_ENTER_DIST_M = 0.40    # [m] 목표 근처 정밀 PD 진입
TERMINAL_EXIT_DIST_M = 0.65     # [m] 상태 전환 히스테리시스
XY_POSITION_TOL_M = 0.10
XY_SPEED_TOL_M_S = 0.10
ARRIVAL_STABLE_S = 0.50
# 장래 군집 협업용 상위 속도 명령: 0.0은 정지 지향, 1.0은 정상 제한.
# 현재는 파일 상단에서 고정 설정하는 실험값이며 실시간 통신 연동은 없음.
SPEED_SCALE = 1.0
# 네비게이션 상태: 0=start hover, 1=approach, 2=terminal PD, 3=hold
NAV_HOVER, NAV_APPROACH, NAV_TERMINAL, NAV_HOLD = 0, 1, 2, 3
SETTLING_TOL_DEG = 0.5
ALT_SETTLING_TOL_M = 0.10

# 기존 Roll/Pitch에서 검증한 PD 토크 이득 (N*m/rad, N*m*s/rad)
KP_ROLL, KD_ROLL = 0.00040, 0.000095
KP_PITCH, KD_PITCH = 0.00040, 0.000095
# 비교용으로만 기존 Yaw PD 값을 보존 (새 Yaw 제어에서는 사용하지 않음)
KP_YAW, KD_YAW = 0.000015, 0.00001
# Yaw 목표: 남은 각도에 따라 허용 각속도를 줄이고 반대 토크로 감속
MAX_YAW_RATE_DEG_S = 45.0              # 최대 목표 회전 속도 [deg/s]
MAX_YAW_ACCEL_DEG_S2 = 60.0            # 명령 각가속도 제한 [deg/s^2]
YAW_BRAKE_ACCEL_DEG_S2 = 35.0          # 제동 계획용 보수적 감속 성능 [deg/s^2]
YAW_RATE_TRACK_K = 5.0                 # 목표 각속도 추종 계수 [1/s]
YAW_TORQUE_RESERVE = 0.75             # 기타 축 모터 추력 여유 보존 (0~1)
YAW_ARRIVAL_ANGLE_DEG = 0.15          # 작은 목표 오차에서 정밀 정착
YAW_ARRIVAL_RATE_DEG_S = 0.25         # 정지 판단 각속도 [deg/s]
YAW_ACCEL_FILTER_TAU_S = 0.05         # 추정 각가속도 저역통과 시간상수 [s]
GAIN_SCALE = 1.0
# 목표 통과 전에 제동하게 만드는 속도기반 가드 (추정 모델/실측 튜닝 대상).
YAW_USE_STOP_GUARD = True
YAW_STOP_LEAD_S = 0.12               # 목표 통과 이전 제동을 위한 각속도 x 지연 여유 [s]
YAW_STOP_MARGIN_DEG = 0.20           # 예측 제동각의 추가 여유 [deg]
YAW_TERMINAL_RATE_GAIN = 1.40        # 작은 각도에서 목표 속도 <= gain*|각도 오차| [1/s]
ALT_LOOKAHEAD_S = 0.40               # z_pred=z+vz*lookahead, 효과적으로 고도 D 감쇠 증가 [s]


# 고도 PD: 목표 수직 가속도 a_z = Kp*오차 - Kd*vz
# 아래 값은 출발점으로만 사용할 미검증 이득. 단독 고도 실험으로 튜닝할 것.
KP_ALT, KD_ALT = 2.0, 2.5  # 각각 1/s^2, 1/s
MAX_VERTICAL_ACCEL = 4.0   # m/s^2, 초기 실험용 보수적 가속도 명령 제한
ENABLE_TILT_COMPENSATION = True
MIN_TILT_COS = 0.5         # 지나치게 큰 기울기 보상 방지

GUI = True
# 시점 제어: 기본 free에서는 초기 1회만 카메라를 설정한다.
# free: 마우스 회전/이동/확대를 코드가 덮어쓰지 않음.
# follow: 기체 위치만 자동 추적하고, 마우스로 조절한 회전/확대는 보존.
CAMERA_MODE = "free"              # "free" 또는 "follow"
CAMERA_UPDATE_HZ = 120            # follow 전용. 240Hz 물리에서 2스텝마다 갱신
CAMERA_FOLLOW_TAU_S = 0.04        # [s] 타깃 위치 저역통과 시간상수 (작을수록 빠름)
CAMERA_DISTANCE = 18.0            # free 기본 화면: 10m 대각선 이동 전체 관찰
CAMERA_YAW = 45
CAMERA_PITCH = -35
SHOW_ATTITUDE_LABEL = True
ATTITUDE_LABEL_UPDATE_HZ = 20     # 텍스트는 적게 갱신해 GUI 부담 감소
SHOW_MOVEMENT_LINES = True        # 녹색=목표점 표식, 빨강=실제 이동
TRAIL_UPDATE_HZ = 60              # GUI 이동 흔적 60Hz
CONSOLE_LOG_HZ = 1               # GUI 부하 경감 (CSV는 매 스텝 저장)
REAL_TIME_PLAYBACK = True

# Python 파일이 프로젝트 루트 또는 examples/에 있어도 결과는 project/logs/로 저장.
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "examples":
    PROJECT_ROOT = PROJECT_ROOT.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 결과물 이름은 실행 횟수와 무관하게 고정한다. 다음 실행 시 동일 파일 덮어쓰기.
OUTPUT_FILES = {
    "raw": "drone_motion.csv",                 # 240Hz 전체 상태/제어 로그
    "attitude_txt": "drone_motion_attitude.txt",  # 자세/고도 평가 전문
    "attitude_csv": "drone_motion_attitude.csv",  # 자세/고도 평가 요약
    "kinematics_csv": "drone_motion_kinematics.csv",  # XYZ 운동 평가
    "goal_txt": "drone_motion_goal.txt",          # 목표점/제동 성능 전문
    "goal_csv": "drone_motion_goal.csv",          # 목표점/제동 성능 요약
    "plot": "drone_motion.png",                   # 거리/속도/제동/경로 시각화
}

HEADER = [
    "time_s", "roll_deg", "pitch_deg", "yaw_deg",
    "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s",
    "roll_error_deg", "pitch_error_deg", "altitude_m",
    "roll_p_torque_Nm", "roll_d_torque_Nm", "roll_torque_Nm",
    "pitch_p_torque_Nm", "pitch_d_torque_Nm", "pitch_torque_Nm",
    "motor_1_rpm", "motor_2_rpm", "motor_3_rpm", "motor_4_rpm",
    "motor_saturated",  # 기존 CSV 열 순서 보존
    "vertical_speed_m_s", "altitude_error_m", "alt_p_acc_m_s2",
    "alt_d_acc_m_s2", "alt_acc_cmd_m_s2", "total_thrust_cmd_N",
    "tilt_cos", "tilt_comp_limited", "alt_acc_limited",
    "yaw_error_deg", "yaw_p_torque_Nm", "yaw_d_torque_Nm", "yaw_torque_Nm",
    # 월드 좌표계 위치, 속도, 유한차분 가속도 (기존 열 끝에 추가)
    "x_m", "y_m",
    "vx_m_s", "vy_m_s", "vz_m_s",
    "ax_m_s2", "ay_m_s2", "az_m_s2",
    # Yaw 예측 제동 진단 (기존 CSV 열 순서 유지)
    "yaw_target_rate_deg_s", "yaw_acc_cmd_deg_s2",
    "yaw_brake_capability_deg_s2", "yaw_measured_acc_deg_s2",
    "yaw_control_mode", "yaw_pos_torque_limit_Nm", "yaw_neg_torque_limit_Nm",
    # 이동제어 열은 기존 49열 뒤에 추가
    "target_x_m", "target_y_m", "xy_error_norm_m", "xy_speed_norm_m_s",
    "xy_ax_cmd_m_s2", "xy_ay_cmd_m_s2", "roll_target_deg", "pitch_target_deg",
    # 기존 제어기 57열을 유지하고 새 항법 진단값을 뒤에 추가.
    "nav_mode", "goal_distance_m", "closing_speed_m_s", "lateral_speed_m_s",
    "approach_speed_ref_m_s", "ideal_stop_distance_m", "brake_margin_m",
    "radial_acc_cmd_m_s2", "lateral_acc_cmd_m_s2", "nav_acc_limited",
    "speed_scale", "accel_cmd_norm_m_s2", "brake_guard_active",
]
IDX = {name: i for i, name in enumerate(HEADER)}


# =================== 2. 독립 PD 제어기 (상단 고정) ===================
def roll_pd(roll_rad, roll_rate_rad_s, target_roll_rad=0.0):
    """Roll 몸체 X축의 요구 토크[N*m] 및 P,D 성분."""
    error = target_roll_rad - roll_rad
    p_torque = (KP_ROLL * GAIN_SCALE) * error
    d_torque = -(KD_ROLL * GAIN_SCALE) * roll_rate_rad_s
    return error, p_torque, d_torque, p_torque + d_torque


def pitch_pd(pitch_rad, pitch_rate_rad_s, target_pitch_rad=0.0):
    """Pitch 몸체 Y축의 요구 토크[N*m] 및 P,D 성분."""
    error = target_pitch_rad - pitch_rad
    p_torque = (KP_PITCH * GAIN_SCALE) * error
    d_torque = -(KD_PITCH * GAIN_SCALE) * pitch_rate_rad_s
    return error, p_torque, d_torque, p_torque + d_torque

def xy_position_pd(pos_xy, velocity_xy, target_xy=TARGET_XY_M):
    """월드 XY 위치/속도 오차로 목표 수평 가속도[m/s²]를 생성한다.

    절대 위치 오차만 쓰면 목표 도달 시 잔류 속도로 오버슈트할 수 있다.
    D항은 현재 월드 수평 속도를 감쇠시킨다.
    """
    error = np.asarray(target_xy, dtype=float) - np.asarray(pos_xy, dtype=float)
    acceleration = KP_XY * error - KD_XY * np.asarray(velocity_xy, dtype=float)
    magnitude = float(np.linalg.norm(acceleration))
    if magnitude > MAX_XY_ACCEL:
        acceleration = acceleration * (MAX_XY_ACCEL / magnitude)
    return error, acceleration


def point_to_point_guidance(pos_xy, velocity_xy, speed_scale=SPEED_SCALE):
    """시간별 경로 위치 명령 없이 고정 목표에 대한 수평 가속도 계산.

    delta = goal - current, d = |delta|; unit = delta / d.
    접근 속도 v_closing = v . unit; 잔여 수평속도 v_lateral = v-v_closing*unit.
    v_allowed = sqrt(2*a_brake*effective_distance), 실제 접근속도 피드백 적용.
    effective_distance = max(0, d - (margin + max(v_closing,0)*response_s)).

    반환: accel_xy[m/s²], diagnostics dict.
    역방향 이동 / 목표 도착 시 방향의 불연속은 정밀 위치 PD가 처리한다.
    """
    if not 0.0 <= speed_scale <= 1.0:
        raise ValueError("SPEED_SCALE은 [0, 1]이어야 합니다.")
    p_xy = np.asarray(pos_xy, dtype=float)
    v_xy = np.asarray(velocity_xy, dtype=float)
    delta = TARGET_XY_M - p_xy
    distance = float(np.linalg.norm(delta))
    speed = float(np.linalg.norm(v_xy))
    if distance > 1e-9:
        unit = delta / distance
    elif speed > 1e-9:
        unit = -v_xy / speed
    else:
        unit = np.array([1.0, 0.0])

    closing_speed = float(np.dot(v_xy, unit))
    lateral_vel = v_xy - closing_speed * unit
    lateral_speed = float(np.linalg.norm(lateral_vel))
    margin = BRAKE_DISTANCE_MARGIN_M + BRAKE_RESPONSE_S * max(0.0, closing_speed)
    effective_distance = max(0.0, distance - margin)
    # v_ref is a ceiling compatible with braking capability, not a time command.
    braking_speed_ceiling = float(np.sqrt(2.0 * BRAKE_ACCEL_M_S2 * effective_distance))
    target_speed = min(MAX_APPROACH_SPEED_M_S * speed_scale, braking_speed_ceiling)

    # Negative acceleration for real closing speeds above the allowed profile.
    desired_radial_acc = SPEED_TRACK_GAIN * (target_speed - closing_speed)
    radial_acc = float(np.clip(desired_radial_acc,
                               -BRAKE_ACCEL_M_S2, APPROACH_ACCEL_M_S2))
    # 실제 접근속도 기준으로 늦은 제동을 방지하는 안전 가드.
    # 속도 추종 PD가 급감속을 뒤늦게 명령하는 경우에도 정지거리 한계를 우선.
    stop_dist = max(0.0, closing_speed)**2 / (2.0 * BRAKE_ACCEL_M_S2)
    emergency_brake = bool(closing_speed > 0.05 and stop_dist + margin >= distance)
    if emergency_brake:
        radial_acc = -BRAKE_ACCEL_M_S2
    lateral_acc_vec = -LATERAL_DAMP_GAIN * lateral_vel
    combined = radial_acc * unit + lateral_acc_vec
    total = float(np.linalg.norm(combined))
    limited = total > MAX_XY_ACCEL
    if limited:
        combined *= MAX_XY_ACCEL / total

    diag = dict(
        distance=distance,
        closing_speed=closing_speed,
        lateral_speed=lateral_speed,
        target_speed=target_speed,
        ideal_stop_distance=stop_dist,
        margin=margin,
        emergency_brake=emergency_brake,
        radial_acc=radial_acc,
        lateral_acc=float(np.linalg.norm(lateral_acc_vec)),
        limited=limited,
        acceleration_norm=float(np.linalg.norm(combined)),
    )
    return combined, diag


def update_navigation_state(previous, time_s, distance_m, speed_m_s, stable_elapsed_s, dt):
    """경과 시간은 출발 전 호버링과 0.5초 도착 확인에만 사용한다.

    FOLLOW(접근)/TERMINAL 전환 및 홀드/외란 복귀는 모두 거리와 속도가 기준.
    목표점을 지나친 경우라도 방향이 재계산되며, 터미널 PD가 복구한다.
    """
    if time_s < START_HOVER_S:
        return NAV_HOVER, 0.0
    state = NAV_APPROACH if previous == NAV_HOVER else previous
    if state == NAV_APPROACH and distance_m <= TERMINAL_ENTER_DIST_M:
        state = NAV_TERMINAL
    elif state == NAV_TERMINAL and distance_m >= TERMINAL_EXIT_DIST_M:
        state = NAV_APPROACH
    elif state == NAV_HOLD and (
        distance_m >= 1.5 * XY_POSITION_TOL_M
        or speed_m_s >= 1.5 * XY_SPEED_TOL_M_S
    ):
        state = NAV_TERMINAL
    if state == NAV_TERMINAL:
        if distance_m <= XY_POSITION_TOL_M and speed_m_s <= XY_SPEED_TOL_M_S:
            stable_elapsed_s += dt
            if stable_elapsed_s >= ARRIVAL_STABLE_S:
                state = NAV_HOLD
        else:
            stable_elapsed_s = 0.0
    else:
        stable_elapsed_s = 0.0
    return state, stable_elapsed_s


def point_to_point_controller(pos_xy, velocity_xy, state, speed_scale=SPEED_SCALE):
    """상태별 XY 목표 가속도. 기존 XY 위치 PD는 초기/종단 정착에 재사용."""
    p_xy = np.asarray(pos_xy, dtype=float)
    v_xy = np.asarray(velocity_xy, dtype=float)
    if state == NAV_HOVER:
        _, acc = xy_position_pd(p_xy, v_xy, START_XY_M)
    elif state in (NAV_TERMINAL, NAV_HOLD):
        _, acc = xy_position_pd(p_xy, v_xy, TARGET_XY_M)
    else:
        acc, _ = point_to_point_guidance(p_xy, v_xy, speed_scale)
    # Diagnostics always reflect current state, including terminal/hold.
    _, diag = point_to_point_guidance(p_xy, v_xy, speed_scale)
    diag["acceleration_norm"] = float(np.linalg.norm(acc))
    if state != NAV_APPROACH:
        # 대기/최종 정착에는 진행 목표 속도를 주지 않는다.
        diag["target_speed"] = 0.0
        selected_goal = START_XY_M if state == NAV_HOVER else TARGET_XY_M
        raw_pd_acc = KP_XY * (selected_goal - p_xy) - KD_XY * v_xy
        diag["limited"] = float(np.linalg.norm(raw_pd_acc)) > MAX_XY_ACCEL
        diag["radial_acc"] = float(np.dot(acc, (TARGET_XY_M-p_xy) /
                                          max(1e-9, np.linalg.norm(TARGET_XY_M-p_xy))))
        diag["lateral_acc"] = float(np.linalg.norm(acc - diag["radial_acc"] *
                                    (TARGET_XY_M-p_xy) / max(1e-9, np.linalg.norm(TARGET_XY_M-p_xy))))
    return acc, diag


def xy_acceleration_to_attitude(accel_xy_world, yaw_rad, vertical_accel_m_s2, gravity_m_s2):
    """목표 월드 추력 방향 -> 기체 목표 Roll/Pitch (ZYX Euler, body Z up).

    R = Rz(yaw) Ry(pitch) Rx(roll); b3=R@[0,0,1].
    월드 희망 추력 방향은 [a_x, a_y, g+a_z]에 평행해야 한다.
    먼저 허용 기울기 내로 XY 가속도 명령을 잘라낸다.
    """
    effective_g = max(0.5, float(gravity_m_s2 + vertical_accel_m_s2))
    accel = np.asarray(accel_xy_world, dtype=float).copy()
    horizontal = float(np.linalg.norm(accel))
    max_horizontal = effective_g * np.tan(np.deg2rad(MAX_XY_TILT_DEG))
    if horizontal > max_horizontal:
        accel *= max_horizontal / horizontal
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    # Rz(-yaw) @ desired_world_thrust
    bx = c * accel[0] + s * accel[1]
    by = -s * accel[0] + c * accel[1]
    norm = float(np.hypot(np.hypot(bx, by), effective_g))
    roll_des = -np.arcsin(np.clip(by / norm, -1.0, 1.0))
    pitch_des = np.arctan2(bx, effective_g)
    return float(roll_des), float(pitch_des), accel


def yaw_pd(yaw_rad, yaw_rate_rad_s, target_yaw_rad=0.0):
    """이전 이득 결과와 비교를 위한 기준 PD (새 제어 루프에서는 사용 안 함)."""
    delta = target_yaw_rad - yaw_rad
    error = float(np.arctan2(np.sin(delta), np.cos(delta)))
    p_torque = KP_YAW * error
    d_torque = -KD_YAW * yaw_rate_rad_s
    return error, p_torque, d_torque, p_torque + d_torque


def yaw_torque_limits(total_thrust, roll_torque, pitch_torque,
                      arm, kf, km, max_motor_thrust):
    """Yaw를 제외한 네 모터 기본 추력에서 양/음 Yaw 토크 가용 한계를 구한다.

    모터 추력은 [0, max_motor_thrust]에 있어야 하므로, yaw 패턴
    [-1,+1,-1,+1] 방향으로 추가할 수 있는 차등추력을 계산한다.
    다른 축의 Roll/Pitch/총추력을 먼저 확보한다.
    """
    roll_delta = roll_torque / (4.0 * arm)
    pitch_delta = pitch_torque / (4.0 * arm)
    base = (np.full(4, total_thrust / 4.0)
            + roll_delta * np.array([-1., -1., 1., 1.])
            + pitch_delta * np.array([-1., 1., 1., -1.]))
    sign = np.array([-1., 1., -1., 1.])
    # 양의 yaw_delta에서 +패턴 모터는 상한, -패턴 모터는 하한이 제약.
    pos_delta = min(np.min(max_motor_thrust - base[sign > 0]),
                    np.min(base[sign < 0]))
    neg_delta = min(np.min(base[sign > 0]),
                    np.min(max_motor_thrust - base[sign < 0]))
    coefficient = 4.0 * km / kf
    return (max(0.0, pos_delta) * coefficient * YAW_TORQUE_RESERVE,
            max(0.0, neg_delta) * coefficient * YAW_TORQUE_RESERVE)


def yaw_predictive_control(yaw_rad, yaw_rate_rad_s, target_yaw_rad,
                           yaw_inertia, torque_positive_limit, torque_negative_limit):
    """Yaw 속도 계획 + 예측 제동 가드. 각도만 보고 제동하면 잔류 각속도로 오버슈트.

    theta_stop = omega_closing**2/(2*a_brake) + omega_closing*lead + margin.
    theta_stop >= 남은 각도이고 실제 closing rate가 목표 속도보다 크면 제동 토크 명령.
    작은 오차에서는 목표 각속도 <= gain*|e|로 근접하며 속도 추종으로 정지.
    실제 토크 명령에는 [RPM 0, MAX_RPM]에서 계산한 양/음 Yaw 토크 상한을 적용.
    이 모델은 검증된 수학적 무오버슈트 보장이 아니라 실험으로 평가할 대상.
    """
    error = float(np.arctan2(np.sin(target_yaw_rad - yaw_rad),
                             np.cos(target_yaw_rad - yaw_rad)))
    max_acc = np.deg2rad(MAX_YAW_ACCEL_DEG_S2)
    pos_acc_limit = max(0., torque_positive_limit / yaw_inertia)
    neg_acc_limit = max(0., torque_negative_limit / yaw_inertia)
    braking = max(1e-8, min(np.deg2rad(YAW_BRAKE_ACCEL_DEG_S2),
                            pos_acc_limit, neg_acc_limit, max_acc))
    rate_limit = np.deg2rad(MAX_YAW_RATE_DEG_S)
    target_rate = float(np.sign(error)*min(rate_limit,
                        np.sqrt(2.*braking*abs(error))))
    if YAW_USE_STOP_GUARD:
        target_rate = float(np.sign(error)*min(abs(target_rate),
                                     YAW_TERMINAL_RATE_GAIN*abs(error)))
    arrived = (abs(error)<=np.deg2rad(YAW_ARRIVAL_ANGLE_DEG)
               and abs(yaw_rate_rad_s)<=np.deg2rad(YAW_ARRIVAL_RATE_DEG_S))
    if arrived:
        # 위치 오차가 여전히 남아 있으면 낮은 목표 속도로 조금씩 보정한다.
        # 완전히 0으로 고정해 보정이 멈추는 deadband를 방지한다.
        target_rate = float(YAW_TERMINAL_RATE_GAIN*error) if YAW_USE_STOP_GUARD else 0.
    raw_accel = YAW_RATE_TRACK_K*(target_rate-yaw_rate_rad_s)
    accel_cmd = float(np.clip(raw_accel,-max_acc,max_acc))
    emergency=False
    if YAW_USE_STOP_GUARD and abs(error)>1e-10:
        direction=np.sign(error)
        closing_rate=direction*yaw_rate_rad_s
        stop_angle=(max(0.,closing_rate)**2/(2.*braking)
                    + max(0.,closing_rate)*YAW_STOP_LEAD_S
                    + np.deg2rad(YAW_STOP_MARGIN_DEG))
        # 큰 제동 토크를 마구 전환하면 240Hz 샘플마다 앞뒤로 진동할 수 있다.
        # 실제 닫힘 각속도가 목표 속도보다 빠를 때만 가드를 발동한다.
        if (closing_rate>0. and stop_angle>=abs(error)
                and closing_rate>max(0.,direction*target_rate)+np.deg2rad(0.08)):
            accel_cmd=-direction*max_acc
            emergency=True
    # 양/음의 실제 토크 가용량은 비대칭일 수 있으므로 최종 제한은 토크 단계에서 실시.
    torque=float(np.clip(yaw_inertia*accel_cmd,
                         -torque_negative_limit,torque_positive_limit))
    actual_accel_cmd=torque/yaw_inertia
    drive_torque=yaw_inertia*YAW_RATE_TRACK_K*target_rate
    damping_torque=-yaw_inertia*YAW_RATE_TRACK_K*yaw_rate_rad_s
    mode=(3 if emergency else 0 if arrived else
          2 if (abs(target_rate)<abs(yaw_rate_rad_s) and target_rate*yaw_rate_rad_s>=0)
          else 1)
    return (error,drive_torque,damping_torque,torque,
            target_rate,actual_accel_cmd,braking,mode,
            torque_positive_limit,torque_negative_limit)



def yaw_kinematic_terms(rpy_rad, body_rates_rad_s):
    """ZYX Euler yaw rate as a function of body angular velocity.

    psi_dot = (q*sin(phi) + r*cos(phi))/cos(theta)
            = coupling_term + yaw_scale*r

    Roll/Pitch near 90 degrees is an Euler singularity; the test abort guards
    well before that limit, but we also reject division near singularity.
    """
    phi, theta = float(rpy_rad[0]), float(rpy_rad[1])
    q, r = float(body_rates_rad_s[1]), float(body_rates_rad_s[2])
    cth, cphi = float(np.cos(theta)), float(np.cos(phi))
    if abs(cth) < 0.20 or abs(cphi) < 0.20:
        raise ValueError('Roll/Pitch near Euler yaw kinematic singularity')
    coupling = q * np.sin(phi) / cth
    yaw_scale = cphi / cth
    euler_rate = coupling + yaw_scale * r
    return float(euler_rate), float(coupling), float(yaw_scale)


def yaw_predictive_control_coupled(yaw_rad, rpy_rad, body_rates_rad_s,
                                   target_yaw_rad, yaw_inertia,
                                   torque_positive_limit, torque_negative_limit):
    """Euler yaw trajectory planning with BODY-Z angular-rate actuation.

    The original yaw predictive profile and braking guard are evaluated
    against the *actual Euler yaw rate*, not body r. In speed tracking,
    the required body r reference is obtained by inverse ZYX kinematics:

        r_ref = (psi_dot_ref * cos(theta) - q*sin(phi))/cos(phi)

    This changes only Yaw feedback/kinematic compensation. XY guidance,
    altitude PD, Roll/Pitch PD, yaw limits and RPM mixing stay unchanged.
    No perfect tracking guarantee under rapid multi-axis dynamics.
    """
    actual_rate, coupling, gain = yaw_kinematic_terms(rpy_rad, body_rates_rad_s)
    body_r = float(body_rates_rad_s[2])
    base = yaw_predictive_control(
        yaw_rad, actual_rate, target_yaw_rad, yaw_inertia,
        torque_positive_limit, torque_negative_limit)
    desired_euler_rate = float(base[4])
    body_r_ref = (desired_euler_rate-coupling) / gain
    max_acc = np.deg2rad(MAX_YAW_ACCEL_DEG_S2)
    if base[7] == 3:
        # Preserve the existing predictive emergency braking direction, but
        # decide to brake using the measured Euler yaw rate (base controller).
        torque = float(base[3])
    else:
        cmd_acc = float(np.clip(YAW_RATE_TRACK_K * (body_r_ref-body_r),
                                -max_acc, max_acc))
        torque = float(np.clip(yaw_inertia*cmd_acc,
                               -torque_negative_limit, torque_positive_limit))
    drive = yaw_inertia*YAW_RATE_TRACK_K*body_r_ref
    damping = -yaw_inertia*YAW_RATE_TRACK_K*body_r
    result = (base[0],drive,damping,torque,desired_euler_rate,
              torque/yaw_inertia,base[6],base[7],base[8],base[9])
    diagnostics = dict(euler_yaw_rate_rad_s=actual_rate,
                       yaw_coupling_rad_s=coupling,
                       body_yaw_rate_rad_s=body_r,
                       target_body_yaw_rate_rad_s=body_r_ref,
                       target_euler_yaw_rate_rad_s=desired_euler_rate)
    return result, diagnostics


def altitude_pd(z_m, vz_m_s, target_z_m, mass_kg, gravity_m_s2,
                roll_rad, pitch_rad):
    """고도 오차/수직속도로 네 모터의 총추력[N]을 계산한다.

    m*z_ddot = T*cos(roll)*cos(pitch) - m*g (단순 무항력 모델)
    고도 PD는 토크가 아닌 가속도를 계산한 후 질량으로 힘을 환산한다.
    """
    error = target_z_m - z_m
    p_acc = KP_ALT * error
    d_acc = -(KD_ALT + KP_ALT * ALT_LOOKAHEAD_S) * vz_m_s
    raw_acc = p_acc + d_acc
    accel = float(np.clip(raw_acc, -MAX_VERTICAL_ACCEL, MAX_VERTICAL_ACCEL))
    accel_limited = not np.isclose(accel, raw_acc)

    raw_cos = float(np.cos(roll_rad) * np.cos(pitch_rad))
    tilt_cos = max(raw_cos, MIN_TILT_COS) if ENABLE_TILT_COMPENSATION else 1.0
    tilt_limited = ENABLE_TILT_COMPENSATION and (raw_cos < MIN_TILT_COS)
    total_thrust = mass_kg * max(0.0, gravity_m_s2 + accel) / tilt_cos
    return (error, p_acc, d_acc, accel, total_thrust,
            raw_cos, bool(tilt_limited), bool(accel_limited))


# ======================= 3. 관측과 모터 배분 =======================
def measure_state(state):
    """속도 state[10:13]과 각속도 state[13:16]은 월드좌표계."""
    pos = np.asarray(state[:3], dtype=float)
    world_vel = np.asarray(state[10:13], dtype=float)
    rpy = np.asarray(state[7:10], dtype=float)
    rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
    body_rates = rot.T @ np.asarray(state[13:16], dtype=float)
    return pos, world_vel, rpy, body_rates


def allocate_motor_rpm(roll_torque, pitch_torque, yaw_torque, total_thrust,
                       arm, kf, km, max_motor_thrust):
    """CF2X 믹싱: T는 네 모터 합, 각 회전 토크 배분의 추력 합은 0.

    tau_roll = arm*(-F1-F2+F3+F4)
    tau_pitch= arm*(-F1+F2+F3-F4)
    tau_yaw  = (KM/KF)*(-F1+F2-F3+F4)
    모터별 포화 이후에는 요청된 T 및 토크가 모두 정확히 실현되지 않을 수 있음.
    """
    roll_delta = roll_torque / (4.0 * arm)
    pitch_delta = pitch_torque / (4.0 * arm)
    # KM은 RPM^2 -> Z축 반작용 토크 계수, KF는 RPM^2 -> 추력 계수.
    yaw_delta = yaw_torque * kf / (4.0 * km)
    thrust = (np.full(4, total_thrust / 4.0)
              + roll_delta * np.array([-1., -1., 1., 1.])
              + pitch_delta * np.array([-1., 1., 1., -1.])
              + yaw_delta * np.array([-1., 1., -1., 1.]))
    saturated = bool(np.any((thrust < 0.0) | (thrust > max_motor_thrust)))
    thrust = np.clip(thrust, 0.0, max_motor_thrust)
    return np.sqrt(thrust / kf).reshape(1, 4), saturated


# ======================= 4. 평가와 CSV/TXT 저장 =======================
def settling_time(t, errors, tolerance):
    """이후 모든 표본이 허용 범위를 만족하는 첫 시각(해당 구간 내 기준)."""
    ok = np.all(np.abs(errors) <= tolerance, axis=1) if errors.ndim == 2 else (np.abs(errors) <= tolerance)
    bad = np.flatnonzero(~ok)
    start = int(bad[-1] + 1) if len(bad) else 0
    return float(t[start]) if start < len(t) else float("nan")


def rise_time(t, error):
    """초기 오차 90% -> 10%까지 감소 시간."""
    initial = abs(error[0])
    if initial < 1e-12:
        return float("nan")
    a = np.flatnonzero(np.abs(error) <= 0.9 * initial)
    b = np.flatnonzero(np.abs(error) <= 0.1 * initial)
    if not len(a) or not len(b):
        return float("nan")
    b = b[b >= a[0]]
    return float(t[b[0]] - t[a[0]]) if len(b) else float("nan")


def evaluate_and_save(data):
    windows = [
        ("initial_recovery", "초기 복귀", data[data[:, 0] <= INITIAL_EVAL_END + 1e-9]),
        ("long_term_hold", "장시간 유지", data[data[:, 0] > INITIAL_EVAL_END + 1e-9]),
        ("full_run", "전체 실행", data),
    ]
    report = [
        "======= Roll/Pitch/Altitude PD + Yaw 예측 제동 평가 =======",
        "참고: 이동 중 Roll/Pitch 목표각이 0°가 아니므로, 고정 RPY=0° 오차는 자세 추종 오차가 아닙니다.",
        "별도 drone_motion_goal 보고서에서 목표점 도달과 제동을 평가하세요.",
        f"초기 RPY={INITIAL_RPY_DEG.tolist()} deg, 초기 고도={INITIAL_ALTITUDE:.3f}m",
        f"목표 RPY={TARGET_RPY_DEG.tolist()} deg, 목표 고도={TARGET_ALTITUDE:.3f}m",
        f"주파수={SIM_FREQ}Hz | Roll Kp/Kd={KP_ROLL}/{KD_ROLL} | Pitch Kp/Kd={KP_PITCH}/{KD_PITCH}",
        f"Yaw predictive: vmax={MAX_YAW_RATE_DEG_S}deg/s, amax={MAX_YAW_ACCEL_DEG_S2}deg/s², "
        f"braking={YAW_BRAKE_ACCEL_DEG_S2}deg/s², Kv={YAW_RATE_TRACK_K}1/s (미검증)",
        f"고도 Kp/Kd={KP_ALT}/{KD_ALT} (초기 미검증 값) | 기울기 보상={ENABLE_TILT_COMPENSATION}",
        f"평가: 0~{INITIAL_EVAL_END:g}s, {INITIAL_EVAL_END:g}~{DURATION:g}s, 전체",
        f"허용 오차: Roll/Pitch/Yaw ±{SETTLING_TOL_DEG}deg, 고도 ±{ALT_SETTLING_TOL_M}m",
        "주의: 각 구간 정착 시간은 해당 구간 내에서 계속 허용 오차를 만족하는 최초 시각.",
    ]
    rows = []
    motion_rows = []
    axes = [
        ("Roll", "roll_deg", "roll_rate_deg_s", TARGET_RPY_DEG[0], SETTLING_TOL_DEG, "deg"),
        ("Pitch", "pitch_deg", "pitch_rate_deg_s", TARGET_RPY_DEG[1], SETTLING_TOL_DEG, "deg"),
        ("Yaw", "yaw_deg", "yaw_rate_deg_s", TARGET_RPY_DEG[2], SETTLING_TOL_DEG, "deg"),
        ("Altitude", "altitude_m", "vertical_speed_m_s", TARGET_ALTITUDE, ALT_SETTLING_TOL_M, "m"),
    ]
    for section, label, chunk in windows:
        if not len(chunk):
            continue
        t = chunk[:, IDX["time_s"]]
        report.append(f"\n[{label}] {t[0]:.4f}~{t[-1]:.4f}s, {len(chunk)} samples")
        for axis, field, rate_field, target, tol, unit in axes:
            errors = chunk[:, IDX[field]] - target
            if axis == "Yaw":
                errors = np.rad2deg(np.arctan2(
                    np.sin(np.deg2rad(errors)), np.cos(np.deg2rad(errors))))
            speeds = chunk[:, IDX[rate_field]]
            entry = np.flatnonzero(abs(errors) <= tol)
            first_entry = float(t[entry[0]]) if len(entry) else float("nan")
            max_i = int(np.argmax(np.abs(speeds)))
            row = dict(section=section, axis=axis,
                       rise_time_s=rise_time(t, errors) if section == "initial_recovery" else float("nan"),
                       first_entry_s=first_entry,
                       settling_time_s=settling_time(t, errors, tol),
                       max_abs_rate=float(abs(speeds[max_i])), max_rate_at_s=float(t[max_i]),
                       mae=float(np.mean(abs(errors))), max_abs_error=float(max(abs(errors))),
                       final_abs_error=float(abs(errors[-1])),
                       out_of_tolerance_samples=int(np.count_nonzero(abs(errors) > tol)))
            rows.append(row)
            report.append(
                f" {axis:8s} | 최초 진입={first_entry:.4f}s | 정착={row['settling_time_s']:.4f}s"
                f" | 최대 속도={row['max_abs_rate']:.5f}{unit}/s (t={row['max_rate_at_s']:.4f}s)"
                f" | MAE={row['mae']:.6f}{unit} | 최대 오차={row['max_abs_error']:.6f}{unit}"
                f" | 최종 오차={row['final_abs_error']:.6f}{unit}"
                f" | 허용범위 이탈={row['out_of_tolerance_samples']}스텝"
            )
            if section == "initial_recovery":
                report.append(f"          상승 시간={row['rise_time_s']:.4f}s")
        # 제어 루프와 별개로 월드 좌표계 운동 상태를 정량 평가
        for axis_name, velocity_key, accel_key in (
            ("X", "vx_m_s", "ax_m_s2"),
            ("Y", "vy_m_s", "ay_m_s2"),
            ("Z", "vz_m_s", "az_m_s2"),
        ):
            velocity = chunk[:, IDX[velocity_key]]
            acceleration = chunk[:, IDX[accel_key]]
            finite_acc = acceleration[np.isfinite(acceleration)]
            max_speed = float(np.max(np.abs(velocity)))
            max_acc = float(np.max(np.abs(finite_acc))) if len(finite_acc) else float("nan")
            rms_speed = float(np.sqrt(np.mean(velocity**2)))
            motion_rows.append(dict(
                section=section, axis=axis_name,
                max_abs_velocity_m_s=max_speed,
                max_abs_acceleration_m_s2=max_acc,
                rms_velocity_m_s=rms_speed,
                final_velocity_m_s=float(velocity[-1]),
            ))
            report.append(
                f" {axis_name} 이동  | 최대 |v|={max_speed:.5f}m/s"
                f" | 최대 |a|={max_acc:.5f}m/s²"
                f" | RMS 속도={rms_speed:.5f}m/s"
                f" | 최종 v={velocity[-1]:+.5f}m/s"
            )
        attitude_error = chunk[:, [IDX["roll_deg"], IDX["pitch_deg"], IDX["yaw_deg"]]] - TARGET_RPY_DEG
        attitude_error[:, 2] = np.rad2deg(np.arctan2(
            np.sin(np.deg2rad(attitude_error[:, 2])),
            np.cos(np.deg2rad(attitude_error[:, 2]))))
        report.append(
            f" 3축 동시 자세 정착={settling_time(t, attitude_error, SETTLING_TOL_DEG):.4f}s"
            f" | 모터 포화={int(np.count_nonzero(chunk[:, IDX['motor_saturated']]))}스텝"
            f" | 기울기 보상 제한={int(np.count_nonzero(chunk[:, IDX['tilt_comp_limited']]))}스텝"
            f" | 가속도 명령 제한={int(np.count_nonzero(chunk[:, IDX['alt_acc_limited']]))}스텝"
            f" | 고도 변화={chunk[-1, IDX['altitude_m']] - chunk[0, IDX['altitude_m']]:+.4f}m"
            f" | Yaw 변화={chunk[-1, IDX['yaw_deg']] - chunk[0, IDX['yaw_deg']]:+.4f}deg")
    report.append("===================================================")
    txt_path = LOG_DIR / OUTPUT_FILES["attitude_txt"]
    csv_path = LOG_DIR / OUTPUT_FILES["attitude_csv"]
    txt_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    motion_path = LOG_DIR / OUTPUT_FILES["kinematics_csv"]
    with motion_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(motion_rows[0]))
        writer.writeheader()
        writer.writerows(motion_rows)
    print("\n".join(report))
    print(f"평가 TXT: {txt_path}\n평가 CSV: {csv_path}"
          f"\n운동 평가 CSV: {motion_path}")


def evaluate_navigation_and_save(data):
    """고정 목표점 도달 속도/제동/복귀와 공간적 이탈을 평가.

    시간별 기준 경로가 없기 때문에 tracking RMS 같은 시간 오차는 측정하지 않음.
    """
    t = data[:, IDX["time_s"]]
    xy = data[:, [IDX["x_m"], IDX["y_m"]]]
    vel = data[:, [IDX["vx_m_s"], IDX["vy_m_s"]]]
    distance = np.linalg.norm(xy - TARGET_XY_M, axis=1)
    speed = np.linalg.norm(vel, axis=1)
    moving = t >= START_HOVER_S - 1e-9
    settled = (distance <= XY_POSITION_TOL_M) & (speed <= XY_SPEED_TOL_M_S)
    valid_ix = np.flatnonzero(moving)
    first_distance = float('nan')
    first_arrival = float('nan')
    steady_time = float('nan')
    if len(valid_ix):
        d_ix = valid_ix[distance[valid_ix] <= XY_POSITION_TOL_M]
        if len(d_ix): first_distance = float(t[d_ix[0]])
        arrive_ix = valid_ix[settled[valid_ix]]
        if len(arrive_ix): first_arrival = float(t[arrive_ix[0]])
        # Only declare global settling if no later sample violates either condition.
        last_bad = np.flatnonzero(~settled[valid_ix])
        local = int(last_bad[-1] + 1) if len(last_bad) else 0
        if local < len(valid_ix): steady_time = float(t[valid_ix[local]])

    length_vec = TARGET_XY_M - START_XY_M
    direct_length = float(np.linalg.norm(length_vec))
    cross_track_max = float('nan')
    along_overshoot = 0.0
    if direct_length > 1e-9:
        unit = length_vec / direct_length
        displacement = xy - START_XY_M
        along = displacement @ unit
        cross = displacement[:, 0] * unit[1] - displacement[:, 1] * unit[0]
        cross_track_max = float(np.max(np.abs(cross)))
        along_overshoot = max(0.0, float(np.max(along)) - direct_length)
    travelled = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    hold = np.flatnonzero(data[:, IDX["nav_mode"]] == NAV_HOLD)
    held_time = float(t[hold[0]]) if len(hold) else float('nan')
    record = dict(
        start_hover_s=START_HOVER_S,
        max_approach_speed_m_s=MAX_APPROACH_SPEED_M_S,
        brake_accel_m_s2=BRAKE_ACCEL_M_S2,
        accel_limit_m_s2=MAX_XY_ACCEL,
        planned_distance_m=direct_length,
        actual_travelled_distance_m=travelled,
        earliest_goal_distance_entry_s=first_distance,
        earliest_goal_pos_and_vel_entry_s=first_arrival,
        continuous_goal_settling_s=steady_time,
        elapsed_settling_after_hover_s=steady_time-START_HOVER_S if np.isfinite(steady_time) else float('nan'),
        hold_mode_entry_s=held_time,
        max_xy_speed_m_s=float(np.max(speed)),
        max_along_target_overshoot_m=along_overshoot,
        max_deviation_from_start_goal_line_m=cross_track_max,
        final_xy_error_m=float(distance[-1]),
        final_xy_speed_m_s=float(speed[-1]),
        nav_acceleration_limit_samples=int(np.count_nonzero(data[:, IDX["nav_acc_limited"]])),
        brake_guard_samples=int(np.count_nonzero(data[:, IDX["brake_guard_active"]])),
        motor_saturation_samples=int(np.count_nonzero(data[:, IDX["motor_saturated"]])),
        max_altitude_error_m=float(np.max(np.abs(data[:, IDX["altitude_m"]] - TARGET_ALTITUDE))),
    )
    txt_path = LOG_DIR / OUTPUT_FILES["goal_txt"]
    csv_path = LOG_DIR / OUTPUT_FILES["goal_csv"]
    lines = ["======= 고정 목표점 거리 기반 고속 접근 평가 =======",
             f"시작점 {START_XY_M.tolist()} -> 고정 목표점 {TARGET_XY_M.tolist()}m",
             "시간별 위치 경로를 지정하지 않음. 도착 판정은 위치/속도 기준.",
             "최대 시작-목표 직선 이탈은 진단용이며 추종 제약이 아님.",
             "가속도/속도 한계에서의 고속 접근이며 수학적 최소시간 해는 아님."]
    lines += [f"{k}: {v}" for k,v in record.items()]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(record))
        writer.writeheader(); writer.writerow(record)
    print("\n".join(lines))
    print(f"위치접근 평가 TXT: {txt_path}\n위치접근 평가 CSV: {csv_path}")


def save_navigation_plot(data):
    """시각별 궤적 비교가 아닌 거리/속도/속도 상한/실제 XY 이동 표시."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 미설치: PNG 생성 생략. CSV/TXT 결과는 정상 저장됨.")
        return
    t = data[:, IDX["time_s"]]
    xy = data[:, [IDX["x_m"], IDX["y_m"]]]
    vxy = data[:, [IDX["vx_m_s"], IDX["vy_m_s"]]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(t, data[:, IDX["goal_distance_m"]], label="Distance to goal")
    ax[0].axhline(XY_POSITION_TOL_M, linestyle=":", label="Position tolerance")
    ax[0].set(ylabel="Distance (m)", title="Distance-based approach")
    ax[1].plot(t, np.linalg.norm(vxy,axis=1), label="Measured horizontal speed")
    ax[1].plot(t, data[:, IDX["approach_speed_ref_m_s"]], label="Allowed approach speed")
    ax[1].plot(t, data[:, IDX["closing_speed_m_s"]], label="Actual closing speed", linestyle="--")
    ax[1].set(ylabel="Speed (m/s)")
    ax[2].plot(t, data[:, IDX["radial_acc_cmd_m_s2"]], label="Commanded radial accel")
    ax[2].plot(t, data[:, IDX["accel_cmd_norm_m_s2"]], label="Total commanded accel norm")
    ax[2].set(xlabel="Time (s)", ylabel="Acceleration (m/s²)")
    ax[3].plot(xy[:,0], xy[:,1], label="Actual XY", linewidth=1.8)
    ax[3].plot([START_XY_M[0],TARGET_XY_M[0]],
               [START_XY_M[1],TARGET_XY_M[1]], linestyle=":", label="Direct line (reference only)")
    ax[3].scatter([START_XY_M[0],TARGET_XY_M[0]],
                  [START_XY_M[1],TARGET_XY_M[1]], label="Start / goal")
    ax[3].set(xlabel="X (m)", ylabel="Y (m)")
    ax[3].axis("equal")
    for a in ax:
        a.grid(True,alpha=.3);a.legend()
    for a in ax[:3]:
        a.axvline(START_HOVER_S, linestyle=":", linewidth=1)
    png = LOG_DIR / OUTPUT_FILES["plot"]
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"위치 접근 결과 PNG: {png}")


# ======================= 5. PyBullet 카메라 =======================
def camera_step_interval(update_hz):
    """240Hz 물리 루프에서 목표 GUI 작업 주파수의 정수 스텝 간격."""
    if update_hz <= 0:
        raise ValueError("카메라/GUI 갱신 주파수는 0보다 커야 합니다.")
    return max(1, round(SIM_FREQ / update_hz))


def initialize_camera(physics_client_id):
    """자유 카메라에서는 이 함수가 카메라를 설정하는 유일한 시점이다."""
    if CAMERA_MODE not in ("free", "follow"):
        raise ValueError('CAMERA_MODE는 "free" 또는 "follow"만 가능합니다.')
    initial_focus = np.array([
        (START_XY_M[0] + TARGET_XY_M[0]) / 2.0,
        (START_XY_M[1] + TARGET_XY_M[1]) / 2.0,
        INITIAL_ALTITUDE,
    ], dtype=float)
    if CAMERA_MODE == "follow":
        initial_focus = np.array([*START_XY_M, INITIAL_ALTITUDE], dtype=float)
    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=initial_focus.tolist(),
        physicsClientId=physics_client_id,
    )
    return initial_focus


def update_follow_camera(physics_client_id, position_world, previous_target, update_interval_steps):
    """드론을 따라가되 사용자가 조절한 yaw/pitch/zoom 값은 보존한다.

    getDebugVisualizerCamera 반환 튜플에서 [8:12]는 yaw, pitch, 거리, 초점.
    카메라 위치 타깃은 지수평활로 완충한다. (물리/제어에는 영향 없음)
    """
    dt_update = update_interval_steps / SIM_FREQ
    if CAMERA_FOLLOW_TAU_S <= 0.0:
        alpha = 1.0
    else:
        alpha = 1.0 - np.exp(-dt_update / CAMERA_FOLLOW_TAU_S)
    new_target = np.asarray(previous_target, dtype=float) + alpha * (
        np.asarray(position_world, dtype=float) - np.asarray(previous_target, dtype=float))
    cam = p.getDebugVisualizerCamera(physicsClientId=physics_client_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=float(cam[10]),
        cameraYaw=float(cam[8]),
        cameraPitch=float(cam[9]),
        cameraTargetPosition=new_target.tolist(),
        physicsClientId=physics_client_id,
    )
    return new_target


def main():
    if EXPERIMENT_MODE == "yaw_coupling_ab":
        return run_yaw_coupling_ab()
    if EXPERIMENT_MODE == "combo_search":
        return run_combo_search()
    if EXPERIMENT_MODE == "settling_tune":
        return run_settling_tuning()
    if EXPERIMENT_MODE == "progressive_tune":
        return run_progressive_tuning()
    if EXPERIMENT_MODE == "all_axes_batch":
        return run_all_axes_batch()
    if EXPERIMENT_MODE == "yaw_batch":
        return run_yaw_batch_experiment()
    if EXPERIMENT_MODE != "navigation":
        raise ValueError(f"알 수 없는 EXPERIMENT_MODE={EXPERIMENT_MODE!r}")
    if DURATION <= START_HOVER_S:
        raise ValueError("DURATION은 START_HOVER_S보다 길어야 합니다.")
    if BRAKE_ACCEL_M_S2 <= 0 or MAX_XY_ACCEL <= 0:
        raise ValueError("제동 가속도와 수평 가속도 한계는 0보다 커야 합니다.")
    init_rpy = np.deg2rad(INITIAL_RPY_DEG)
    target_rpy = np.deg2rad(TARGET_RPY_DEG)
    dt = 1.0 / SIM_FREQ
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([[*START_XY_M, INITIAL_ALTITUDE]]),
        initial_rpys=init_rpy.reshape(1, 3),
        pyb_freq=SIM_FREQ, ctrl_freq=SIM_FREQ, gui=GUI,
        user_debug_gui=False,
    )
    gravity_force = float(env.GRAVITY)
    mass = float(env.M)
    gravity_acc = gravity_force / mass
    arm = float(env.L / np.sqrt(2.0))
    max_motor_thrust = float(env.KF * env.MAX_RPM ** 2)
    km = float(env.KM)
    if km <= 0:
        raise ValueError("CF2X Yaw 모터 토크 계수 KM이 0 이하입니다.")

    action = np.full((1,4), np.sqrt((gravity_force/4.0) / env.KF))
    nav_state = NAV_HOVER
    stable_elapsed_s = 0.0
    initial_alt = altitude_pd(INITIAL_ALTITUDE, 0., TARGET_ALTITUDE,
                              mass, gravity_acc, init_rpy[0], init_rpy[1])
    initial_xy_acc, initial_diag = point_to_point_controller(
        START_XY_M, np.zeros(2), nav_state, SPEED_SCALE)
    initial_xy_error = TARGET_XY_M - START_XY_M
    initial_roll_target, initial_pitch_target, initial_xy_acc = xy_acceleration_to_attitude(
        initial_xy_acc, init_rpy[2], initial_alt[3], gravity_acc)
    initial_roll = roll_pd(init_rpy[0], 0., initial_roll_target)
    initial_pitch = pitch_pd(init_rpy[1], 0., initial_pitch_target)
    yaw_inertia = float(env.J[2,2])
    if yaw_inertia <= 0:
        raise ValueError("Yaw 관성모멘트 Izz가 0 이하입니다.")
    positive_limit, negative_limit = yaw_torque_limits(
        initial_alt[4],initial_roll[3],initial_pitch[3],arm,env.KF,km,max_motor_thrust)
    initial_yaw = yaw_predictive_control(
        init_rpy[2], 0.0, target_rpy[2], yaw_inertia, positive_limit, negative_limit)
    log = [[
        0., *INITIAL_RPY_DEG, 0.,0.,0.,
        np.rad2deg(initial_roll[0]),np.rad2deg(initial_pitch[0]), INITIAL_ALTITUDE,
        *initial_roll[1:],*initial_pitch[1:],*action[0],0.,
        0., initial_alt[0],initial_alt[1],initial_alt[2],initial_alt[3],
        initial_alt[4],initial_alt[5],float(initial_alt[6]),float(initial_alt[7]),
        np.rad2deg(initial_yaw[0]),*initial_yaw[1:4],
        *START_XY_M, 0.,0.,0., float('nan'),float('nan'),float('nan'),
        np.rad2deg(initial_yaw[4]),np.rad2deg(initial_yaw[5]),
        np.rad2deg(initial_yaw[6]),float('nan'),float(initial_yaw[7]),
        initial_yaw[8],initial_yaw[9],
        *TARGET_XY_M,float(np.linalg.norm(initial_xy_error)),0.,
        *initial_xy_acc, np.rad2deg(initial_roll_target),np.rad2deg(initial_pitch_target),
        float(nav_state),initial_diag['distance'],initial_diag['closing_speed'],
        initial_diag['lateral_speed'],initial_diag['target_speed'],
        initial_diag['ideal_stop_distance'],initial_diag['margin'],
        initial_diag['radial_acc'],initial_diag['lateral_acc'],
        float(initial_diag['limited']),SPEED_SCALE,initial_diag['acceleration_norm'],
        float(initial_diag['emergency_brake']),
    ]]
    prev_world_vel = np.zeros(3)
    prev_yaw_rate = 0.0
    filtered_yaw_acc = 0.0

    camera_interval = camera_step_interval(CAMERA_UPDATE_HZ)
    label_interval = camera_step_interval(ATTITUDE_LABEL_UPDATE_HZ)
    trail_interval = camera_step_interval(TRAIL_UPDATE_HZ)
    console_interval = camera_step_interval(CONSOLE_LOG_HZ)
    camera_target = np.array([*START_XY_M,INITIAL_ALTITUDE],dtype=float)
    if GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=env.CLIENT)
        camera_target = initialize_camera(env.CLIENT)
        print(f"카메라 모드: {CAMERA_MODE} | 자동 추적 갱신: "
              f"{SIM_FREQ/camera_interval:g}Hz (follow 전용)")
    text_id = -1
    last_draw_position = np.array([*START_XY_M,INITIAL_ALTITUDE],dtype=float)
    if GUI and SHOW_MOVEMENT_LINES:
        # 목표점 표시용 십자선. 경로를 강제하는 직선은 그리지 않는다.
        tx, ty = TARGET_XY_M
        p.addUserDebugLine([tx-.35,ty,TARGET_ALTITUDE],
                           [tx+.35,ty,TARGET_ALTITUDE],
                           lineColorRGB=[0.,.7,0.],lineWidth=3,lifeTime=0,
                           physicsClientId=env.CLIENT)
        p.addUserDebugLine([tx,ty-.35,TARGET_ALTITUDE],
                           [tx,ty+.35,TARGET_ALTITUDE],
                           lineColorRGB=[0.,.7,0.],lineWidth=3,lifeTime=0,
                           physicsClientId=env.CLIENT)
    start = time.time()
    try:
        for i in range(int(DURATION*SIM_FREQ)):
            obs,_,terminated,truncated,_ = env.step(action)
            pos,world_vel,rpy,body_rates = measure_state(obs[0])
            world_acc = (world_vel-prev_world_vel)/dt
            prev_world_vel=world_vel.copy()
            alt=altitude_pd(pos[2],world_vel[2],TARGET_ALTITUDE,
                            mass,gravity_acc,rpy[0],rpy[1])
            t=(i+1)*dt
            goal_delta=TARGET_XY_M-pos[:2]
            goal_distance=float(np.linalg.norm(goal_delta))
            goal_speed=float(np.linalg.norm(world_vel[:2]))
            nav_state,stable_elapsed_s = update_navigation_state(
                nav_state,t,goal_distance,goal_speed,stable_elapsed_s,dt)
            xy_acc,nav_diag=point_to_point_controller(
                pos[:2],world_vel[:2],nav_state,SPEED_SCALE)
            roll_target,pitch_target,xy_acc=xy_acceleration_to_attitude(
                xy_acc,rpy[2],alt[3],gravity_acc)
            roll=roll_pd(rpy[0],body_rates[0],roll_target)
            pitch=pitch_pd(rpy[1],body_rates[1],pitch_target)
            raw_yaw_acc=(body_rates[2]-prev_yaw_rate)/dt
            filtered_yaw_acc+=dt/(YAW_ACCEL_FILTER_TAU_S+dt)*(
                raw_yaw_acc-filtered_yaw_acc)
            prev_yaw_rate=float(body_rates[2])
            positive_limit,negative_limit=yaw_torque_limits(
                alt[4],roll[3],pitch[3],arm,env.KF,km,max_motor_thrust)
            yaw=yaw_predictive_control(rpy[2],body_rates[2],target_rpy[2],
                                        yaw_inertia,positive_limit,negative_limit)
            action,saturated=allocate_motor_rpm(
                roll[3],pitch[3],yaw[3],alt[4],arm,env.KF,km,max_motor_thrust)
            log.append([
                t,*np.rad2deg(rpy),*np.rad2deg(body_rates),
                np.rad2deg(roll[0]),np.rad2deg(pitch[0]),pos[2],
                *roll[1:],*pitch[1:],*action[0],float(saturated),
                world_vel[2],alt[0],alt[1],alt[2],alt[3],
                alt[4],alt[5],float(alt[6]),float(alt[7]),
                np.rad2deg(yaw[0]),*yaw[1:4],
                pos[0],pos[1],*world_vel,*world_acc,
                np.rad2deg(yaw[4]),np.rad2deg(yaw[5]),
                np.rad2deg(yaw[6]),np.rad2deg(filtered_yaw_acc),
                float(yaw[7]),yaw[8],yaw[9],
                *TARGET_XY_M,goal_distance,goal_speed,
                *xy_acc,np.rad2deg(roll_target),np.rad2deg(pitch_target),
                float(nav_state),nav_diag['distance'],nav_diag['closing_speed'],
                nav_diag['lateral_speed'],nav_diag['target_speed'],
                nav_diag['ideal_stop_distance'],nav_diag['margin'],
                nav_diag['radial_acc'],nav_diag['lateral_acc'],
                float(nav_diag['limited']),SPEED_SCALE,nav_diag['acceleration_norm'],
                float(nav_diag['emergency_brake'] and nav_state==NAV_APPROACH),
            ])
            # free: 카메라 강제 리셋 없음 / follow: 사용자 yaw,pitch,zoom 유지
            if GUI and CAMERA_MODE=='follow' and i%camera_interval==0:
                camera_target=update_follow_camera(env.CLIENT,pos,camera_target,camera_interval)
            if GUI and SHOW_MOVEMENT_LINES and i%trail_interval==0:
                p.addUserDebugLine(last_draw_position.tolist(),pos.tolist(),
                    lineColorRGB=[1.,.2,.2],lineWidth=2,lifeTime=0,physicsClientId=env.CLIENT)
                last_draw_position=pos.copy()
            if GUI and SHOW_ATTITUDE_LABEL and i%label_interval==0:
                text_id=p.addUserDebugText(
                    text=(f"R {np.rad2deg(rpy[0]):+.1f} | P {np.rad2deg(rpy[1]):+.1f} "
                          f"| Y {np.rad2deg(rpy[2]):+.1f} | Z {pos[2]:.2f}m "
                          f"| dist {goal_distance:.2f}m"),
                    textPosition=[0,0,.1],textColorRGB=[0,0,0],textSize=1.1,
                    parentObjectUniqueId=int(env.DRONE_IDS[0]),
                    replaceItemUniqueId=text_id,physicsClientId=env.CLIENT)
            if i%console_interval==0:
                print(f"t={t:5.2f}s | R/P/Y={np.rad2deg(rpy).round(2)}deg"
                      f" | XY=({pos[0]:+.3f},{pos[1]:+.3f})m"
                      f" | distance={goal_distance:.3f}m | speed={goal_speed:.3f}m/s"
                      f" | closing={nav_diag['closing_speed']:.3f}m/s"
                      f" | v_ref={nav_diag['target_speed']:.3f}m/s"
                      f" | mode={nav_state} | Z={pos[2]:.3f}m"
                      f" | motor_saturation={saturated}")
            if REAL_TIME_PLAYBACK:
                remaining=t-(time.time()-start)
                if remaining>0: time.sleep(remaining)
            if terminated or truncated: break
    finally:
        env.close()
        bad_rows=[(idx,len(r)) for idx,r in enumerate(log) if len(r)!=len(HEADER)]
        if bad_rows:
            raise ValueError(f"CSV 열 불일치: HEADER={len(HEADER)}; bad={bad_rows[:5]}")
        csv_path=LOG_DIR / OUTPUT_FILES["raw"]
        with csv_path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(HEADER);writer.writerows(log)
        print(f"원본 CSV: {csv_path}")
    arr=np.asarray(log,dtype=float)
    evaluate_and_save(arr)
    evaluate_navigation_and_save(arr)
    if SAVE_NAVIGATION_PLOT:
        save_navigation_plot(arr)
    print(f"XY 최종 위치: ({arr[-1,IDX['x_m']]:.4f}, {arr[-1,IDX['y_m']]:.4f}) m")
    print(f"XY 최종 오차: {arr[-1,IDX['goal_distance_m']]:.4f}m"
          f", 최종 속도: {arr[-1,IDX['xy_speed_norm_m_s']]:.4f}m/s")
    print(f"로그 폴더: {LOG_DIR}")




# ======================= 8. Yaw 성능 계측 + 목표각 자동 스윕 =======================
# 1) calibration: 초당 Yaw 각가속도, 펄스별 최고 각속도, 역토크 제동시간/각도
# 2) target: 각 목표값, 방향, 토크 사용률, 감속 계획 계수 스윕
# 출력: logs/drone_motion_yaw.csv, drone_motion_yaw_trials.csv,
#       drone_motion_yaw.txt, drone_motion_yaw.png, drone_motion_yaw_config.json
# 모든 실험은 env.reset()으로 같은 초기 위치/속도/자세에서 시작한다.

YAW_BATCH_FIELDS = [
    'case_id','case_type','phase','step','trial_time_s',
    'direction','torque_fraction','pulse_duration_s','target_yaw_deg',
    'brake_model_scale','brake_alpha_calib_deg_s2',
    'yaw_unwrapped_deg','yaw_wrapped_deg','yaw_rate_deg_s','yaw_acc_deg_s2',
    'yaw_acc_filtered_deg_s2','yaw_error_deg','stop_angle_plan_deg',
    'yaw_torque_requested_Nm','yaw_torque_realized_Nm',
    'yaw_torque_positive_available_Nm','yaw_torque_negative_available_Nm',
    'motor_1_rpm','motor_2_rpm','motor_3_rpm','motor_4_rpm','motor_saturated',
    'roll_deg','pitch_deg','roll_rate_deg_s','pitch_rate_deg_s',
    'z_m','vz_m_s','x_m','y_m','vx_m_s','vy_m_s',
]
YAW_BATCH_SUMMARY_FIELDS = [
    'case_id','case_type','direction','torque_fraction','pulse_duration_s',
    'target_yaw_deg','brake_model_scale','status','reason',
    'duration_s','peak_abs_yaw_rate_deg_s','peak_abs_yaw_acc_deg_s2',
    'accel_signed_median_deg_s2','brake_decel_p10_deg_s2',
    'brake_start_rate_deg_s','brake_time_s','brake_angle_deg',
    'brake_effective_acc_deg_s2',
    'settle_time_s','overshoot_deg','overshoot_any','overshoot_gt_tol',
    'final_yaw_error_deg','final_yaw_rate_deg_s',
    'max_roll_deg','max_pitch_deg','max_altitude_error_m','max_xy_drift_m',
    'max_motor_rpm','motor_saturation_steps','n_samples',
]


def _yaw_wrap_delta_deg(a):
    """-180 <= 회전 변화량 <= 180. 샘플 간 실제 회전은 180도 미만이어야 한다."""
    return (float(a) + 180.0) % 360.0 - 180.0


def _yaw_batch_steps(seconds):
    return max(1, int(round(seconds * SIM_FREQ)))


class _YawBatchRig:
    """기존 내부 PD/모터 할당 함수를 재사용하는 반복 실험기."""
    def __init__(self, env, writer):
        self.env = env
        self.writer = writer
        self.dt = 1.0 / SIM_FREQ
        self.mass = float(env.M)
        self.kf = float(env.KF)
        self.km = float(env.KM)
        self.izz = float(env.J[2, 2])
        self.arm = float(env.L / np.sqrt(2.0))
        self.mg = float(env.GRAVITY)
        self.gravity = self.mg / self.mass
        self.max_f = self.kf * float(env.MAX_RPM)**2
        self.total_rows = 0
        if min(self.kf, self.km, self.izz, self.max_f) <= 0:
            raise ValueError('드론 계수 KF/KM/Izz/MAX_RPM이 유효하지 않습니다.')
        self.new_case()

    def new_case(self):
        obs, _ = self.env.reset()
        self.state = np.asarray(obs[0], dtype=float)
        self.t = 0.0
        self.step_index = 0
        self.yaw_wrapped_prev = float(np.rad2deg(self.state[9]))
        self.yaw_unwrapped = 0.0
        self.yaw_initial_deg = self.yaw_wrapped_prev
        _, _, _, omega = measure_state(self.state)
        self.prev_rate = float(np.rad2deg(omega[2]))
        self.filtered_acc = 0.0

    def _available_torque(self, t_total, tau_roll, tau_pitch):
        """원래 yaw_torque_limits의 reserve=0.75를 사용하지 않고 RPM 물리 한계만 반영."""
        dr = tau_roll / (4.0*self.arm)
        dp = tau_pitch / (4.0*self.arm)
        f_base = (np.full(4, t_total/4.0)
                  + dr*np.array([-1.,-1.,1.,1.])
                  + dp*np.array([-1.,1.,1.,-1.]))
        plus = np.array([-1.,1.,-1.,1.])
        pos = min(np.min(self.max_f-f_base[plus>0]), np.min(f_base[plus<0]))
        neg = min(np.min(f_base[plus>0]), np.min(self.max_f-f_base[plus<0]))
        coef = 4.0*self.km/self.kf
        return max(0.,float(pos))*coef, max(0.,float(neg))*coef

    def step(self, case, phase, yaw_cmd_mode, sign=0, fraction=1.0,
             target=0., alpha=0., scale=1., terminal=False):
        """한 물리 스텝: 상태 -> 토크 -> RPM -> env.step -> 기록.

        yaw_cmd_mode: idle, torque, brake, bang. bang은 목표까지 제동거리 검사.
        반환: 최신 CSV dict (사후 측정), 비정상 상태 발생 시 _abort_reason.
        """
        pos, vel, rpy, omega = measure_state(self.state)
        alt = altitude_pd(pos[2],vel[2],YAW_BATCH_START_ALT_M,
                          self.mass,self.gravity,rpy[0],rpy[1])
        r = roll_pd(rpy[0],omega[0],0.0)
        pit = pitch_pd(rpy[1],omega[1],0.0)
        tau_pos, tau_neg = self._available_torque(alt[4],r[3],pit[3])
        wdeg = float(np.rad2deg(omega[2]))
        error = float(target - self.yaw_unwrapped)
        planned_stop = float('nan')
        if yaw_cmd_mode == 'torque':
            tau = fraction*(tau_pos if sign>=0 else -tau_neg)
        elif yaw_cmd_mode == 'brake':
            tau_need = -self.izz*float(omega[2])/self.dt
            tau = float(np.clip(tau_need,-fraction*tau_neg,fraction*tau_pos))
        elif yaw_cmd_mode == 'bang':
            towards_speed = np.sign(error)*wdeg
            decel_model = max(1.0, alpha * fraction * scale)
            if towards_speed > 0.0:
                planned_stop = (towards_speed**2/(2.*decel_model)
                                + towards_speed*YAW_BATCH_BRAKE_DELAY_S
                                + YAW_BATCH_BRAKE_MARGIN_DEG)
            else:
                planned_stop = YAW_BATCH_BRAKE_MARGIN_DEG
            # 근접 구간에서는 감속과 정밀 PD. 속도가 여전히 크면 bang 제동 유지.
            if terminal:
                acc_req = (YAW_BATCH_TERMINAL_KP*error
                           - YAW_BATCH_TERMINAL_KD*wdeg)  # deg/s²
                tau_need = self.izz*np.deg2rad(acc_req)
                tau = float(np.clip(tau_need,-fraction*tau_neg,fraction*tau_pos))
            elif towards_speed < -1.0 or (towards_speed > 0 and abs(error) <= planned_stop):
                tau = fraction*(-tau_neg if wdeg>0 else tau_pos)
            else:
                tau = fraction*(tau_pos if error>=0 else -tau_neg)
        else:  # hover/unpowered yaw
            tau = 0.0

        rpm, sat = allocate_motor_rpm(r[3],pit[3],float(tau),alt[4],
                                       self.arm,self.kf,self.km,self.max_f)
        rpmv = rpm.reshape(4)
        actual_tau = self.km*float(-rpmv[0]**2+rpmv[1]**2-rpmv[2]**2+rpmv[3]**2)
        obs, _, terminated, truncated, _ = self.env.step(rpm)
        self.state = np.asarray(obs[0],dtype=float)
        new_pos,new_vel,new_rpy,new_omega=measure_state(self.state)
        self.t += self.dt
        self.step_index += 1
        yaw_wrapped = float(np.rad2deg(new_rpy[2]))
        self.yaw_unwrapped += _yaw_wrap_delta_deg(yaw_wrapped-self.yaw_wrapped_prev)
        self.yaw_wrapped_prev = yaw_wrapped
        new_rate = float(np.rad2deg(new_omega[2]))
        yaw_acc = (new_rate-self.prev_rate)/self.dt
        self.filtered_acc += self.dt/(0.02+self.dt)*(yaw_acc-self.filtered_acc)
        self.prev_rate = new_rate
        row = {
            'case_id':case['case_id'],'case_type':case['case_type'],
            'phase':phase,'step':self.step_index,'trial_time_s':self.t,
            'direction':case['direction'],'torque_fraction':case['torque_fraction'],
            'pulse_duration_s':case.get('pulse_duration_s',''),
            'target_yaw_deg':case.get('target_yaw_deg',''),
            'brake_model_scale':case.get('brake_model_scale',''),
            'brake_alpha_calib_deg_s2':alpha if alpha else '',
            'yaw_unwrapped_deg':self.yaw_unwrapped,
            'yaw_wrapped_deg':yaw_wrapped,'yaw_rate_deg_s':new_rate,
            'yaw_acc_deg_s2':yaw_acc,'yaw_acc_filtered_deg_s2':self.filtered_acc,
            'yaw_error_deg':target-self.yaw_unwrapped,
            'stop_angle_plan_deg':planned_stop,
            'yaw_torque_requested_Nm':tau,'yaw_torque_realized_Nm':actual_tau,
            'yaw_torque_positive_available_Nm':tau_pos,
            'yaw_torque_negative_available_Nm':tau_neg,
            'motor_1_rpm':rpmv[0],'motor_2_rpm':rpmv[1],
            'motor_3_rpm':rpmv[2],'motor_4_rpm':rpmv[3],
            'motor_saturated':int(sat),
            'roll_deg':np.rad2deg(new_rpy[0]),'pitch_deg':np.rad2deg(new_rpy[1]),
            'roll_rate_deg_s':np.rad2deg(new_omega[0]),
            'pitch_rate_deg_s':np.rad2deg(new_omega[1]),
            'z_m':new_pos[2],'vz_m_s':new_vel[2],
            'x_m':new_pos[0],'y_m':new_pos[1],
            'vx_m_s':new_vel[0],'vy_m_s':new_vel[1],
        }
        self.writer.writerow(row)
        self.total_rows += 1
        fail = ''
        if not np.all(np.isfinite([self.yaw_unwrapped,new_rate,yaw_acc,*new_pos,*new_rpy])):
            fail = 'nonfinite_state'
        elif abs(new_rate)>YAW_BATCH_ABORT_RATE_DEG_S:
            fail = 'rate_guard'
        elif max(abs(row['roll_deg']),abs(row['pitch_deg']))>YAW_BATCH_ABORT_TILT_DEG:
            fail = 'tilt_guard'
        elif abs(new_pos[2]-YAW_BATCH_START_ALT_M)>YAW_BATCH_ABORT_Z_ERROR_M:
            fail = 'altitude_guard'
        elif np.linalg.norm(new_pos[:2])>YAW_BATCH_ABORT_XY_DRIFT_M:
            fail = 'xy_guard'
        elif terminated or truncated:
            fail = 'environment_terminated'
        return row,fail


def _yaw_batch_empty_summary(case):
    result={k:'' for k in YAW_BATCH_SUMMARY_FIELDS}
    result.update({k:v for k,v in case.items() if k in result})
    return result


def _yaw_batch_fill_common(result, samples):
    if not samples:
        result['n_samples']=0
        return
    a = lambda key: np.array([r[key] for r in samples],dtype=float)
    result.update({
        'duration_s':samples[-1]['trial_time_s'],
        'peak_abs_yaw_rate_deg_s':float(np.max(np.abs(a('yaw_rate_deg_s')))),
        'peak_abs_yaw_acc_deg_s2':float(np.max(np.abs(a('yaw_acc_deg_s2')))),
        'final_yaw_rate_deg_s':samples[-1]['yaw_rate_deg_s'],
        'max_roll_deg':float(np.max(np.abs(a('roll_deg')))),
        'max_pitch_deg':float(np.max(np.abs(a('pitch_deg')))),
        'max_altitude_error_m':float(np.max(np.abs(a('z_m')-YAW_BATCH_START_ALT_M))),
        'max_xy_drift_m':float(np.max(np.hypot(a('x_m'),a('y_m')))),
        'max_motor_rpm':max(float(max(r[f'motor_{i}_rpm'] for i in range(1,5))) for r in samples),
        'motor_saturation_steps':int(np.sum(a('motor_saturated'))),
        'n_samples':len(samples),
    })


def _yaw_batch_calibration(rig, case):
    samples=[]; abort=''; direction=case['direction']
    for _ in range(_yaw_batch_steps(YAW_BATCH_WARMUP_S)):
        row,abort=rig.step(case,'warmup','idle'); samples.append(row)
        if abort: break
    if not abort:
        for _ in range(_yaw_batch_steps(case['pulse_duration_s'])):
            row,abort=rig.step(case,'accel','torque',direction,case['torque_fraction'])
            samples.append(row)
            if abort: break
    if not abort:
        for _ in range(_yaw_batch_steps(YAW_BATCH_COAST_S)):
            row,abort=rig.step(case,'coast','idle');samples.append(row)
            if abort:break
    brake_begin=None; brake_yaw=None; brake_rate=None; brake_elapsed=0.0
    if not abort:
        brake_begin=rig.t; brake_yaw=rig.yaw_unwrapped; brake_rate=rig.prev_rate
        for _ in range(_yaw_batch_steps(YAW_BATCH_BRAKE_TIMEOUT_S)):
            if abs(rig.prev_rate)<YAW_BATCH_BRAKE_STOP_RATE_DEG_S:break
            row,abort=rig.step(case,'brake','brake',fraction=case['torque_fraction'])
            samples.append(row)
            if abort:break
        brake_elapsed=rig.t-brake_begin
    result=_yaw_batch_empty_summary(case)
    _yaw_batch_fill_common(result,samples)
    accel=np.array([direction*r['yaw_acc_deg_s2'] for r in samples if r['phase']=='accel'])
    decel=np.array([-direction*r['yaw_acc_deg_s2'] for r in samples
                    if r['phase']=='brake' and direction*r['yaw_rate_deg_s']>25.0])
    decel=decel[np.isfinite(decel)&(decel>0)]
    result['accel_signed_median_deg_s2']=float(np.median(accel)) if len(accel) else ''
    result['brake_decel_p10_deg_s2']=float(np.percentile(decel,10)) if len(decel) else ''
    result['brake_start_rate_deg_s']=brake_rate if brake_rate is not None else ''
    result['brake_time_s']=brake_elapsed
    if brake_yaw is not None:
        dist=direction*(rig.yaw_unwrapped-brake_yaw)
        result['brake_angle_deg']=float(dist)
        if dist>1e-5:
            result['brake_effective_acc_deg_s2']=float(max(0.,brake_rate**2-rig.prev_rate**2)/(2*dist))
    result['status']='ABORT' if abort else ('PASS' if abs(rig.prev_rate)<YAW_BATCH_BRAKE_STOP_RATE_DEG_S else 'TIMEOUT')
    result['reason']=abort or ('braking_not_finished' if result['status']=='TIMEOUT' else '')
    return result


def _yaw_batch_target(rig,case,alpha):
    samples=[];abort='';stable=0.;settled_at=None;terminal=False
    target=float(case['target_yaw_deg'])
    min_total_steps=_yaw_batch_steps(YAW_BATCH_TARGET_TIMEOUT_S)
    for _ in range(_yaw_batch_steps(YAW_BATCH_WARMUP_S)):
        row,abort=rig.step(case,'warmup','idle',target=target)
        samples.append(row)
        if abort:break
    settle_end=None
    if not abort:
        for _ in range(min_total_steps):
            err=target-rig.yaw_unwrapped
            # terminal 진입은 위치·속도 조건 모두 만족했을 때만. 진입 후 유지.
            if abs(err)<=YAW_BATCH_TERMINAL_START_ANGLE_DEG and abs(rig.prev_rate)<=YAW_BATCH_TERMINAL_START_RATE_DEG_S:
                terminal=True
            row,abort=rig.step(case,'terminal' if terminal else 'bang', 'bang',
                               fraction=case['torque_fraction'],target=target,
                               alpha=alpha,scale=case['brake_model_scale'],terminal=terminal)
            samples.append(row)
            if abort:break
            if abs(row['yaw_error_deg'])<=YAW_BATCH_ARRIVAL_ANGLE_DEG and abs(row['yaw_rate_deg_s'])<=YAW_BATCH_ARRIVAL_RATE_DEG_S:
                stable+=rig.dt
            else:
                stable=0.
            # 처음 정착한 뒤에도 조건이 깨지면 정착 판정을 무효화하고 재평가한다.
            if settled_at is not None and stable == 0.0:
                settled_at = None
                settle_end = None
            if settled_at is None and stable>=YAW_BATCH_SETTLE_HOLD_S:
                settled_at=rig.t-YAW_BATCH_SETTLE_HOLD_S
                settle_end=rig.t+YAW_BATCH_POST_SETTLE_OBSERVE_S
            if settle_end is not None and rig.t>=settle_end:
                break
    result=_yaw_batch_empty_summary(case)
    _yaw_batch_fill_common(result,samples)
    relevant=[r for r in samples if r['phase']!='warmup']
    overshoot=max([0.]+[max(0.,case['direction']*(r['yaw_unwrapped_deg']-target)) for r in relevant])
    result.update({
        'settle_time_s':settled_at if settled_at is not None else '',
        'overshoot_deg':float(overshoot),
        'overshoot_any':int(overshoot>1e-6),
        'overshoot_gt_tol':int(overshoot>YAW_BATCH_OVERSHOOT_TOL_DEG),
        'final_yaw_error_deg':float(target-rig.yaw_unwrapped),
    })
    if abort:
        result['status']='ABORT';result['reason']=abort
    elif settled_at is None:
        result['status']='TIMEOUT';result['reason']='not_settled'
    elif overshoot>YAW_BATCH_OVERSHOOT_TOL_DEG:
        result['status']='OVERSHOOT';result['reason']='overshoot_exceeds_tolerance'
    elif overshoot>1e-6:
        # 사용자 목표는 '오버슈트 없음'. 허용오차 이내의 작은 교차도 PASS로 취급하지 않음.
        result['status']='SMALL_OVERSHOOT';result['reason']='crossed_target_below_tolerance'
    elif (abs(result['final_yaw_error_deg'])>YAW_BATCH_ARRIVAL_ANGLE_DEG or
          abs(rig.prev_rate)>YAW_BATCH_ARRIVAL_RATE_DEG_S):
        result['status']='UNSTABLE';result['reason']='drift_after_settle'
    else:
        result['status']='PASS';result['reason']=''
    return result,samples


def _yaw_batch_report(summaries,calibration,rig,env,run_info):
    import json
    from collections import defaultdict
    from matplotlib import pyplot as plt
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    summary_path=LOG_DIR/'drone_motion_yaw_trials.csv'
    with summary_path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=YAW_BATCH_SUMMARY_FIELDS,extrasaction='ignore')
        writer.writeheader();writer.writerows(summaries)
    targets=[s for s in summaries if s['case_type']=='target']
    passing=[s for s in targets if s['status']=='PASS']
    by_group=defaultdict(list)
    for s in targets:
        by_group[(abs(float(s['target_yaw_deg'])),s['torque_fraction'],s['brake_model_scale'])].append(s)
    winners=[]
    for angle in YAW_BATCH_TARGET_ANGLES_DEG:
        opts=[]
        for (a,fraction,scale),grp in by_group.items():
            if a==angle and len(grp)==2 and all(q['status']=='PASS' for q in grp):
                worst=max(float(q['settle_time_s']) for q in grp)
                opts.append((worst,-float(fraction),float(scale)))
        if opts:
            worst,neg_fraction,scale=min(opts)
            winners.append((angle,-neg_fraction,scale,worst))
    calib_min={}
    for direction in YAW_BATCH_DIRECTIONS:
        valid=[s for s in calibration if s['direction']==direction and s['status']=='PASS'
               and s['torque_fraction']==1.0 and s['pulse_duration_s']>=0.05
               and isinstance(s['brake_decel_p10_deg_s2'],float)
               and s['brake_decel_p10_deg_s2']>0]
        calib_min[direction]=min([float(s['brake_decel_p10_deg_s2']) for s in valid],default=None)
    lines=[
        'CF2X Yaw batch: motor-capability + target overshoot/settling experiments',
        f'Cases: {len(summaries)}; calibration={len(calibration)}, target={len(targets)}; logged 240Hz rows={rig.total_rows}',
        'Each trial uses env.reset(): same initial pose and zero velocity.',
        'No Yaw software 45deg/s, 60deg/s2, 35deg/s2 or 0.75 torque reserve in batch mode.',
        'Motor limits (0..MAX_RPM), altitude PD, roll/pitch PD ARE retained.',
        f'Hover RPM={float(env.HOVER_RPM):.4f}; max RPM={float(env.MAX_RPM):.4f}',
        f'KF={rig.kf:.12g}, KM={rig.km:.12g}, Izz={rig.izz:.12g}',
        f'Theoretical hover yaw torque at zero roll/pitch: ±{run_info["theory_torque_Nm"]:.9g} Nm',
        f'Theoretical initial alpha: ±{run_info["theory_alpha_deg_s2"]:.2f} deg/s2',
        'Max yaw rate in tests is duration-dependent, NOT a unique maximum of the motor model.',
        f'Absolute overshoot threshold: {YAW_BATCH_OVERSHOOT_TOL_DEG:.3f} deg;',
        'PASS requires zero observed crossing (>1e-6deg), settling, and no late drift.',
        'SMALL_OVERSHOOT means crossed goal by >1e-6deg but <= tolerance (NOT PASS).',
        f'Arrival: |angle error| <= {YAW_BATCH_ARRIVAL_ANGLE_DEG}deg and |rate| <= {YAW_BATCH_ARRIVAL_RATE_DEG_S}deg/s',
        f'for >= {YAW_BATCH_SETTLE_HOLD_S}s, followed by {YAW_BATCH_POST_SETTLE_OBSERVE_S}s drift check.',
        'Settle time includes warmup (subtract warmup to compare travel-only duration).',
        'Braking capability p10 is based on discrete finite differences; velocity is body yaw rate.',
        'Use calibration by direction; later trials assume measured deceleration * torque_fraction * brake_model_scale.',
        'Brake scales above 1 are stress tests, NOT safe operational estimates.',
        'Simulation results are not hardware performance and do not guarantee zero overshoot elsewhere.',
        '', 'Calibration minimum p10 decel by sign:',
        *[f'  {s:+d}: {calib_min[s] if calib_min[s] is not None else "UNAVAILABLE"} deg/s2' for s in YAW_BATCH_DIRECTIONS],
        '',
        f'Target PASS={len(passing)} / {len(targets)}',
        f'OVER={sum(s["status"]=="OVERSHOOT" for s in targets)}, SMALL_OVER={sum(s["status"]=="SMALL_OVERSHOOT" for s in targets)}, TIMEOUT={sum(s["status"]=="TIMEOUT" for s in targets)}, ABORT={sum(s["status"]=="ABORT" for s in targets)}, UNSTABLE={sum(s["status"]=="UNSTABLE" for s in targets)}, SKIP={sum(s["status"]=="SKIP" for s in targets)}',
        '', 'Best candidates that passed BOTH + and - target directions for each angle:',
    ]
    for angle,fraction,scale,worst in winners:
        lines.append(f'  target=±{angle:.0f}deg; torque_fraction={fraction:.2f}, brake_model_scale={scale:.2f}; worst_settle={worst:.4f}s')
    if not winners:lines.append('  None passed both directions; do not use as a validated motion setting.')
    lines+=['', 'All target cases ranked by status/settle time:']
    for s in sorted(targets,key=lambda q:(q['status']!='PASS',float(q['settle_time_s']) if q['settle_time_s']!='' else 1e9)):
        over = (float(s['overshoot_deg']) if s['overshoot_deg'] != '' else float('nan'))
        err = (float(s['final_yaw_error_deg']) if s['final_yaw_error_deg'] != '' else float('nan'))
        lines.append(f"  #{s['case_id']} target={float(s['target_yaw_deg']):+g}deg fraction={s['torque_fraction']} scale={s['brake_model_scale']} status={s['status']} overshoot={over:.4f}deg settle={s['settle_time_s']} final_err={err:+.4f}deg")
    txt_path=LOG_DIR/'drone_motion_yaw.txt'
    txt_path.write_text('\n'.join(lines)+'\n',encoding='utf8')
    cfg={
        'model':'CF2X','sim_hz':SIM_FREQ,'physics':'PYB default',
        'total_cases':len(summaries),'calibration_cases':len(calibration),
        'target_cases':len(targets),'hover_rpm':float(env.HOVER_RPM),
        'max_rpm':float(env.MAX_RPM),'kf':rig.kf,'km':rig.km,'Izz':rig.izz,
        'theory_yaw_torque_Nm':run_info['theory_torque_Nm'],
        'theory_initial_acc_deg_s2':run_info['theory_alpha_deg_s2'],
        'calibration':{'fractions':list(YAW_BATCH_PULSE_FRACTIONS),'durations_s':list(YAW_BATCH_PULSE_DURATIONS_S),'directions':list(YAW_BATCH_DIRECTIONS),'p10_by_direction':{str(k):v for k,v in calib_min.items()}},
        'target':{'angles_deg':list(YAW_BATCH_TARGET_ANGLES_DEG),'fractions':list(YAW_BATCH_TARGET_TORQUE_FRACTIONS),'brake_scales':list(YAW_BATCH_BRAKE_MODEL_SCALES),'timeout_s':YAW_BATCH_TARGET_TIMEOUT_S},
        'criteria':{'overshoot_tol_deg':YAW_BATCH_OVERSHOOT_TOL_DEG,'angle_tol_deg':YAW_BATCH_ARRIVAL_ANGLE_DEG,'rate_tol_deg_s':YAW_BATCH_ARRIVAL_RATE_DEG_S,'stable_s':YAW_BATCH_SETTLE_HOLD_S},
        'abort_guards':{'max_rate_deg_s':YAW_BATCH_ABORT_RATE_DEG_S,'max_tilt_deg':YAW_BATCH_ABORT_TILT_DEG,'max_alt_error_m':YAW_BATCH_ABORT_Z_ERROR_M,'max_xy_drift_m':YAW_BATCH_ABORT_XY_DRIFT_M},
        'source_mode':EXPERIMENT_MODE,'log_schema':YAW_BATCH_FIELDS
    }
    config_path=LOG_DIR/'drone_motion_yaw_config.json'
    config_path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    # 간결한 2x2 분석 그래프: 각 방향/크기별 최고 속도, target 시험 성공률/제동 효과.
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    cal_good=[s for s in calibration if s['status']=='PASS']
    for d in YAW_BATCH_DIRECTIONS:
        sub=[s for s in cal_good if s['direction']==d and s['torque_fraction']==1.0]
        if sub:
            axs[0,0].plot([s['pulse_duration_s'] for s in sub],
                          [s['peak_abs_yaw_rate_deg_s'] for s in sub],marker='o',label=f'direction {d:+}')
    axs[0,0].set_title('Calibration peak yaw rate vs full-torque pulse')
    axs[0,0].set_xlabel('Pulse [s]');axs[0,0].set_ylabel('Peak |yaw rate| [deg/s]');axs[0,0].legend()
    angles=list(YAW_BATCH_TARGET_ANGLES_DEG)
    axs[0,1].bar([str(a) for a in angles],[sum(s['status']=='PASS' for s in targets if abs(float(s['target_yaw_deg']))==a) for a in angles])
    axs[0,1].set_title('Target passes by angle (16 trials / angle)')
    axs[0,1].set_xlabel('Target |angle| [deg]');axs[0,1].set_ylabel('Number passed')
    for d in YAW_BATCH_DIRECTIONS:
        sub=[s for s in calibration if s['direction']==d and s['status']=='PASS' and s['torque_fraction']==1.0 and s['brake_decel_p10_deg_s2']!='']
        if sub:
            axs[1,0].plot([s['pulse_duration_s'] for s in sub],
                          [s['brake_decel_p10_deg_s2'] for s in sub],marker='o',label=f'{d:+}')
    axs[1,0].set_title('Observed 10th-percentile braking decel')
    axs[1,0].set_xlabel('Pulse [s]');axs[1,0].set_ylabel('Brake decel [deg/s²]');axs[1,0].legend()
    for frac in YAW_BATCH_TARGET_TORQUE_FRACTIONS:
        sub=[s for s in passing if s['torque_fraction']==frac]
        if sub:
            axs[1,1].scatter([float(s['settle_time_s']) for s in sub],
                             [float(s['overshoot_deg']) for s in sub],label=f'torque={frac}')
    axs[1,1].axhline(YAW_BATCH_OVERSHOOT_TOL_DEG,ls='--',color='black',lw=1)
    axs[1,1].set_title('Passing cases: settling time vs overshoot')
    axs[1,1].set_xlabel('Settle time [s]');axs[1,1].set_ylabel('Overshoot [deg]');axs[1,1].legend()
    for ax in axs.flat:ax.grid(alpha=.25)
    fig.tight_layout()
    plot_path=LOG_DIR/'drone_motion_yaw.png'
    fig.savefig(plot_path,dpi=130)
    plt.close(fig)
    print('\nYaw batch outputs:')
    bundle_path = LOG_DIR/'drone_motion_yaw_bundle.zip'
    from zipfile import ZipFile, ZIP_DEFLATED
    outputs = [LOG_DIR/'drone_motion_yaw.csv',summary_path,txt_path,plot_path,config_path]
    # 업로드할 때는 ZIP 한 파일만 전달하면 전체 원본/평가/설정이 포함된다.
    with ZipFile(bundle_path,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for item in outputs:
            archive.write(item,arcname=item.name)
    for f in [*outputs,bundle_path]:print(' -',f)
    return winners


def run_yaw_batch_experiment():
    """한 번 실행으로 +/− Yaw 물리 응답과 오버슈트/정착 후보 성능 자동 측정."""
    import itertools
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
                   initial_xyzs=np.array([[0.,0.,YAW_BATCH_START_ALT_M]]),
                   initial_rpys=np.zeros((1,3)),
                   pyb_freq=SIM_FREQ,ctrl_freq=SIM_FREQ,gui=YAW_BATCH_GUI,
                   user_debug_gui=False)
    mg=float(env.GRAVITY)
    thrust0=mg/4.0
    fmax=float(env.KF*env.MAX_RPM**2)
    torque_theory=(4.0*env.KM/env.KF)*min(thrust0,fmax-thrust0)
    alpha_theory=float(np.rad2deg(torque_theory/env.J[2,2]))
    calib=[];summaries=[];case_no=0
    output=LOG_DIR/'drone_motion_yaw.csv'
    start_wall=time.perf_counter()
    failure=None
    print('Yaw automatic batch: 32 calibration + 80 target trials = 112 independent resets')
    print('Physical RPM limits retained. Yaw 45/60/35 software limits bypassed.')
    print(f'Theoretical initial torque={torque_theory:.7g} Nm, alpha={alpha_theory:.1f} deg/s2')
    try:
        with output.open('w',newline='',encoding='utf-8-sig') as fp:
            writer=csv.DictWriter(fp,fieldnames=YAW_BATCH_FIELDS,extrasaction='ignore')
            writer.writeheader()
            rig=_YawBatchRig(env,writer)
            for fraction,duration,direction in itertools.product(
                YAW_BATCH_PULSE_FRACTIONS,YAW_BATCH_PULSE_DURATIONS_S,YAW_BATCH_DIRECTIONS):
                case_no+=1
                case={'case_id':case_no,'case_type':'calibration','direction':direction,
                      'torque_fraction':fraction,'pulse_duration_s':duration}
                rig.new_case()
                result=_yaw_batch_calibration(rig,case)
                calib.append(result);summaries.append(result)
                print(f"[{case_no:03d}/112] calib frac={fraction:.2f} pulse={duration:.4f}s dir={direction:+} | maxrate={float(result['peak_abs_yaw_rate_deg_s']):.1f} | decel_p10={result['brake_decel_p10_deg_s2']} | {result['status']}")
            # 물리 한계 토크의 100% 제동 데이터에서 방향별 보수적 p10 확보.
            calib_alpha={}
            for direction in YAW_BATCH_DIRECTIONS:
                valid=[float(x['brake_decel_p10_deg_s2']) for x in calib
                       if x['direction']==direction and x['status']=='PASS'
                       and x['torque_fraction']==1.0 and x['pulse_duration_s']>=0.05
                       and isinstance(x['brake_decel_p10_deg_s2'],float)
                       and x['brake_decel_p10_deg_s2']>0]
                calib_alpha[direction]=min(valid) if valid else None
            print('Calibration p10 minima:',calib_alpha)
            for angle,direction,fraction,scale in itertools.product(
                YAW_BATCH_TARGET_ANGLES_DEG,YAW_BATCH_DIRECTIONS,
                YAW_BATCH_TARGET_TORQUE_FRACTIONS,YAW_BATCH_BRAKE_MODEL_SCALES):
                case_no+=1
                case={'case_id':case_no,'case_type':'target','direction':direction,
                      'torque_fraction':fraction,'target_yaw_deg':direction*angle,
                      'brake_model_scale':scale}
                if calib_alpha[direction] is None:
                    s=_yaw_batch_empty_summary(case)
                    s['status']='SKIP';s['reason']='calibration_missing';s['n_samples']=0
                else:
                    rig.new_case()
                    s,_=_yaw_batch_target(rig,case,calib_alpha[direction])
                summaries.append(s)
                print(f"[{case_no:03d}/112] target={direction*angle:+g}° torque={fraction:.2f} scale={scale:.2f} | {s['status']} | over={s['overshoot_deg']} | settle={s['settle_time_s']}")
            fp.flush()
    except Exception as ex:
        failure=ex
        print('Batch exception: partial raw CSV will be kept:',repr(ex))
    finally:
        # 요약/분석은 환경을 닫기 전에 물리 파라미터를 읽어서 저장한다.
        if 'rig' in locals() and summaries:
            try:
                _yaw_batch_report(summaries,calib,rig,env,
                   {'theory_torque_Nm':float(torque_theory),
                    'theory_alpha_deg_s2':alpha_theory})
            except Exception as ex:
                print('Report generation failed:',repr(ex))
                if failure is None: failure=ex
        env.close()
    print(f'Batch finished; wall time {time.perf_counter()-start_wall:.2f}s; completed {len(summaries)} cases.')
    if failure is not None:raise RuntimeError('일괄 실험 일부 중단. 저장된 요약과 로그 확인 필요.') from failure
    return summaries





# ===================== 9. ROLL / PITCH / YAW 물리 성능 일괄 실험 =====================
# 목표: 각 축 단독, 축쌍, 3축 동시 최대 가속·제동 능력과 목표각 정착 성능을 측정.
# 중요: 이 모드는 안전한 실제 비행 정책이 아니라 물리적 한계 탐색용 시뮬레이션.
#        물리적 모터 RPM 한계는 절대 해제하지 않는다.
#        기존 navigation / yaw_batch 구현은 변경 없이 유지한다.

MOTION_BATCH_PROFILE = 'full'    # full=2052 cases, smoke=빠른 설정 검증
MOTION_BATCH_GUI = False
MOTION_BATCH_REAL_TIME = False
MOTION_BATCH_START_ALT_M = 150.0   # 지면 충돌 대신 안정적 고도 손실 측정
MOTION_BATCH_WARMUP_S = 0.10
MOTION_BATCH_COAST_S = 0.0125
MOTION_BATCH_BRAKE_MAX_S = 0.75
MOTION_BATCH_TARGET_TIMEOUT_S = 2.50
MOTION_BATCH_STABLE_S = 0.20
MOTION_BATCH_OBSERVE_AFTER_S = 0.15
MOTION_BATCH_ANGLE_TOL_DEG = 0.25
MOTION_BATCH_RATE_TOL_DEG_S = 1.0
MOTION_BATCH_OVERSHOOT_EPS_DEG = 0.005
MOTION_BATCH_TERMINAL_ANGLE_DEG = 3.5
MOTION_BATCH_TERMINAL_RATE_DEG_S = 65.0
MOTION_BATCH_BRAKE_RESPONSE_S = 0.0125
MOTION_BATCH_BRAKE_MARGIN_DEG = 0.35
MOTION_BATCH_ABORT_TILT_DEG = 55.0
MOTION_BATCH_ABORT_BODY_RATE_DEG_S = 3500.0
MOTION_BATCH_ABORT_ALTITUDE_LOSS_M = 12.0
MOTION_BATCH_ABORT_XY_DRIFT_M = 25.0
MOTION_BATCH_VERBOSE_EVERY = 25  # 콘솔에 N회당 1번 요약
# NOTE: 목적은 zero-crossing 없는 목표 정착. 가장 빠른 시간 이외에도 제동 여유와 축간 간섭을 분석한다.

_AXES = ('roll','pitch','yaw')
_ROLL_PATTERN = np.array([-1.,-1.,1.,1.])
_PITCH_PATTERN = np.array([-1.,1.,1.,-1.])
_YAW_PATTERN = np.array([-1.,1.,-1.,1.])

MOTION_FIELDS = [
    'case_id','case_type','axes','phase','step','t_s',
    'fraction','pulse_s','brake_scale','calibration_used',
    'target_roll_deg','target_pitch_deg','target_yaw_deg',
    'roll_deg','pitch_deg','yaw_deg','yaw_unwrapped_deg',
    'roll_error_deg','pitch_error_deg','yaw_error_deg',
    'roll_body_rate_deg_s','pitch_body_rate_deg_s','yaw_body_rate_deg_s',
    'roll_body_acc_deg_s2','pitch_body_acc_deg_s2','yaw_body_acc_deg_s2',
    'roll_stop_angle_deg','pitch_stop_angle_deg','yaw_stop_angle_deg',
    'torque_roll_req_Nm','torque_pitch_req_Nm','torque_yaw_req_Nm',
    'torque_roll_act_Nm','torque_pitch_act_Nm','torque_yaw_act_Nm',
    'torque_scale_physical','motor_saturated','motor_1_rpm','motor_2_rpm',
    'motor_3_rpm','motor_4_rpm','z_m','vz_m_s','x_m','y_m','vx_m_s','vy_m_s',
    'total_thrust_req_N','total_thrust_actual_N', 'abort_reason',
]
MOTION_SUMMARY_FIELDS = [
    'case_id','case_type','axes','dir_roll','dir_pitch','dir_yaw','target_roll_deg','target_pitch_deg','target_yaw_deg',
    'fraction','pulse_s','brake_scale','calibration_used','brake_model_quality','status','reason',
    'duration_s','move_time_s','settle_time_s','n_samples',
    'roll_peak_rate_deg_s','pitch_peak_rate_deg_s','yaw_peak_rate_deg_s',
    'roll_accel_median_deg_s2','pitch_accel_median_deg_s2','yaw_accel_median_deg_s2',
    'roll_brake_p10_deg_s2','pitch_brake_p10_deg_s2','yaw_brake_p10_deg_s2',
    'roll_brake_angle_deg','pitch_brake_angle_deg','yaw_brake_angle_deg',
    'roll_overshoot_deg','pitch_overshoot_deg','yaw_overshoot_deg',
    'overshoot_any','max_target_axis_overshoot_deg',
    'final_roll_error_deg','final_pitch_error_deg','final_yaw_error_deg',
    'final_roll_rate_deg_s','final_pitch_rate_deg_s','final_yaw_rate_deg_s',
    'max_roll_deg','max_pitch_deg','max_yaw_rate_deg_s',
    'max_tilt_deg','max_altitude_loss_m','max_altitude_gain_m',
    'max_xy_drift_m','max_motor_rpm','motor_saturation_steps',
    'min_torque_scale_physical','mean_torque_scale_physical',
]


def _motion_quat_error_body(current_rpy_rad, target_rpy_rad):
    """SO(3) 목표-현재 자세 상대 회전의 몸체좌표계 회전벡터[rad].

    R_err=R_current.T @ R_target: 현재 몸체에서 목표 몸체로 가는 최소회전.
    회전벡터를 사용해야 Roll·Pitch·Yaw 동시 시험에서 단순 Euler 오차보다 일관적.
    특이한 180° 상대회전은 보수적으로 abort 대상; 이 실험의 목표 범위는 그보다 작다.
    """
    qc=p.getQuaternionFromEuler(np.asarray(current_rpy_rad).tolist())
    qt=p.getQuaternionFromEuler(np.asarray(target_rpy_rad).tolist())
    a=np.asarray(p.getMatrixFromQuaternion(qc)).reshape(3,3)
    b=np.asarray(p.getMatrixFromQuaternion(qt)).reshape(3,3)
    r=a.T @ b
    cosine=float(np.clip((np.trace(r)-1)/2,-1.,1.))
    theta=float(np.arccos(cosine))
    skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])
    if theta < 1e-5: return 0.5*skew
    if abs(np.sin(theta))<1e-5:  # near pi; strict fallback
        return np.zeros(3)
    return theta/(2.*np.sin(theta))*skew


def _motion_case_library():
    """완전 결정적인 sweep, 개별 case_id로 raw+summary 교차검증 가능."""
    import itertools
    if MOTION_BATCH_PROFILE=='smoke':
        fractions=(0.65,1.0)
        durations=(0.01666667,0.03333333)
        scales=(0.65,0.90)
        levels=(0,1)
    elif MOTION_BATCH_PROFILE=='full':
        fractions=(0.40,0.70,1.0)
        durations=(0.00833333,0.01666667,0.025,0.05)
        scales=(0.50,0.65,0.80,0.95,1.10)
        levels=None
    else:
        raise ValueError('MOTION_BATCH_PROFILE은 full 또는 smoke여야 합니다.')
    configs=[]
    # 1) 3개 단일 축 교정 (+/-) * 토크/시간
    singles=[(ax,) for ax in _AXES]
    doubles=[('roll','pitch'),('roll','yaw'),('pitch','yaw')]
    triples=[_AXES]
    for axes in [*singles,*doubles,*triples]:
        signs=list(itertools.product((-1,1),repeat=len(axes)))
        if len(axes)==1:
            fs=fractions; ts=durations
        else:
            fs=(0.60,1.0) if levels is None else (1.0,)
            ts=(0.01666667,0.025,0.05) if levels is None else (0.01666667,)
        for sign,frac,sec in itertools.product(signs,fs,ts):
            target={a:0.0 for a in _AXES}
            cmd={a:0 for a in _AXES}
            for a,d in zip(axes,sign):cmd[a]=d
            configs.append({'case_type':'calibration','axes':axes,'signs':cmd,
                            'target':target,'fraction':frac,'pulse_s':sec,
                            'brake_scale':'','level':''})
    # 2) 목표각 실험: 단일 축 6 크기, 2축 4 크기, 3축 5 크기
    magnitudes={
        'roll':(5,10,15,25,35,45),
        'pitch':(5,10,15,25,35,45),
        'yaw':(10,25,45,75,110,150),
    }
    for axes in [*singles,*doubles,*triples]:
        sign_combos=list(itertools.product((-1,1),repeat=len(axes)))
        if len(axes)==1:
            count=6
        elif len(axes)==2:
            count=4
        else:
            count=5
        if levels is not None:count=2
        for signs,k,frac,scale in itertools.product(sign_combos,range(count),
                             (0.50,0.75,1.0) if levels is None else (0.75,),scales):
            t={a:0. for a in _AXES}; d={a:0 for a in _AXES}
            for ax,sgn in zip(axes,signs):
                d[ax]=sgn
                if len(axes)==1: mag=magnitudes[ax][k]
                elif len(axes)==2: mag=(6,12,20,30)[k] if ax!='yaw' else (15,35,65,100)[k]
                else: mag=(5,10,15,25,35)[k] if ax!='yaw' else (10,25,45,70,110)[k]
                t[ax]=sgn*mag
            configs.append({'case_type':'target','axes':axes,'signs':d,
                            'target':t,'fraction':frac,'pulse_s':'',
                            'brake_scale':scale,'level':k})
    for i,c in enumerate(configs,1):c['case_id']=i
    return configs


class _MotionRig:
    """CF2X 직접 RPM 입력. 동시축 토크는 모두 동일한 scaling으로 feasible 투영."""
    def __init__(self,env,writer):
        self.env=env;self.writer=writer
        self.dt=1./SIM_FREQ
        self.mass=float(env.M);self.kf=float(env.KF);self.km=float(env.KM)
        self.arm=float(env.L/np.sqrt(2.0));self.j=np.diag(np.asarray(env.J,dtype=float)).copy()
        self.max_f=float(self.kf*env.MAX_RPM**2)
        self.grav=float(env.GRAVITY)/self.mass
        self.hover_f=float(env.GRAVITY)/4.
        self.cap=np.array([
            4.*self.arm*min(self.hover_f,self.max_f-self.hover_f),
            4.*self.arm*min(self.hover_f,self.max_f-self.hover_f),
            4.*self.km/self.kf*min(self.hover_f,self.max_f-self.hover_f),
        ])
        self.total_rows=0
        if np.any(self.cap<=0) or np.any(self.j<=0):
            raise ValueError('모터/관성/호버 파라미터가 유효하지 않습니다.')
        self.new_case()

    def new_case(self):
        obs,_=self.env.reset()
        self.state=np.asarray(obs[0],dtype=float)
        self.step_id=0;self.t=0.;self.case_wall_start=time.perf_counter();self.yaw_unwrapped=float(np.rad2deg(self.state[9]))
        self.prev_yaw_wrapped=float(np.rad2deg(self.state[9]))
        _,_,_,rate=measure_state(self.state)
        self.prev_body_rate=np.rad2deg(rate)

    def step(self,case,phase,mode,alpha=None):
        pos,vel,rpy,omega=measure_state(self.state)
        body_rates=np.rad2deg(omega)
        target_deg=np.array([case['target'][a] for a in _AXES],dtype=float)
        error_body=np.rad2deg(_motion_quat_error_body(rpy,np.deg2rad(target_deg)))
        target_angles=np.rad2deg(rpy).copy();target_angles[2]=self.yaw_unwrapped
        target_error=target_deg-target_angles
        # hover 고도 제어는 기존 검증 PD를 그대로 사용.
        alt=altitude_pd(pos[2],vel[2],MOTION_BATCH_START_ALT_M,
                        self.mass,self.grav,rpy[0],rpy[1])
        total_thrust=float(alt[4]);base=np.full(4,total_thrust/4.)
        chosen=np.zeros(3);planned=np.full(3,np.nan)
        signs=np.array([case['signs'][a] for a in _AXES],dtype=float)
        frac=float(case['fraction'])
        if mode=='pulse':
            chosen=signs*self.cap*frac
        elif mode=='brake':
            # 현재 각속도의 반대 방향으로 토크 인가: 축별 속도가 먼저 정지할 수 있음.
            for j in range(3):
                if signs[j]==0 or abs(body_rates[j])<0.8:continue
                direction=-np.sign(body_rates[j])
                chosen[j]=direction*self.cap[j]*frac
        elif mode=='target':
            count=int(np.count_nonzero(signs))
            scale=float(case['brake_scale'])
            # 다축 동시 명령 시 가용 모터 추력 공유 → 보수적 감속 추정.
            # scale>1은 낙관적 감속 가정의 스트레스 시험이며 추천 값 아님.
            for j in range(3):
                if signs[j]==0:
                    # 시험 대상이 아닌 축도 기존에 검증한 저수준 PD로 0도를 유지.
                    gains_kp=(KP_ROLL,KP_PITCH,KP_YAW)
                    gains_kd=(KD_ROLL,KD_PITCH,KD_YAW)
                    hold_tau=(gains_kp[j]*np.deg2rad(error_body[j])
                              -gains_kd[j]*np.deg2rad(body_rates[j]))
                    chosen[j]=float(np.clip(hold_tau,-self.cap[j],self.cap[j]))
                    continue
                e=float(error_body[j]);w=float(body_rates[j]);towards=np.sign(e)*w
                a0=float(alpha[j]) if alpha is not None and alpha[j]>0 else 80.
                # 동시축 실측 제동 모델에는 이미 모터 출력 공유가 반영됨.
                # 단일축 추정 모델은 축 수로 나눠 보수적으로 계획.
                sharing=max(1.,float(case.get('calibration_sharing_factor',count)))
                decel=max(1.,a0*frac*scale/sharing)
                stop=w*w/(2.*decel)+abs(w)*MOTION_BATCH_BRAKE_RESPONSE_S+MOTION_BATCH_BRAKE_MARGIN_DEG
                planned[j]=stop
                # 목표 근처/각속도 작은 구간은 연속 PD로 전환. SO3 몸체 오차 기반.
                if abs(e)<MOTION_BATCH_TERMINAL_ANGLE_DEG and abs(w)<MOTION_BATCH_TERMINAL_RATE_DEG_S:
                    # 원하는 몸체 각가속도 = Kp*e - Kd*omega [deg/s²]
                    acc=95.*e-12.*w
                    chosen[j]=float(np.clip(self.j[j]*np.deg2rad(acc),-self.cap[j]*frac,self.cap[j]*frac))
                elif e*w>0 and abs(e)<=stop:
                    chosen[j]=-np.sign(w)*self.cap[j]*frac
                elif e*w<0 and abs(w)>1.0:
                    chosen[j]=-np.sign(w)*self.cap[j]*frac
                else:
                    chosen[j]=np.sign(e)*self.cap[j]*frac
        # roll/pitch/yaw 토크 -> 모터 추력 변화량; 한 비율로 동시 토크 축소.
        contrib=(chosen[0]/(4.*self.arm)*_ROLL_PATTERN
                 +chosen[1]/(4.*self.arm)*_PITCH_PATTERN
                 +chosen[2]*self.kf/(4.*self.km)*_YAW_PATTERN)
        feasible=1.
        for b,x in zip(base,contrib):
            if x>1e-15:feasible=min(feasible,(self.max_f-b)/x)
            elif x< -1e-15:feasible=min(feasible, -b/x)
        feasible=float(np.clip(feasible,0.,1.))
        actual_motor_f=np.clip(base+feasible*contrib,0.,self.max_f)
        # base 자체가 물리 한계 바깥인 경우, 고도 요구를 충족하지 못했음을 기록.
        sat=bool(feasible<0.999999 or np.any(base<0) or np.any(base>self.max_f))
        rpm=np.sqrt(actual_motor_f/self.kf)
        rpm=np.clip(rpm,0.,self.env.MAX_RPM)
        f=self.kf*rpm*rpm
        realized=np.array([
            self.arm*np.dot(_ROLL_PATTERN,f),
            self.arm*np.dot(_PITCH_PATTERN,f),
            self.km/self.kf*np.dot(_YAW_PATTERN,f),
        ])
        obs,_,term,trunc,_=self.env.step(rpm.reshape(1,4))
        self.state=np.asarray(obs[0],dtype=float)
        p2,v2,rpy2,w2=measure_state(self.state)
        angle=np.rad2deg(rpy2); wrapped=float(angle[2])
        self.yaw_unwrapped+=_yaw_wrap_delta_deg(wrapped-self.prev_yaw_wrapped)
        self.prev_yaw_wrapped=wrapped
        angle[2]=self.yaw_unwrapped
        wdeg=np.rad2deg(w2)
        acc=(wdeg-self.prev_body_rate)/self.dt
        self.prev_body_rate=wdeg.copy()
        self.t+=self.dt;self.step_id+=1
        row={
          'case_id':case['case_id'],'case_type':case['case_type'],
          'axes':'+'.join(case['axes']),'phase':phase,
          'step':self.step_id,'t_s':self.t,'fraction':frac,
          'pulse_s':case['pulse_s'],'brake_scale':case['brake_scale'],
          'calibration_used':int(alpha is not None),
          'torque_scale_physical':feasible,'motor_saturated':int(sat),
          'total_thrust_req_N':total_thrust,'total_thrust_actual_N':float(np.sum(f)),
          'z_m':p2[2],'vz_m_s':v2[2],
          'x_m':p2[0],'y_m':p2[1],'vx_m_s':v2[0],'vy_m_s':v2[1],
          'yaw_unwrapped_deg':self.yaw_unwrapped,
        }
        for k,a in enumerate(_AXES):
            row[f'target_{a}_deg']=target_deg[k]
            row[f'{a}_deg']=angle[k]
            row[f'{a}_error_deg']=target_deg[k]-angle[k]
            row[f'{a}_body_rate_deg_s']=wdeg[k]
            row[f'{a}_body_acc_deg_s2']=acc[k]
            row[f'{a}_stop_angle_deg']=planned[k]
            row[f'torque_{a}_req_Nm']=chosen[k]
            row[f'torque_{a}_act_Nm']=realized[k]
        row['yaw_unwrapped_deg']=self.yaw_unwrapped
        for k in range(4):row[f'motor_{k+1}_rpm']=rpm[k]
        reason=''
        if not np.all(np.isfinite([*p2,*v2,*rpy2,*wdeg,*acc,*rpm])):reason='nonfinite'
        elif np.max(np.abs(wdeg))>MOTION_BATCH_ABORT_BODY_RATE_DEG_S:reason='angular_rate_guard'
        elif max(abs(angle[0]),abs(angle[1]))>MOTION_BATCH_ABORT_TILT_DEG:reason='tilt_guard'
        elif MOTION_BATCH_START_ALT_M-p2[2]>MOTION_BATCH_ABORT_ALTITUDE_LOSS_M:reason='altitude_loss_guard'
        elif np.linalg.norm(p2[:2])>MOTION_BATCH_ABORT_XY_DRIFT_M:reason='xy_drift_guard'
        elif term or trunc:reason='environment_terminated'
        row['abort_reason']=reason
        self.writer.writerow(row)
        self.total_rows+=1
        if MOTION_BATCH_REAL_TIME:
            remain=self.t-(time.perf_counter()-self.case_wall_start)
            if remain>0:time.sleep(remain)
        return row,reason


def _motion_summarize(case,samples,stop_info=None,settle=None,abort=''):
    out={k:'' for k in MOTION_SUMMARY_FIELDS}
    out.update({'case_id':case['case_id'],'case_type':case['case_type'],
                'axes':'+'.join(case['axes']),'fraction':case['fraction'],
                'pulse_s':case['pulse_s'],'brake_scale':case['brake_scale'],
                'calibration_used':int(case['case_type']=='target')})
    for a in _AXES:
        out[f'target_{a}_deg']=case['target'][a]
        out[f'dir_{a}']=case['signs'][a]
    out['brake_model_quality']=case.get('brake_model_quality', 'not_applicable')
    out['n_samples']=len(samples)
    if not samples:
        out.update(status='SKIP',reason=abort or 'no_samples');return out
    def vals(key):return np.asarray([x[key] for x in samples],dtype=float)
    t0=MOTION_BATCH_WARMUP_S
    out['duration_s']=samples[-1]['t_s']
    out['move_time_s']=float(samples[-1]['t_s'])-t0
    out['max_motor_rpm']=max(float(np.max(vals('motor_'+str(i)+'_rpm'))) for i in range(1,5))
    out['motor_saturation_steps']=int(np.sum(vals('motor_saturated')))
    out['min_torque_scale_physical']=float(np.min(vals('torque_scale_physical')))
    out['mean_torque_scale_physical']=float(np.mean(vals('torque_scale_physical')))
    out['max_roll_deg']=float(np.max(np.abs(vals('roll_deg'))))
    out['max_pitch_deg']=float(np.max(np.abs(vals('pitch_deg'))))
    out['max_yaw_rate_deg_s']=float(np.max(np.abs(vals('yaw_body_rate_deg_s'))))
    out['max_tilt_deg']=float(np.max(np.hypot(vals('roll_deg'),vals('pitch_deg'))))
    out['max_altitude_loss_m']=float(np.max(MOTION_BATCH_START_ALT_M-vals('z_m')))
    out['max_altitude_gain_m']=float(np.max(vals('z_m')-MOTION_BATCH_START_ALT_M))
    out['max_xy_drift_m']=float(np.max(np.hypot(vals('x_m'),vals('y_m'))))
    relevant=[x for x in samples if x['phase']!='warmup']
    over_any=False;max_over=0.
    for a in _AXES:
        out[f'{a}_peak_rate_deg_s']=float(np.max(np.abs(vals(f'{a}_body_rate_deg_s'))))
        out[f'final_{a}_error_deg']=float(samples[-1][f'{a}_error_deg'])
        out[f'final_{a}_rate_deg_s']=float(samples[-1][f'{a}_body_rate_deg_s'])
        direction=case['signs'][a]
        if direction:
            ac=[direction*r[f'{a}_body_acc_deg_s2'] for r in samples if r['phase']=='accel']
            bd=[-direction*r[f'{a}_body_acc_deg_s2'] for r in samples
                if r['phase']=='brake' and direction*r[f'{a}_body_rate_deg_s']>8.]
            bd=[float(v) for v in bd if np.isfinite(v) and v>0]
            out[f'{a}_accel_median_deg_s2']=float(np.median(ac)) if ac else ''
            out[f'{a}_brake_p10_deg_s2']=float(np.percentile(bd,10)) if len(bd)>=2 else ''
            if stop_info and a in stop_info:
                angle0=stop_info[a]
                out[f'{a}_brake_angle_deg']=float(direction*(samples[-1][f'{a}_deg']-angle0))
            if case['case_type']=='target':
                o=max([0.]+[max(0.,direction*(r[f'{a}_deg']-case['target'][a])) for r in relevant])
                out[f'{a}_overshoot_deg']=o
                over_any=over_any or o>MOTION_BATCH_OVERSHOOT_EPS_DEG
                max_over=max(max_over,o)
        elif case['case_type']=='target':
            out[f'{a}_overshoot_deg']=0.
    out['max_target_axis_overshoot_deg']=max_over
    out['overshoot_any']=int(over_any)
    out['settle_time_s']=float(settle-t0) if settle is not None else ''
    if abort:out['status']='ABORT';out['reason']=abort
    elif case['case_type']=='calibration':
        # 실제 모든 활성 축의 각속도가 정지하지 않으면 교정 실패.
        ok=all(abs(samples[-1][f'{a}_body_rate_deg_s'])<=2.0 for a in case['axes'])
        out['status']='PASS' if ok else 'TIMEOUT'
        out['reason']='' if ok else 'brake_not_finished'
    elif settle is None:out['status']='TIMEOUT';out['reason']='not_settled'
    elif over_any:out['status']='OVERSHOOT';out['reason']='any_target_axis_crossed'
    elif any(abs(samples[-1][f'{a}_error_deg'])>MOTION_BATCH_ANGLE_TOL_DEG or
             abs(samples[-1][f'{a}_body_rate_deg_s'])>MOTION_BATCH_RATE_TOL_DEG_S for a in _AXES):
        out['status']='UNSTABLE';out['reason']='late_error_or_rate'
    else:out['status']='PASS';out['reason']=''
    if out['status']=='PASS' and case['case_type']=='target' and out['brake_model_quality']!='measured':
        out['status']='UNVERIFIED';out['reason']='missing_calibration_for_one_or_more_axes'
    return out


def _motion_calibration_case(rig,case):
    samples=[];abort=''
    for _ in range(_yaw_batch_steps(MOTION_BATCH_WARMUP_S)):
        r,abort=rig.step(case,'warmup','idle');samples.append(r)
        if abort:break
    if not abort:
        for _ in range(_yaw_batch_steps(case['pulse_s'])):
            r,abort=rig.step(case,'accel','pulse');samples.append(r)
            if abort:break
    if not abort:
        for _ in range(_yaw_batch_steps(MOTION_BATCH_COAST_S)):
            r,abort=rig.step(case,'coast','idle');samples.append(r)
            if abort:break
    brake_start={}
    if not abort:
        brake_start={a:samples[-1][f'{a}_deg'] for a in case['axes']}
        for _ in range(_yaw_batch_steps(MOTION_BATCH_BRAKE_MAX_S)):
            if all(abs(rig.prev_body_rate[i])<2.0 for i,a in enumerate(_AXES) if a in case['axes']):break
            r,abort=rig.step(case,'brake','brake');samples.append(r)
            if abort:break
    return _motion_summarize(case,samples,brake_start,abort=abort)


def _motion_target_case(rig,case,alpha):
    samples=[];abort='';stable=0.;settled=None;end_observe=None
    for _ in range(_yaw_batch_steps(MOTION_BATCH_WARMUP_S)):
        r,abort=rig.step(case,'warmup','idle',alpha=alpha);samples.append(r)
        if abort:break
    if not abort:
        for _ in range(_yaw_batch_steps(MOTION_BATCH_TARGET_TIMEOUT_S)):
            r,abort=rig.step(case,'control','target',alpha=alpha);samples.append(r)
            if abort:break
            within=all(abs(r[f'{a}_error_deg'])<=MOTION_BATCH_ANGLE_TOL_DEG and
                        abs(r[f'{a}_body_rate_deg_s'])<=MOTION_BATCH_RATE_TOL_DEG_S
                        for a in _AXES)
            if within:stable+=rig.dt
            else:
                stable=0.
                settled=None;end_observe=None
            if settled is None and stable>=MOTION_BATCH_STABLE_S:
                settled=rig.t-MOTION_BATCH_STABLE_S
                end_observe=rig.t+MOTION_BATCH_OBSERVE_AFTER_S
            if end_observe is not None and rig.t>=end_observe:break
    return _motion_summarize(case,samples,settle=settled,abort=abort)


def _motion_read_alpha(calib):
    """단일 축 100% 토크의 반복 교정에서 양방향 감속 하위 백분위 계산.

    상태는 보수적으로 반환하지만, 값이 없을 경우 가상 fallback은
    절대로 검증된 물리적 한계로 사용하지 않도록 provenance 플래그로 명시.
    """
    out={};provenance={}
    for a in _AXES:
        for direction in (-1,1):
            valid=[]
            for r in calib:
                if (r['axes']!=a or r['status']!='PASS'
                    or float(r['fraction'])!=1.0
                    or float(r['pulse_s'])<0.016
                    or int(r[f'dir_{a}'])!=direction):
                    continue
                v=r[f'{a}_brake_p10_deg_s2']
                if v=='' or not np.isfinite(float(v)) or float(v)<=0:continue
                valid.append(float(v))
            key=f'{a}:{direction:+d}'
            if valid:
                out[key]=float(np.percentile(valid,20))
                provenance[key]='measured_single_axis_p20_of_p10'
            else:
                out[key]=80.0
                provenance[key]='FALLBACK_NOT_MEASURED'
    return out,provenance


def _motion_joint_key(case):
    return '+'.join(case['axes']) + '|' + ','.join(f'{a}:{case["signs"][a]:+d}' for a in case['axes'])


def _motion_read_joint_alpha(calibration):
    """실제 Roll+Pitch, Roll+Yaw, Pitch+Yaw, 3축 동시 시험으로 얻은 감속 데이터.

    최대토크(100%)에서 목표 방향 조합에 대해 10백분위 감속값을 계산.
    단일 축 기반의 임의 /N 추정 대신, 가능한 경우 이 실측 데이터를 이용.
    """
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in calibration:
        if (r['status']!='PASS' or float(r['fraction'])!=1.0
            or len(str(r['axes']).split('+'))<=1
            or float(r['pulse_s'])<0.016):
            continue
        axes=tuple(r['axes'].split('+'))
        signs={a:int(r[f'dir_{a}']) for a in _AXES}
        key=_motion_joint_key({'axes':axes,'signs':signs})
        sample={}
        for a in axes:
            v=r[f'{a}_brake_p10_deg_s2']
            if v=='' or not np.isfinite(float(v)) or float(v)<=0:break
            sample[a]=float(v)
        if len(sample)==len(axes):grouped[key].append(sample)
    return {key:{a:float(np.percentile([s[a] for s in vals],20))
                 for a in vals[0]}
            for key,vals in grouped.items()}


def _motion_write_report(summaries,calibration,alpha_map,joint_models,provenance,rig,env,start_time,completed):
    import json
    from collections import Counter,defaultdict
    from zipfile import ZipFile,ZIP_DEFLATED
    from matplotlib import pyplot as plt
    totals=Counter((s['case_type'],s['status']) for s in summaries)
    groups=defaultdict(list)
    for s in summaries:
        if s['case_type']=='target':groups[s['axes']].append(s)
    # 성공 조건 중 목표각 크기/부호를 균형 있게 검증한 후보를 평균 도착시간으로 비교
    winners=[]
    for axes,g in groups.items():
        passing=[r for r in g if r['status']=='PASS']
        if passing:
            fastest=min(passing,key=lambda r:float(r['settle_time_s']))
            winners.append((axes,fastest))
    txt=[
       'CF2X Roll/Pitch/Yaw multi-axis batch experiments',
       f'Expected full profile: 2052 trials. Actual completed: {len(summaries)} / {completed}',
       f'Total 240Hz rows: {rig.total_rows}',
       f'Elapsed wall time: {time.perf_counter()-start_time:.1f} sec',
       f'Starting altitude: {MOTION_BATCH_START_ALT_M:.1f} m; reset between EVERY trial',
       'All tests run with motor 0..MAX_RPM limits enforced; no artificial 45deg/s yaw cap.',
       'All simultaneous torques uniformly scaled into 4-motor physical feasible set.',
       'PASS means no measured target-axis overshoot > 0.005deg, all axes settled, held stable.',
       'Over/undershoot evaluated using Euler R/P/Y; control uses SO(3) body attitude error.',
       'Finite 240Hz sampling does not prove mathematically exact zero overshoot.',
       'Attitude benchmarks omit full XY path tracking; large lateral drift is a measured risk.',
       'Physics tests are not safe hardware operating limits.',
       '', 'Model parameters:',
       f'Ixx,Iyy,Izz = {rig.j.tolist()}, hover RPM={float(env.HOVER_RPM):.1f}, max RPM={float(env.MAX_RPM):.1f}',
       f'Theoretical hover isolated max torques [Nm] RPY={rig.cap.tolist()}',
       f'Measured joint braking models: {len(joint_models)} direction combinations',
       '', 'Measured conservative braking estimates [deg/s2]:',
    ]
    for key,a in sorted(alpha_map.items()):txt.append(f'  {key}: {a:.3f} ({provenance[key]})')
    txt+=['', 'Measured joint braking models [deg/s2]:']
    for key,v in sorted(joint_models.items()):txt.append(f'  {key}: {v}')
    txt+=['', 'Results by status:']
    for k,v in sorted(totals.items()):txt.append(f'  {k[0]} {k[1]}: {v}')
    txt+=['', 'Fastest PASS per active axis combination:']
    for axes,s in sorted(winners):
        txt.append(f'  {axes}: case={s["case_id"]} targets=({s["target_roll_deg"]},{s["target_pitch_deg"]},{s["target_yaw_deg"]}) ' +
                   f'fraction={s["fraction"]}, brake_scale={s["brake_scale"]}, move-to-settle={s["settle_time_s"]} sec')
    txt+=['', 'NOT VALIDATED for deployment: some test cases may abort, timeout, or overshoot.',
          'For fair comparisons, compare matching targets and BOTH directions, NOT only fastest success.',
          'Look for long enough post-settle observation before claiming repeatability.']
    txt_path=LOG_DIR/'drone_motion_axes.txt'
    txt_path.write_text('\n'.join(txt)+'\n',encoding='utf-8')
    cfg={
       'experiment':'all_axes_batch','profile':MOTION_BATCH_PROFILE,
       'sim_hz':SIM_FREQ,'start_altitude_m':MOTION_BATCH_START_ALT_M,
       'calibration':calibration,
       'single_axis_alpha_deg_s2':alpha_map,'joint_alpha_deg_s2':joint_models,
       'alpha_provenance':provenance,
       'theory_torque_caps_Nm':rig.cap.tolist(),
       'inertia_kg_m2':rig.j.tolist(),
       'physical_max_rpm':float(env.MAX_RPM),
       'criteria':{'angle_tol_deg':MOTION_BATCH_ANGLE_TOL_DEG,
                  'rate_tol_deg_s':MOTION_BATCH_RATE_TOL_DEG_S,
                  'overshoot_epsilon_deg':MOTION_BATCH_OVERSHOOT_EPS_DEG,
                  'stable_s':MOTION_BATCH_STABLE_S,'observe_s':MOTION_BATCH_OBSERVE_AFTER_S},
       'abort_guards':{'tilt_deg':MOTION_BATCH_ABORT_TILT_DEG,
                       'body_rate_deg_s':MOTION_BATCH_ABORT_BODY_RATE_DEG_S,
                       'altitude_loss_m':MOTION_BATCH_ABORT_ALTITUDE_LOSS_M,
                       'xy_drift_m':MOTION_BATCH_ABORT_XY_DRIFT_M},
       'raw_columns':MOTION_FIELDS,'summary_columns':MOTION_SUMMARY_FIELDS,
    }
    config_path=LOG_DIR/'drone_motion_axes_config.json'
    config_path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    targets=[r for r in summaries if r['case_type']=='target']
    labels=['roll','pitch','yaw','roll+pitch','roll+yaw','pitch+yaw','roll+pitch+yaw']
    pass_pct=[];med_time=[];max_over=[]
    for label in labels:
        g=[x for x in targets if x['axes']==label]
        ok=[x for x in g if x['status']=='PASS']
        pass_pct.append(100.*len(ok)/len(g) if g else 0)
        med_time.append(float(np.median([float(x['settle_time_s']) for x in ok])) if ok else 0)
        overs=[float(x['max_target_axis_overshoot_deg']) for x in g if x['max_target_axis_overshoot_deg']!='']
        max_over.append(max(overs,default=0.0))
    fig,axes=plt.subplots(3,1,figsize=(12,12))
    axes[0].bar(labels,pass_pct);axes[0].set_ylabel('PASS rate [%]');axes[0].set_ylim(0,100)
    axes[0].set_title('Overshoot-free settling success by axis combination')
    axes[1].bar(labels,med_time);axes[1].set_ylabel('Median successful settle time [s]')
    axes[1].set_title('Settling time of successful cases only')
    axes[2].bar(labels,max_over);axes[2].set_ylabel('Max crossing beyond target [deg]')
    axes[2].set_title('Maximum overshoot across cases')
    for ax in axes:ax.grid(axis='y',alpha=.25);ax.tick_params(axis='x',labelrotation=20)
    fig.tight_layout()
    fig_path=LOG_DIR/'drone_motion_axes.png'
    fig.savefig(fig_path,dpi=125);plt.close(fig)
    paths=[LOG_DIR/'drone_motion_axes.csv',LOG_DIR/'drone_motion_axes_trials.csv',txt_path,config_path,fig_path]
    zip_path=LOG_DIR/'drone_motion_axes_bundle.zip'
    with ZipFile(zip_path,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as zf:
        for f in paths:
            if f.exists():zf.write(f,arcname=f.name)
    print('Output: '+str(zip_path))


def run_all_axes_batch():
    """기본 full 2052건. 구간별 CSV 스트리밍, 자동 평가, 같은 파일명으로 덮어쓰기."""
    configs=_motion_case_library()
    from collections import Counter
    counts=Counter(c['case_type'] for c in configs)
    print(f'Full multiaxis batch: {len(configs)} trials, {counts["calibration"]} calibrations + {counts["target"]} targets')
    print(f'Initial altitude={MOTION_BATCH_START_ALT_M}m, GUI={MOTION_BATCH_GUI}, physics 240Hz, per-case reset')
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
                   initial_xyzs=np.array([[0.,0.,MOTION_BATCH_START_ALT_M]]),
                   initial_rpys=np.zeros((1,3)),
                   pyb_freq=SIM_FREQ,ctrl_freq=SIM_FREQ,
                   gui=MOTION_BATCH_GUI,user_debug_gui=False)
    start=time.perf_counter()
    summaries=[];calib=[];alpha_map={};joint_models={};provenance={};error=None
    raw_path=LOG_DIR/'drone_motion_axes.csv';summary_path=LOG_DIR/'drone_motion_axes_trials.csv'
    try:
        with raw_path.open('w',newline='',encoding='utf-8-sig') as fp, summary_path.open('w',newline='',encoding='utf-8-sig') as sp:
            rw=csv.DictWriter(fp,fieldnames=MOTION_FIELDS,extrasaction='ignore')
            sw=csv.DictWriter(sp,fieldnames=MOTION_SUMMARY_FIELDS,extrasaction='ignore')
            rw.writeheader();sw.writeheader()
            rig=_MotionRig(env,rw)
            for i,case in enumerate(configs,1):
                if case['case_type']=='target' and not alpha_map:
                    alpha_map,provenance=_motion_read_alpha(calib)
                    joint_models=_motion_read_joint_alpha(calib)
                    print('Calibrated conservative p10 deceleration:',alpha_map)
                    print('Measured multi-axis braking models:',len(joint_models))
                    print('Quality flags:',provenance)
                if case['case_type']=='calibration':
                    rig.new_case()
                    result=_motion_calibration_case(rig,case)
                    calib.append(result)
                else:
                    # 교정 데이터 없는 방향도 stress test는 실행하되,
                    # 해당 결과는 미검증 제동모델이라는 플래그로 반드시 남긴다.
                    key=_motion_joint_key(case)
                    measured_single=all(
                        provenance.get(f'{a}:{case["signs"][a]:+d}')=='measured_single_axis_p20_of_p10'
                        for a in case['axes'])
                    if key in joint_models:
                        case['brake_model_quality']='measured'
                        case['calibration_sharing_factor']=1.0
                        alpha=np.array([joint_models[key].get(a,80.) for a in _AXES],dtype=float)
                    else:
                        case['brake_model_quality']=('measured' if measured_single and len(case['axes'])==1
                            else 'estimated_from_single' if measured_single else 'fallback_unverified')
                        case['calibration_sharing_factor']=float(len(case['axes']))
                        alpha=np.array([alpha_map.get(f'{a}:{case["signs"][a]:+d}',80.)
                                        if case['signs'][a] else 80. for a in _AXES],dtype=float)
                    rig.new_case()
                    result=_motion_target_case(rig,case,alpha)
                summaries.append(result);sw.writerow(result)
                if i%MOTION_BATCH_VERBOSE_EVERY==0 or i==len(configs):
                    fp.flush();sp.flush()
                    last=result
                    print(f'[{i}/{len(configs)}] axes={last["axes"]} type={last["case_type"]} status={last["status"]} '+
                          f'overshoot={last["max_target_axis_overshoot_deg"]} settle={last["settle_time_s"]} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    except Exception as ex:
        error=ex
        print('ERROR: partial trial/raw CSV saved; report will contain completed cases:',repr(ex))
    finally:
        if 'rig' in locals():
            if not alpha_map:alpha_map,provenance=_motion_read_alpha(calib)
            if not joint_models:joint_models=_motion_read_joint_alpha(calib)
            try:_motion_write_report(summaries,calib,alpha_map,joint_models,provenance,rig,env,start,len(configs))
            except Exception as ex:
                print('Report error:',repr(ex))
                if error is None:error=ex
        env.close()
    if error is not None:raise RuntimeError('Batch incomplete. Inspect logs/drone_motion_axes*.') from error
    print(f'Completed {len(summaries)} tests, {time.perf_counter()-start:.1f}s walltime')
    return summaries


# 실행 엔트리포인트는 최하단에 둔다 (progressive 튜너 정의 후 호출).

# ============================================================================
# 7. 순차적 성능 상향 시험: 검증한 기존 저수준 제어를 그대로 사용
# ============================================================================
# 1) 기존 기준선의 시나리오별 정착/무오버슈트를 먼저 확인한다.
# 2) MAX_* / 접근·제동 제한을 반드시 '한 번에 하나씩' 증가시킨다.
# 3) 매 제안값마다 같은 초기상태/목표 조건을 240Hz로 새로 실행한다.
# 4) 목표 통과/정착 실패/모터 포화/안정성 저하가 나오면 해당 증가분은 거부한다.
# 5) 각 단계의 마지막 안전 통과 상한은 다음 단계에 반영한다.
#    즉시 성능 향상이 없더라도 연동 제한 해제에 필요할 수 있기 때문이다.
#    가장 빠른 개별 후보도 별도 저장하며, 마지막에 전체 조합을 재시험한다.
#    시험하지 않은 조건에서 0 오버슈트를 보증하지 않는다.

TUNE_GUI = False
TUNE_REAL_TIME = False
TUNE_START_ALT_M = 150.0
TUNE_MAX_TRIAL_S = 19.0
TUNE_OBSERVE_AFTER_SETTLE_S = 0.55
TUNE_STABLE_REQUIRED_S = 0.45
TUNE_EXTRA_RETESTS = 1  # full에서 채택 후보를 서로 다른 조건으로 재검증
TUNE_XY_POSITION_TOL_M = 0.10
TUNE_XY_SPEED_TOL_M_S = 0.10
TUNE_YAW_ANGLE_TOL_DEG = 0.25
TUNE_YAW_RATE_TOL_DEG_S = 0.50
TUNE_Z_POSITION_TOL_M = 0.10
TUNE_Z_SPEED_TOL_M_S = 0.10
# 무오버슈트: 실제 목표평면 초과량과 목표각 초과량을 각각 판정한다.
# 이 값은 샘플링/수치오차에 대한 허용값이지 '오버슈트가 0이었다'는 뜻이 아님.
TUNE_XY_OVERSHOOT_EPS_M = 0.001
TUNE_YAW_OVERSHOOT_EPS_DEG = 0.025
TUNE_Z_OVERSHOOT_EPS_M = 0.005
TUNE_ALTITUDE_DRIFT_GUARD_M = 2.0
TUNE_MAX_TILT_ABORT_DEG = 35.0
TUNE_MAX_BODY_RATE_ABORT_DEG_S = 900.0
TUNE_MIN_Z_M = 100.0  # 150m 시작, 낙하/바닥 접근 조기 중단
TUNE_EXTRA_REFINEMENTS = 2  # 실패 경계에서 증가 간격을 절반씩 줄여 재시험
TUNE_TIME_REGRESSION_BUDGET_S = 0.05  # 50ms 이상 느린 큰 상한은 채택하지 않음
TUNE_PROGRESS_EVERY = 1

# 후보 상한은 '검증된 기체 성능'이 아니라 탐색용 경계다.
# 통과 후보가 경계까지 도달하면 탐색 상한을 높여 다음 실험 가능.
_TUNE_STAGE_PLAN = [
    ('MAX_XY_ACCEL',           'xy',   0.15,   3.0),
    ('APPROACH_ACCEL_M_S2',    'xy',   0.10,   2.7),
    ('BRAKE_ACCEL_M_S2',       'xy',   0.10,   2.6),
    ('MAX_APPROACH_SPEED_M_S', 'xy',   0.25,   7.0),
    ('MAX_XY_TILT_DEG',        'xy',   1.50,  25.0),
    ('MAX_YAW_RATE_DEG_S',     'yaw', 15.0,  255.0),
    ('MAX_YAW_ACCEL_DEG_S2',   'yaw', 30.0,  540.0),
    ('YAW_BRAKE_ACCEL_DEG_S2', 'yaw', 20.0,  395.0),
    ('YAW_TORQUE_RESERVE',     'yaw',  0.05,   1.0),
    ('MAX_VERTICAL_ACCEL',     'alt',  0.50,   9.0),
]
_TUNE_BASE_NAMES = [p[0] for p in _TUNE_STAGE_PLAN]


def _tune_scenarios(profile):
    """각 축을 별도로 자극하고, 마지막에는 Roll+Pitch+Yaw+Z 동시 검증.

    시작점은 (0,0,150), 초기 RPY=0°와 초기 속도=0으로 모두 동일.
    XY 변경으로 Roll/Pitch를 시험하고, Yaw는 회전 목표를 별도로 준다.
    """
    xy = [
        dict(name='xy_ne',goal=(10.,10.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_se',goal=(10.,-10.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_nw',goal=(-10.,10.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_sw',goal=(-10.,-10.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_x', goal=(10.,0.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_y', goal=(0.,10.),yaw=0.,dz=0.,kind='xy'),
        dict(name='xy_long',goal=(18.,-6.),yaw=0.,dz=0.,kind='xy'),
    ]
    yaw = [
        dict(name='yaw_p45',goal=(0.,0.),yaw=45.,dz=0.,kind='yaw'),
        dict(name='yaw_m45',goal=(0.,0.),yaw=-45.,dz=0.,kind='yaw'),
        dict(name='yaw_p90',goal=(0.,0.),yaw=90.,dz=0.,kind='yaw'),
        dict(name='yaw_m90',goal=(0.,0.),yaw=-90.,dz=0.,kind='yaw'),
        dict(name='yaw_p135',goal=(0.,0.),yaw=135.,dz=0.,kind='yaw'),
        dict(name='yaw_m135',goal=(0.,0.),yaw=-135.,dz=0.,kind='yaw'),
    ]
    alt = [
        dict(name='alt_up',goal=(0.,0.),yaw=0.,dz=6.,kind='alt'),
        dict(name='alt_down',goal=(0.,0.),yaw=0.,dz=-6.,kind='alt'),
        dict(name='alt_up_long',goal=(0.,0.),yaw=0.,dz=12.,kind='alt'),
        dict(name='alt_down_long',goal=(0.,0.),yaw=0.,dz=-12.,kind='alt'),
    ]
    combined = [
        dict(name='joint_ne_p45',goal=(10.,10.),yaw=45.,dz=0.,kind='combined'),
        dict(name='joint_sw_m45',goal=(-10.,-10.),yaw=-45.,dz=0.,kind='combined'),
        dict(name='joint_nw_p90',goal=(-10.,10.),yaw=90.,dz=0.,kind='combined'),
        dict(name='joint_se_m90',goal=(10.,-10.),yaw=-90.,dz=0.,kind='combined'),
        dict(name='joint_zup',goal=(10.,10.),yaw=45.,dz=5.,kind='combined'),
        dict(name='joint_zdown',goal=(-10.,10.),yaw=-45.,dz=-5.,kind='combined'),
    ]
    if profile == 'smoke':
        return {'xy':xy[:2], 'yaw':yaw[:2], 'alt':alt[:2], 'combined':combined[:2]},2
    if profile == 'standard':
        return {'xy':xy[:5], 'yaw':yaw[:4], 'alt':alt[:2], 'combined':combined[:4]},7
    if profile == 'full':
        return {'xy':xy,'yaw':yaw,'alt':alt,'combined':combined},100
    raise ValueError('TUNE_PROFILE은 smoke, standard, full 중 하나여야 합니다.')


_TUNE_RAW_COLUMNS = [
    'trial_id','stage','parameter','candidate_value','scenario','kind','repeat',
    'time_s','nav_state','x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s',
    'roll_deg','pitch_deg','yaw_unwrapped_deg','roll_rate_deg_s',
    'pitch_rate_deg_s','yaw_rate_deg_s','target_x_m','target_y_m',
    'target_z_m','target_yaw_deg','xy_error_m','xy_speed_m_s','closing_speed_m_s',
    'yaw_error_deg','goal_projection_overshoot_m','yaw_overshoot_deg',
    'alt_overshoot_m','roll_ref_deg','pitch_ref_deg',
    'xy_ax_cmd_m_s2','xy_ay_cmd_m_s2','radial_acc_m_s2',
    'brake_guard','motor_saturated','motor_1_rpm','motor_2_rpm',
    'motor_3_rpm','motor_4_rpm',
    # 추가 진단: 제동 가드의 작동 원인과 최종 토크/속도, 고도 명령
    'yaw_control_mode','yaw_acc_cmd_deg_s2','yaw_brake_plan_deg_s2',
    'yaw_target_rate_deg_s','yaw_torque_Nm','yaw_pos_torque_limit_Nm',
    'yaw_neg_torque_limit_Nm','yaw_predicted_stop_deg',
    'alt_acc_cmd_m_s2','alt_total_thrust_N','alt_tilt_cos',
    'alt_lookahead_s','z_predicted_m','yaw_stop_lead_s',
]
_TUNE_TRIAL_COLUMNS = [
    'trial_id','stage','parameter','candidate_value','scenario','kind','repeat',
    'status','failure_reason','settle_time_s','duration_s','n_steps',
    'max_goal_overshoot_m','max_yaw_overshoot_deg','max_alt_overshoot_m',
    'max_lateral_deviation_m','max_roll_abs_deg','max_pitch_abs_deg',
    'max_yaw_rate_deg_s','max_xy_speed_m_s','max_z_error_m',
    'max_rpm','motor_saturation_steps','final_xy_error_m','final_xy_speed_m_s',
    'final_yaw_error_deg','final_yaw_rate_deg_s','final_z_error_m','final_z_speed_m_s',
    'xy_brake_guard_steps',
]


def _tune_finite(x):
    return float(x) if np.isfinite(x) else None


def _tune_case(env, case, raw_writer, trial_id, stage, parameter, value, rep, yaw_mode="original"):
    """매 회 env.reset 및 hover RPM 시작. yaw_mode=original/legacy/compensated."""
    obs, _ = env.reset()
    state = np.asarray(obs[0],dtype=float)
    init_xyz = np.array([0.,0.,TUNE_START_ALT_M])
    goal=np.asarray(case['goal'],dtype=float)
    target_z=TUNE_START_ALT_M+float(case['dz'])
    target_yaw_deg=float(case['yaw'])
    delta=goal.copy();length=float(np.linalg.norm(delta));unit=delta/length if length>1e-9 else np.zeros(2)
    sign_z=np.sign(target_z-TUNE_START_ALT_M)
    sign_yaw=np.sign(target_yaw_deg)
    dt=1./SIM_FREQ
    hover_action=np.full((1,4),np.sqrt(float(env.GRAVITY)/(4.*float(env.KF))))
    action=hover_action.copy();prev_sat=False
    max_motor_thrust=float(env.KF*env.MAX_RPM**2)
    arm=float(env.L/np.sqrt(2.));mass=float(env.M)
    gravity=float(env.GRAVITY)/mass
    yaw_inertia=float(env.J[2,2])
    nav= NAV_HOVER
    stable_elapsed=0.
    prev_yaw_deg=float(np.rad2deg(state[9]));yaw_unwrapped=prev_yaw_deg
    prev_yaw_rate_deg_s=0.0
    max_overshoot=0.;max_yaw_overshoot=0.;max_z_overshoot=0.;max_lateral=0.
    max_roll=0.;max_pitch=0.;max_yaw_rate=0.;max_speed=0.;max_zerr=0.;max_rpm=0.
    motor_sat_count=0;brake_count=0;steps=0;settle_t=float('nan')
    first_stable_t=None;settled_at=None;abort_reason=''
    final_xy_err=length;final_xy_speed=0.;final_yaw_err=target_yaw_deg
    final_yaw_rate=0.;final_z_err=target_z-TUNE_START_ALT_M;final_z_speed=0.
    max_step=int(np.ceil(TUNE_MAX_TRIAL_S*SIM_FREQ))
    for step in range(max_step):
        obs,_,terminated,truncated,_=env.step(action)
        state=np.asarray(obs[0],dtype=float)
        t=(step+1)*dt
        pos,vel,rpy,body_rates=measure_state(state)
        rpy_deg=np.rad2deg(rpy);rates_deg=np.rad2deg(body_rates)
        dyaw=_yaw_wrap_delta_deg(rpy_deg[2]-prev_yaw_deg)
        yaw_unwrapped+=dyaw;prev_yaw_deg=float(rpy_deg[2])
        # Yaw와 Alt의 스텝 명령은 출발 전 호버링 1초를 마친 뒤 인가한다.
        ref_z=target_z if t>=START_HOVER_S else TUNE_START_ALT_M
        ref_yaw=target_yaw_deg if t>=START_HOVER_S else 0.
        goal_dist=float(np.linalg.norm(goal-pos[:2]));xy_speed=float(np.linalg.norm(vel[:2]))
        nav,stable_elapsed=update_navigation_state(nav,t,goal_dist,xy_speed,stable_elapsed,dt)
        # 전역 참조 목표점/제어 한계는 trial 스코프에서 할당하고 다음에 다시 복구.
        alt=altitude_pd(pos[2],vel[2],ref_z,mass,gravity,rpy[0],rpy[1])
        xy_acc,nav_diag=point_to_point_controller(pos[:2],vel[:2],nav,SPEED_SCALE)
        roll_ref,pitch_ref,xy_acc=xy_acceleration_to_attitude(xy_acc,rpy[2],alt[3],gravity)
        roll=roll_pd(rpy[0],body_rates[0],roll_ref)
        pitch=pitch_pd(rpy[1],body_rates[1],pitch_ref)
        yp,yn=yaw_torque_limits(alt[4],roll[3],pitch[3],arm,env.KF,env.KM,max_motor_thrust)
        euler_rate, coupling_rate, yaw_r_gain = yaw_kinematic_terms(rpy,body_rates)
        yaw_diag = dict(euler_yaw_rate_rad_s=euler_rate,
                        yaw_coupling_rad_s=coupling_rate,
                        body_yaw_rate_rad_s=float(body_rates[2]),
                        target_body_yaw_rate_rad_s=float('nan'))
        if yaw_mode == "compensated":
            yaw,yaw_diag=yaw_predictive_control_coupled(
                rpy[2],rpy,body_rates,np.deg2rad(ref_yaw),yaw_inertia,yp,yn)
        else:
            yaw=yaw_predictive_control(
                rpy[2],body_rates[2],np.deg2rad(ref_yaw),yaw_inertia,yp,yn)
            yaw_diag['target_body_yaw_rate_rad_s']=yaw[4]
        # For AB fairness, BOTH variants use true Euler yaw rate to compute
        # settling and predicted stop. Original modes retain their old metric.
        measured_yaw_rate=(euler_rate if yaw_mode in ('legacy','compensated')
                           else float(body_rates[2]))
        clos_rate=max(0.,np.sign(yaw[0])*measured_yaw_rate)
        yaw_stop_est=np.rad2deg(clos_rate**2/(2.*max(yaw[6],1e-8))
                            + clos_rate*YAW_STOP_LEAD_S)
        yaw_stop_est+=YAW_STOP_MARGIN_DEG
        new_action,sat_new=allocate_motor_rpm(roll[3],pitch[3],yaw[3],alt[4],arm,env.KF,env.KM,max_motor_thrust)
        # step()에 들어간 이전 명령 기준으로 포화 기록; 새 명령은 다음 스텝에 사용.
        applied_action=action.copy();applied_sat=prev_sat
        if applied_sat:motor_sat_count+=1
        max_rpm=max(max_rpm,float(np.max(action)))
        action=new_action;prev_sat=sat_new
        overshoot=float(max(0.,np.dot(pos[:2],unit)-length)) if length>1e-9 else 0.
        lateral=float(abs(unit[0]*pos[1]-unit[1]*pos[0])) if length>1e-9 else 0.
        yaw_overshoot=max(0.,sign_yaw*yaw_unwrapped-abs(target_yaw_deg)) if sign_yaw else 0.
        z_overshoot=max(0.,sign_z*(pos[2]-TUNE_START_ALT_M)-abs(float(case['dz']))) if sign_z else 0.
        max_overshoot=max(max_overshoot,overshoot)
        max_yaw_overshoot=max(max_yaw_overshoot,yaw_overshoot)
        max_z_overshoot=max(max_z_overshoot,z_overshoot)
        max_lateral=max(max_lateral,lateral)
        max_roll=max(max_roll,abs(float(rpy_deg[0])));max_pitch=max(max_pitch,abs(float(rpy_deg[1])))
        max_yaw_rate=max(max_yaw_rate,abs(float(rates_deg[2])));max_speed=max(max_speed,xy_speed)
        max_zerr=max(max_zerr,abs(float(pos[2]-ref_z)))
        if nav==NAV_APPROACH and nav_diag['emergency_brake']:brake_count+=1
        final_xy_err=goal_dist;final_xy_speed=xy_speed
        final_yaw_err=abs(float(_yaw_wrap_delta_deg(target_yaw_deg-yaw_unwrapped)))
        final_yaw_rate=abs(float(np.rad2deg(measured_yaw_rate)))
        final_z_err=abs(float(target_z-pos[2]));final_z_speed=abs(float(vel[2]))
        row=dict(
            trial_id=trial_id,stage=stage,parameter=parameter,candidate_value=value,
            scenario=case['name'],kind=case['kind'],repeat=rep,time_s=t,nav_state=nav,
            x_m=pos[0],y_m=pos[1],z_m=pos[2],vx_m_s=vel[0],vy_m_s=vel[1],vz_m_s=vel[2],
            roll_deg=rpy_deg[0],pitch_deg=rpy_deg[1],yaw_unwrapped_deg=yaw_unwrapped,
            roll_rate_deg_s=rates_deg[0],pitch_rate_deg_s=rates_deg[1],yaw_rate_deg_s=rates_deg[2],
            target_x_m=goal[0],target_y_m=goal[1],target_z_m=target_z,target_yaw_deg=ref_yaw,
            xy_error_m=goal_dist,xy_speed_m_s=xy_speed,closing_speed_m_s=nav_diag['closing_speed'],
            yaw_error_deg=abs(_yaw_wrap_delta_deg(ref_yaw-yaw_unwrapped)),
            goal_projection_overshoot_m=overshoot,yaw_overshoot_deg=yaw_overshoot,
            alt_overshoot_m=z_overshoot,roll_ref_deg=np.rad2deg(roll_ref),
            pitch_ref_deg=np.rad2deg(pitch_ref),xy_ax_cmd_m_s2=xy_acc[0],xy_ay_cmd_m_s2=xy_acc[1],
            radial_acc_m_s2=nav_diag['radial_acc'],brake_guard=int(nav_diag['emergency_brake'] and nav==NAV_APPROACH),
            motor_saturated=int(applied_sat),motor_1_rpm=applied_action[0,0],motor_2_rpm=applied_action[0,1],
            motor_3_rpm=applied_action[0,2],motor_4_rpm=applied_action[0,3],
            yaw_control_mode=yaw[7],yaw_acc_cmd_deg_s2=np.rad2deg(yaw[5]),
            yaw_brake_plan_deg_s2=np.rad2deg(yaw[6]),
            yaw_target_rate_deg_s=np.rad2deg(yaw[4]),yaw_torque_Nm=yaw[3],
            yaw_pos_torque_limit_Nm=yaw[8],yaw_neg_torque_limit_Nm=yaw[9],
            yaw_predicted_stop_deg=yaw_stop_est,
            alt_acc_cmd_m_s2=alt[3],alt_total_thrust_N=alt[4],alt_tilt_cos=alt[5],
            alt_lookahead_s=ALT_LOOKAHEAD_S,z_predicted_m=pos[2]+vel[2]*ALT_LOOKAHEAD_S,
            yaw_stop_lead_s=YAW_STOP_LEAD_S,
        )
        if 'yaw_variant' in raw_writer.fieldnames:
            # Diagnostic commands are calculated at this sample time and
            # are applied on the following env.step(); RPM fields above are
            # the ALREADY-APPLIED motor commands for this sample.
            row.update(yaw_variant=yaw_mode,
                       yaw_euler_rate_deg_s=np.rad2deg(euler_rate),
                       yaw_body_r_deg_s=np.rad2deg(body_rates[2]),
                       yaw_coupling_rate_deg_s=np.rad2deg(coupling_rate),
                       yaw_r_ref_next_deg_s=np.rad2deg(yaw_diag['target_body_yaw_rate_rad_s']),
                       yaw_psi_ref_next_deg_s=np.rad2deg(yaw[4]),
                       yaw_torque_next_Nm=yaw[3])
        raw_writer.writerow(row)
        steps+=1
        if not np.all(np.isfinite(np.r_[pos,vel,rpy,body_rates,action.flatten()])):
            abort_reason='nonfinite_state'
        elif pos[2]<TUNE_MIN_Z_M:
            abort_reason='altitude_floor'
        elif abs(pos[2]-ref_z)>max(TUNE_ALTITUDE_DRIFT_GUARD_M,abs(float(case['dz']))+TUNE_ALTITUDE_DRIFT_GUARD_M):
            abort_reason='altitude_deviation_guard'
        elif max(max_roll,max_pitch)>TUNE_MAX_TILT_ABORT_DEG:
            abort_reason='tilt_guard'
        elif float(np.max(np.abs(rates_deg)))>TUNE_MAX_BODY_RATE_ABORT_DEG_S:
            abort_reason='rate_guard'
        if terminated or truncated:abort_reason='env_terminated'
        if abort_reason:break
        # 네 종류 조건을 동시에 만족해야 정착; 이후 감시 구간까지 유지해야 PASS.
        stable = (t>=START_HOVER_S and goal_dist<=TUNE_XY_POSITION_TOL_M
                  and xy_speed<=TUNE_XY_SPEED_TOL_M_S
                  and abs(_yaw_wrap_delta_deg(ref_yaw-yaw_unwrapped))<=TUNE_YAW_ANGLE_TOL_DEG
                  and abs(float(np.rad2deg(measured_yaw_rate)))<=TUNE_YAW_RATE_TOL_DEG_S
                  and abs(pos[2]-ref_z)<=TUNE_Z_POSITION_TOL_M
                  and abs(float(vel[2]))<=TUNE_Z_SPEED_TOL_M_S)
        if stable:
            if first_stable_t is None:first_stable_t=t
            if settled_at is None and t-first_stable_t>=TUNE_STABLE_REQUIRED_S:
                settled_at=t;settle_t=t
            if settled_at is not None and t-settled_at>=TUNE_OBSERVE_AFTER_SETTLE_S:
                break
        else:
            first_stable_t=None
            if settled_at is not None:
                settled_at=None
                settle_t=float('nan')
    # 서로 다른 failure mode는 자동으로 구분하고, 오버슈트 존재 여부는 독립 지표로 남긴다.
    if abort_reason:status='ABORT';reason=abort_reason
    elif max_overshoot>TUNE_XY_OVERSHOOT_EPS_M:
        status='OVERSHOOT';reason='xy_target_plane_crossed'
    elif max_yaw_overshoot>TUNE_YAW_OVERSHOOT_EPS_DEG:
        status='OVERSHOOT';reason='yaw_target_crossed'
    elif max_z_overshoot>TUNE_Z_OVERSHOOT_EPS_M:
        status='OVERSHOOT';reason='altitude_target_crossed'
    elif motor_sat_count>0:
        status='SATURATED';reason='motor_rpm_clipped'
    elif settled_at is None:
        status='TIMEOUT';reason='not_settled_and_observed'
    else:status='PASS';reason=''
    return dict(
        trial_id=trial_id,stage=stage,parameter=parameter,candidate_value=value,
        scenario=case['name'],kind=case['kind'],repeat=rep,status=status,failure_reason=reason,
        settle_time_s=_tune_finite(settle_t),duration_s=t,n_steps=steps,
        max_goal_overshoot_m=max_overshoot,max_yaw_overshoot_deg=max_yaw_overshoot,
        max_alt_overshoot_m=max_z_overshoot,max_lateral_deviation_m=max_lateral,
        max_roll_abs_deg=max_roll,max_pitch_abs_deg=max_pitch,
        max_yaw_rate_deg_s=max_yaw_rate,max_xy_speed_m_s=max_speed,max_z_error_m=max_zerr,
        max_rpm=max_rpm,motor_saturation_steps=motor_sat_count,
        final_xy_error_m=final_xy_err,final_xy_speed_m_s=final_xy_speed,
        final_yaw_error_deg=final_yaw_err,final_yaw_rate_deg_s=final_yaw_rate,
        final_z_error_m=final_z_err,final_z_speed_m_s=final_z_speed,
        xy_brake_guard_steps=brake_count,
    )


def _tune_run_group(env,cases,raw_writer,trial_writer,trial_counter,stage,parameter,value,rep=0,yaw_mode="original"):
    rows=[]
    for scenario in cases:
        trial_counter[0]+=1
        # 기존 함수를 수정하지 않기 위해 전역 목표점을 매 사례 설정 후 다시 복구.
        original_goal=TARGET_XY_M.copy()
        try:
            globals()['TARGET_XY_M']=np.asarray(scenario['goal'],dtype=float)
            r=_tune_case(env,scenario,raw_writer,trial_counter[0],stage,parameter,value,rep,yaw_mode=yaw_mode)
        finally:
            globals()['TARGET_XY_M']=original_goal
        rows.append(r)
        trial_writer.writerow(r)
    return rows


def _tune_group_summary(rows):
    passing=all(r['status']=='PASS' for r in rows)
    durations=[float(r['settle_time_s'])-START_HOVER_S for r in rows if r['settle_time_s'] is not None]
    score=float(np.mean(durations)) if len(durations)==len(rows) and rows else float('inf')
    return passing,score


def _tune_plot(rows,outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['parameter'] for r in rows if r['stage'] not in ('baseline','audit')))
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    for p in names:
        group=[r for r in rows if r['parameter']==p and r['repeat']==0 and r['stage']=='candidate']
        by={}
        for r in group:by.setdefault(float(r['candidate_value']),[]).append(r)
        xs=[];ys=[]
        for val,grp in sorted(by.items()):
            xs.append(val)
            if all(x['status']=='PASS' for x in grp):
                ys.append(np.mean([float(x['settle_time_s'])-START_HOVER_S for x in grp]))
            else:ys.append(np.nan)
        if xs:axes[0].plot(xs,ys,marker='.',label=p)
    axes[0].set_title('PASS-only average settling time (separate parameter value scales)')
    axes[0].set_xlabel('candidate parameter value (different units)')
    axes[0].set_ylabel('Mean settling time after departure [s]')
    if axes[0].lines:axes[0].legend(fontsize=7,ncol=2)
    bykind={}
    for r in rows:
        if r['stage']=='baseline':bykind.setdefault(r['kind'],[]).append(r)
    labels=list(bykind);passed=[100*sum(x['status']=='PASS' for x in bykind[k])/len(bykind[k]) for k in labels]
    axes[1].bar(labels,passed);axes[1].set_ylim(0,100)
    axes[1].set_title('Baseline PASS rate by scenario family')
    axes[1].set_ylabel('PASS rate [%]')
    for ax in axes:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(outpath,dpi=130);plt.close(fig)


def run_progressive_tuning():
    """구성된 모든 실험을 한 프로세스에서 자동 실행하고 ZIP 1개로 출력."""
    import json
    from zipfile import ZipFile,ZIP_DEFLATED
    from collections import Counter
    scenarios,steps_cap=_tune_scenarios(TUNE_PROFILE)
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    prefixes='drone_motion_tune'
    rawpath=LOG_DIR/f'{prefixes}.csv'
    trialpath=LOG_DIR/f'{prefixes}_trials.csv'
    reportpath=LOG_DIR/f'{prefixes}.txt'
    cfgpath=LOG_DIR/f'{prefixes}_config.json'
    plotpath=LOG_DIR/f'{prefixes}.png'
    zippath=LOG_DIR/f'{prefixes}_bundle.zip'
    original={n:float(globals()[n]) for n in _TUNE_BASE_NAMES}
    current=original.copy()
    # 기존 (0,0,100)->(10,10) 검증 조건에서 제어 수치 그대로 시작하되,
    # 매 실험의 초기 고도를 150m로 변경한다 (충돌 여유 확보).
    env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
        initial_xyzs=np.array([[0.,0.,TUNE_START_ALT_M]]),initial_rpys=np.zeros((1,3)),
        pyb_freq=SIM_FREQ,ctrl_freq=SIM_FREQ,gui=TUNE_GUI,user_debug_gui=False)
    start=time.perf_counter();results=[];stage_decisions=[];trial_counter=[0];error=None
    print(f'Progressive tuning: {TUNE_PROFILE}, initial height {TUNE_START_ALT_M}m',flush=True)
    print('All tested settings start from previous no-overshoot navigation controller.',flush=True)
    def run_cases(cases,stage,param,value,rep,rw,tw):
        r=_tune_run_group(env,cases,rw,tw,trial_counter,stage,param,value,rep)
        results.extend(r)
        return _tune_group_summary(r),r
    try:
        with rawpath.open('w',newline='',encoding='utf-8-sig') as f,trialpath.open('w',newline='',encoding='utf-8-sig') as s:
            rw=csv.DictWriter(f,fieldnames=_TUNE_RAW_COLUMNS)
            tw=csv.DictWriter(s,fieldnames=_TUNE_TRIAL_COLUMNS)
            rw.writeheader();tw.writeheader()
            # (A) baseline. 같은 초기 조건을 반복해 이 조건의 실제 무오버슈트를 검증.
            baseline_valid={}
            for kind in ('xy','yaw','alt','combined'):
                (ok,score),rs=run_cases(scenarios[kind],'baseline','baseline',0.,0,rw,tw)
                baseline_valid[kind]=ok
                print(f'[baseline/{kind}] {len(rs)} cases PASS={ok}, score={score:.3f}s; '+
                      str(Counter(x['status'] for x in rs)),flush=True)
                f.flush();s.flush()
            # (B) 1개 파라미터만 변경. baseline 실패 축은 일단 건드리지 않는다.
            for name,kind,increment,cap in _TUNE_STAGE_PLAN:
                startval=current[name];bestval=startval;best_score=float('inf')
                tested=0;last_safe=startval;failure='';last_reject=None
                if not baseline_valid[kind]:
                    stage_decisions.append(dict(parameter=name,kind=kind,start=startval,
                        selected=startval,last_pass=startval,cases_tested=0,
                        decision='SKIPPED_BASELINE_FAILED'))
                    print(f'[SKIP] {name}: baseline {kind} not PASS',flush=True)
                    continue
                # baseline 현재 설정의 동일 사례 score로 후보와 정확히 비교.
                (bs,bscore),_ = run_cases(scenarios[kind],'reference',name,startval,0,rw,tw)
                if not bs:
                    stage_decisions.append(dict(parameter=name,kind=kind,start=startval,
                        selected=startval,last_pass=startval,cases_tested=0,
                        decision='SKIPPED_REFERENCE_FAILED'))
                    continue
                best_score=bscore
                safe_scores=[(startval,bscore)]
                value=startval;lower=startval;upper=None
                while tested<steps_cap:
                    proposal=min(cap,round(value+increment,9))
                    if proposal<=value+1e-10:break
                    globals()[name]=proposal
                    (passed,score),rs=run_cases(scenarios[kind],'candidate',name,proposal,0,rw,tw)
                    tested+=1
                    # no overshoot and stable are mandatory; do not trade safety for speed.
                    if passed:
                        value=proposal;lower=proposal;last_safe=proposal
                        safe_scores.append((proposal,score))
                        if score<best_score-1e-8:best_score=score;bestval=proposal
                        print(f'[PASS] {name}: {proposal:g} score={score:.3f}s',flush=True)
                    else:
                        upper=proposal
                        failure=str(Counter(r['status'] for r in rs))
                        last_reject=proposal
                        print(f'[STOP] {name}: {proposal:g} rejected: {failure}',flush=True)
                        break
                    if tested%TUNE_PROGRESS_EVERY==0:
                        f.flush();s.flush()
                # 실패 경계의 제어 한계를 2분 탐색하되, 소프트값 1개만 조정.
                if upper is not None and TUNE_PROFILE!='smoke':
                    for _ in range(TUNE_EXTRA_REFINEMENTS):
                        middle=round((lower+upper)/2.,9)
                        if middle<=lower+1e-9:break
                        globals()[name]=middle
                        (passed,score),rs=run_cases(scenarios[kind],'refinement',name,middle,0,rw,tw)
                        tested+=1
                        if passed:
                            lower=middle;last_safe=middle
                            safe_scores.append((middle,score))
                            if score<best_score-1e-8:best_score=score;bestval=middle
                        else:upper=middle
                # 빠르지 않아도 동일 성능인 한도에서는 최댓값을 유지해 다음 축 한계를 열어준다.
                # 다만 확실히 느려지는 경우(평균 정착시간 > 최고값+50ms)만 보류.
                selected=max(v for v,sc in safe_scores
                             if sc<=best_score+TUNE_TIME_REGRESSION_BUDGET_S)
                globals()[name]=selected;current[name]=selected
                stage_decisions.append(dict(parameter=name,kind=kind,start=startval,
                    selected=selected,fastest_tested_value=bestval,last_pass=last_safe,cases_tested=tested,
                    reference_score_s=bscore,fastest_tested_score_s=best_score,
                    first_failure=last_reject,reject_reason=failure,decision='SEARCHED'))
                print(f'[SELECT] {name}: {startval:g} -> {selected:g} '+
                      f'(fastest tested={bestval:g}, score={best_score:.3f}s)',flush=True)
                f.flush();s.flush()
            # (C) 조합 평가: 수정된 전 파라미터를 함께 사용해 XY/Yaw/고도 동시 동작 확인.
            # baseline combined failure가 있었다면 '검증 완료'라고 보고하지 않는다.
            for kind in ('xy','yaw','alt','combined'):
                (ok,score),rs=run_cases(scenarios[kind],'audit','final',0.,0,rw,tw)
                print(f'[FINAL AUDIT] {kind}: PASS={ok}, score={score:.3f}s, '+
                      str(Counter(x['status'] for x in rs)),flush=True)
            # 진짜 반복 시험: 선정 파라미터 그대로 전체 시나리오를 재실행.
            if TUNE_PROFILE=='full':
                for rep in range(1,TUNE_EXTRA_RETESTS+1):
                    for kind in ('xy','yaw','alt','combined'):
                        (ok,score),_ = run_cases(scenarios[kind],'audit','final',0.,rep,rw,tw)
                        print(f'[REPEAT {rep}] {kind}: PASS={ok}, {score:.3f}s',flush=True)
            f.flush();s.flush()
    except Exception as exc:
        error=exc
        print('TUNER ERROR. Completed partial CSVs remain available:',repr(exc),flush=True)
    finally:
        for n,val in original.items():globals()[n]=val
        env.close()
        total=Counter(r['status'] for r in results)
        audit=[r for r in results if r['stage']=='audit']
        audit_pass=bool(audit) and all(r['status']=='PASS' for r in audit)
        lines=[
            'CF2X PROGRESSIVE MOTION TUNING, 240 Hz',
            'Original controller: proven distance-based target approach + predictive braking',
            f'Profile: {TUNE_PROFILE}, start altitude={TUNE_START_ALT_M}m',
            f'Completed trials={len(results)}, sample rows={sum(int(r["n_steps"]) for r in results)}',
            f'Result counts={dict(total)}',
            'Overrun threshold: XY>0.001 m, yaw>0.025 deg, Z>0.005 m',
            'Motor RPM clipping=FAIL. Unsettled/aborted=FAIL. PASS requires stability+post-check.',
            'Initial baseline verification:',
        ]
        for kind in ('xy','yaw','alt','combined'):
            base=[r for r in results if r['stage']=='baseline' and r['kind']==kind]
            lines.append(f'  {kind}: {dict(Counter(r["status"] for r in base))}')
        lines.extend(['\nSequential one-at-a-time adjustments:'])
        for d in stage_decisions:
            lines.append('  '+str(d))
        lines.extend(['\nFinal cross-axis audit:',
                      f'  PASSED_ALL={audit_pass}',
                      '  Test-specific PASS is not a universal mathematical zero-overshoot guarantee.',
                      '  Cases with failed baseline are not allowed to silently certify a tuned limit.',
                      '  If any final audit failed, do not deploy the selected combined settings.',
                      f'  RECOMMENDED={current if audit_pass else original}',
                      '  Physical MAX_RPM remains enforced.\n'])
        for kind in ('xy','yaw','alt','combined'):
            group=[r for r in audit if r['kind']==kind]
            lines.append(f'  {kind}: '+str(dict(Counter(r['status'] for r in group))))
        reportpath.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        cfg=dict(profile=TUNE_PROFILE,sim_hz=SIM_FREQ,start_altitude=TUNE_START_ALT_M,
                 original_settings=original,tuned_settings_proposed=current,
                 recommended_settings=(current if audit_pass else original),
                 baseline_verification={k:bool(v) for k,v in (baseline_valid if 'baseline_valid' in locals() else {}).items()},
                 stage_decisions=stage_decisions,final_audit_passed=audit_pass,
                 sample_count=sum(int(r['n_steps']) for r in results),trial_count=len(results),
                 criteria=dict(xy_overshoot_epsilon_m=TUNE_XY_OVERSHOOT_EPS_M,
                    yaw_overshoot_epsilon_deg=TUNE_YAW_OVERSHOOT_EPS_DEG,
                    z_overshoot_epsilon_m=TUNE_Z_OVERSHOOT_EPS_M,
                    stable_required_s=TUNE_STABLE_REQUIRED_S,
                    post_settle_observe_s=TUNE_OBSERVE_AFTER_SETTLE_S,
                    max_duration_s=TUNE_MAX_TRIAL_S,
                    max_mean_settling_regression_s=TUNE_TIME_REGRESSION_BUDGET_S),scenario_families=scenarios,
                 assumptions='Config automatically selected by simulated 240Hz trials, not real-world proof')
        cfgpath.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        try:
            _tune_plot(results,plotpath)
        except Exception as exc:
            print('Could not plot results:',repr(exc))
        files=[rawpath,trialpath,reportpath,cfgpath,plotpath]
        with ZipFile(zippath,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as zf:
            for path in files:
                if path.exists():zf.write(path,arcname=path.name)
        print('Fixed-name bundle:',zippath,flush=True)
        print('Total trials:',len(results),'final audit:',audit_pass,flush=True)
    if error is not None:raise RuntimeError('Progressive tuning incomplete; inspect drone_motion_tune_bundle.zip') from error
    return results


# =================== 8. Yaw/고도 무오버슈트 기반 순차 튜닝 ===================
# 이전 545회 결과: XY (3.0, 2.7, 2.05, 3.6875, 25°)는 XY 단독에서만 통과.
# 따라서 여기에서는 확정된 통합 성능이라고 주장하지 않고, Yaw/고도/동시 감사 필수.
# 실행: EXPERIMENT_MODE='settling_tune', TUNE_PROFILE='full'
# 파일: logs/drone_motion_tune*.{csv,txt,json,png,zip} (이전 파일 덮어쓰기).

# 복구(기준선 실패 시)와 성능 향상(기준선 통과 이후)을 별도로 실시한다.
# 한 후보당 변경하는 전역 변수는 반드시 하나이며, 불합격 시 복원한다.
_SETTLE_RESCUE = {
    'yaw': [
        ('YAW_STOP_LEAD_S', (0.16, 0.20, 0.28, 0.36, 0.50, 0.65)),
        ('YAW_TERMINAL_RATE_GAIN', (1.2, 1.0, 0.8, 0.6, 0.4)),
        ('YAW_BRAKE_ACCEL_DEG_S2', (30., 25., 20., 15., 10.)),
        ('YAW_RATE_TRACK_K', (6., 8., 10., 12.)),
    ],
    'alt': [
        ('ALT_LOOKAHEAD_S', (0.5, 0.6, 0.8, 1.0, 1.2, 1.5)),
        ('KD_ALT', (2.7, 2.9, 3.1, 3.3, 3.5, 4.0, 4.5)),
        ('KP_ALT', (1.8, 1.6, 1.4, 1.2)),
    ],
}
_SETTLE_PERFORMANCE = [
    # 첫번째 그룹: 더 큰 Yaw 회전 성능을 시험한다.
    ('MAX_YAW_RATE_DEG_S',     'yaw', +15., 195.),
    ('MAX_YAW_ACCEL_DEG_S2',   'yaw', +20., 240.),
    ('YAW_BRAKE_ACCEL_DEG_S2', 'yaw',  +5., 100.), # 값이 커지면 제동을 늦게 시작하므로 실패할 수 있다.
    ('YAW_TORQUE_RESERVE',     'yaw', +0.05, 1.0),
    ('YAW_TERMINAL_RATE_GAIN', 'yaw', +0.10, 2.6),
    ('YAW_STOP_LEAD_S',        'yaw', -0.02, 0.0), # 여유를 축소해도 통과하는지 검증.
    # 두번째 그룹: 단독 고도 상승/하강과 정밀 정착.
    ('MAX_VERTICAL_ACCEL',     'alt', +0.5, 9.0),
    ('KP_ALT',                 'alt', +0.2, 4.0),
    ('ALT_LOOKAHEAD_S',       'alt', -0.05, 0.0),
]
_SETTLE_PARAMETER_NAMES = sorted(set(
    [x[0] for fam in _SETTLE_RESCUE.values() for x in fam]
    + [x[0] for x in _SETTLE_PERFORMANCE]
    + ['KD_ALT','KP_ALT','MAX_VERTICAL_ACCEL',
       'MAX_XY_ACCEL','APPROACH_ACCEL_M_S2','BRAKE_ACCEL_M_S2',
       'MAX_APPROACH_SPEED_M_S','MAX_XY_TILT_DEG',
       'YAW_USE_STOP_GUARD','YAW_STOP_MARGIN_DEG']))


def _settle_scenario_status(rows):
    """성공 판정은 느슨한 평균이 아닌 모든 사례의 PASS와 정착 여부."""
    passed,score=_tune_group_summary(rows)
    from collections import Counter
    return passed,score,dict(Counter(r['status'] for r in rows))


def _settle_candidate_loss(rows):
    """보정 탐색용 임시 손실. PASS가 아니면 안전/정착 인증이 아님.

    물리 안전 중단과 모터 포화를 우선, 목표 초과량을 그 다음에 평가.
    이 점수는 통과 판정을 대체하지 않으며 임시 보정 방향 선택에만 사용한다.
    """
    hard=sum(r['status'] in ('ABORT','SATURATED') for r in rows)
    excess=0.
    for r in rows:
        excess += max(0.,r['max_goal_overshoot_m']-TUNE_XY_OVERSHOOT_EPS_M) / TUNE_XY_OVERSHOOT_EPS_M
        excess += max(0.,r['max_yaw_overshoot_deg']-TUNE_YAW_OVERSHOOT_EPS_DEG) / TUNE_YAW_OVERSHOOT_EPS_DEG
        excess += max(0.,r['max_alt_overshoot_m']-TUNE_Z_OVERSHOOT_EPS_M) / TUNE_Z_OVERSHOOT_EPS_M
    timeout=sum(r['status']=='TIMEOUT' for r in rows)
    enderr=sum(r['final_yaw_error_deg']/0.25+r['final_z_error_m']/0.10+
               r['final_xy_error_m']/0.10 for r in rows)
    return 10000.*hard+10.*excess+30.*timeout+0.01*enderr


def run_settling_tuning():
    """초기 150m, 원본 240Hz + 후보별 요약. 모든 반복의 파일명은 고정한다.

    1) XY+개선 Yaw+개선 ALT 기준 검사
    2) 기준선에서 실패하면 *한 파라미터씩* 보수화하여 복구
    3) PASS 기준선에서 속도/가속도 한계를 하나씩 증가, 무오버슈트 실패는 즉시 거부
    4) 수평 이동/회전/고도 동시 감사. 실패 시 시험 통과값도 미검증으로 표시
    """
    import json
    from zipfile import ZipFile, ZIP_DEFLATED
    from collections import Counter
    scenarios, steps_cap=_tune_scenarios(TUNE_PROFILE)
    base={n:globals()[n] for n in _SETTLE_PARAMETER_NAMES}
    selected=base.copy()
    log_prefix='drone_motion_tune'
    rawpath=LOG_DIR/f'{log_prefix}.csv'
    trialpath=LOG_DIR/f'{log_prefix}_trials.csv'
    reportpath=LOG_DIR/f'{log_prefix}.txt'
    cfgpath=LOG_DIR/f'{log_prefix}_config.json'
    plotpath=LOG_DIR/f'{log_prefix}.png'
    zippath=LOG_DIR/f'{log_prefix}_bundle.zip'
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
        initial_xyzs=np.array([[0.,0.,TUNE_START_ALT_M]]),
        initial_rpys=np.zeros((1,3)),pyb_freq=SIM_FREQ,ctrl_freq=SIM_FREQ,
        gui=TUNE_GUI,user_debug_gui=False)
    results=[];decisions=[];trial_counter=[0];error=None
    began=time.perf_counter()
    max_attempts=2 if TUNE_PROFILE=='smoke' else (5 if TUNE_PROFILE=='standard' else steps_cap)
    print(f'[SETTLING TUNE] profile={TUNE_PROFILE}, z0={TUNE_START_ALT_M}m, '+
          f'XY previously measured={tuple(selected[n] for n in ("MAX_XY_ACCEL","APPROACH_ACCEL_M_S2","BRAKE_ACCEL_M_S2","MAX_APPROACH_SPEED_M_S","MAX_XY_TILT_DEG"))}',flush=True)

    def run_set(cases, stage, parameter, value, rep, rw,tw):
        rr=_tune_run_group(env,cases,rw,tw,trial_counter,stage,parameter,value,rep)
        results.extend(rr)
        passed,score,status=_settle_scenario_status(rr)
        return passed,score,status

    def set_global(name,value):
        globals()[name]=float(value) if name!='YAW_USE_STOP_GUARD' else bool(value)
        selected[name]=globals()[name]

    baseline_check={};rescued={}
    try:
        with rawpath.open('w',newline='',encoding='utf-8-sig') as rf, trialpath.open('w',newline='',encoding='utf-8-sig') as sf:
            rw=csv.DictWriter(rf,fieldnames=_TUNE_RAW_COLUMNS)
            tw=csv.DictWriter(sf,fieldnames=_TUNE_TRIAL_COLUMNS)
            rw.writeheader();tw.writeheader()

            # A) 최초 기준선: XY 보정값이 포함된 제어기로 시작한다.
            for fam in ('xy','yaw','alt','combined'):
                passed,score,status=run_set(scenarios[fam],'baseline','baseline',0.,0,rw,tw)
                baseline_check[fam]={'passed':passed,'settling_s':score,'statuses':status}
                print(f'[BASELINE/{fam}] PASS={passed} score={score:.4f} status={status}',flush=True)
                rf.flush();sf.flush()

            # B) 기준선 복구: Yaw/고도 각각 실패하면 여유를 단계적으로 높인다.
            # 단일 변수로 곧바로 PASS가 안 나더라도 실패 정도가 개선되는 후보는
            # "잠정"으로 유지하고 다음 변수의 복구를 계속 시도한다.
            # 최종 FAIL 상태는 절대로 성공 설정으로 인정하지 않는다.
            for fam in ('yaw','alt'):
                if baseline_check[fam]['passed']:
                    rescued[fam]=True
                    continue
                rescued[fam]=False
                ref_pass,ref_score,_=run_set(scenarios[fam],'repair_reference','initial',0.,0,rw,tw)
                previous_loss=_settle_candidate_loss(results[-len(scenarios[fam]):])
                for name,options in _SETTLE_RESCUE[fam]:
                    original_val=selected[name]
                    passes=[];improving=[]
                    for candidate in options[:max_attempts]:
                        set_global(name,candidate)
                        passed,score,status=run_set(scenarios[fam],'repair',name,candidate,0,rw,tw)
                        loss=_settle_candidate_loss(results[-len(scenarios[fam]):])
                        decisions.append(dict(step='repair',group=fam,parameter=name,
                            value=candidate,passed=passed,mean_settle_s=score,status=status,
                            tentative_loss=loss))
                        print(f'[REPAIR/{fam}] {name}={candidate:g}: {status} '
                              f'score={score:.3f}, provisional_loss={loss:.1f}',flush=True)
                        if passed:passes.append((score,candidate))
                        else:improving.append((loss,candidate))
                        rf.flush();sf.flush()
                    if passes:
                        score,val=min(passes)
                        set_global(name,val)
                        rescued[fam]=True
                        print(f'[REPAIR SELECT/{fam}] {name}={val:g} settle={score:.3f}s',flush=True)
                        break
                    if improving:
                        loss,val=min(improving)
                        if loss+1e-8<previous_loss:
                            set_global(name,val)
                            previous_loss=loss
                            decisions.append(dict(step='repair',group=fam,parameter=name,
                                                  provisional=val,decision='PROVISIONAL_IMPROVEMENT_NOT_PASS'))
                            print(f'[REPAIR PROVISIONAL/{fam}] {name}={val:g} '
                                  f'(NOT YET PASS, loss={loss:.1f})',flush=True)
                            continue
                    set_global(name,original_val)
                if not rescued[fam]:
                    print(f'[BLOCKED/{fam}] 설정을 일부 보수화했지만, 기준선 전체 PASS는 아직 없음.',flush=True)

            # C) 안정화된 기준 설정을 새 출발점으로 검증.
            start_status={}
            for fam in ('xy','yaw','alt','combined'):
                passed,score,status=run_set(scenarios[fam],'repaired_baseline','after_repair',0.,0,rw,tw)
                start_status[fam]=dict(passed=passed,mean_settling_s=score,status=status)
                print(f'[REPAIRED BASELINE/{fam}] {passed} {status}',flush=True)
                rf.flush();sf.flush()

            # D) 복구된 그룹의 목표 성능 제한값을 한 번에 하나만 조정한다.
            #    실패한 수치는 버리고 직전 안전 수치를 남긴다.
            for name,fam,increment,cap in _SETTLE_PERFORMANCE:
                starting=selected[name]
                if not start_status[fam]['passed']:
                    decisions.append(dict(step='optimize',parameter=name,group=fam,
                        start=starting,selected=starting,decision='SKIP_UNSAFE_BASELINE'))
                    continue
                passed,initial_score,_=run_set(scenarios[fam],'reference',name,starting,0,rw,tw)
                if not passed:
                    decisions.append(dict(step='optimize',parameter=name,group=fam,
                        start=starting,selected=starting,decision='SKIP_REFERENCE_FAIL'))
                    continue
                safe=[(starting,initial_score)]
                previous=starting;reject=None;tested=0
                best=(initial_score,starting)
                # 탐색 방향이 음수이면 cap은 하한이다.
                while tested<max_attempts:
                    candidate=(min(cap,previous+increment) if increment>0
                               else max(cap,previous+increment))
                    candidate=round(candidate,9)
                    if abs(candidate-previous)<1e-10:break
                    set_global(name,candidate)
                    ok,score,status=run_set(scenarios[fam],'candidate',name,candidate,0,rw,tw)
                    tested+=1
                    print(f'[CANDIDATE/{fam}] {name}={candidate:g} PASS={ok} score={score:.3f}, {status}',flush=True)
                    if ok:
                        previous=candidate
                        safe.append((candidate,score))
                        if score<best[0]-1e-8:best=(score,candidate)
                    else:
                        reject=candidate
                        break
                    rf.flush();sf.flush()
                # 정밀화: 위에서 실패한 경계만 중간값으로 재시험.
                if reject is not None and TUNE_PROFILE=='full':
                    lo=previous;hi=reject
                    for _ in range(TUNE_EXTRA_REFINEMENTS):
                        candidate=round((lo+hi)/2.,9)
                        if abs(candidate-lo)<1e-9:break
                        set_global(name,candidate)
                        ok,score,status=run_set(scenarios[fam],'refinement',name,candidate,0,rw,tw)
                        tested+=1
                        if ok:
                            lo=candidate;safe.append((candidate,score))
                            if score<best[0]-1e-8:best=(score,candidate)
                        else:hi=candidate
                # 실험에서 빠른 값 우선. 비슷한 성능이면 더 적은 제한 완화량을 채택한다.
                allowed=[(v,sc) for v,sc in safe
                         if sc<=best[0]+TUNE_TIME_REGRESSION_BUDGET_S]
                chosen=min(allowed,key=lambda x:(abs(x[0]-starting),x[1]))[0]
                set_global(name,chosen)
                decisions.append(dict(step='optimize',group=fam,parameter=name,
                    start=starting,selected=chosen,fastest_value=best[1],
                    fastest_score_s=best[0],first_failure=reject,
                    tested_steps=tested,decision='SEARCHED'))
                print(f'[SELECT/{fam}] {name}: {starting:g} -> {chosen:g} '+
                      f'(fastest={best[1]:g}, first_failure={reject})',flush=True)
                rf.flush();sf.flush()

            # E) 복합 비행에서만 실패하는 경우, 한 변수씩 보수적으로 조정한다.
            # XY 최고속도 3.6875m/s는 XY 전용 PASS이므로 다축에서는 제한을 낮출 수 있음.
            if start_status['yaw']['passed'] and start_status['alt']['passed']:
                ok,score,_=run_set(scenarios['combined'],'joint_reference','combined',0.,0,rw,tw)
                if not ok:
                    joint_attempts=[
                        ('MAX_APPROACH_SPEED_M_S',(3.5,3.3,3.1,2.9,2.7,2.5)),
                        ('BRAKE_ACCEL_M_S2',(1.95,1.85,1.7,1.5,1.2,0.9)),
                        ('YAW_STOP_LEAD_S',(0.16,0.2,0.28,0.38,0.5)),
                        ('ALT_LOOKAHEAD_S',(0.5,0.6,0.8,1.0)),
                    ]
                    for name,options in joint_attempts:
                        startval=selected[name];passing=[]
                        for val in options[:max_attempts]:
                            if abs(val-startval)<1e-10:continue
                            set_global(name,val)
                            valid,score,status=run_set(scenarios['combined'],'joint_repair',name,val,0,rw,tw)
                            decisions.append(dict(step='joint_repair',parameter=name,value=val,
                                                 pass_all=valid,mean_settle_s=score,status=status))
                            if valid:passing.append((score,val))
                            print(f'[JOINT REPAIR] {name}={val:g} PASS={valid} {status}',flush=True)
                            rf.flush();sf.flush()
                        if passing:
                            score,val=min(passing)
                            set_global(name,val)
                            print(f'[JOINT REPAIR SELECT] {name}={val:g} score={score:.3f}',flush=True)
                            break
                        set_global(name,startval)

            # E) 통합 검증. 각 축 각각의 PASS라도 혼합 자세와 모터 추력 공유로 실패 가능.
            final_audit={}
            for fam in ('xy','yaw','alt','combined'):
                for rep in range((1+TUNE_EXTRA_RETESTS) if TUNE_PROFILE=='full' else 1):
                    ok,score,status=run_set(scenarios[fam],'audit','final',0.,rep,rw,tw)
                    final_audit[f'{fam}_rep{rep}']=dict(pass_all=ok,mean_settle_s=score,status=status)
                    print(f'[FINAL AUDIT/{fam}/rep{rep}] PASS={ok}, {status}',flush=True)
                    rf.flush();sf.flush()
    except Exception as exc:
        error=exc
        print('[TUNER ERROR] partial CSV saved: ',repr(exc),flush=True)
    finally:
        for name,value in base.items():globals()[name]=value
        env.close()
        by_status=dict(Counter(r['status'] for r in results))
        audit_results=[r for r in results if r['stage']=='audit']
        audited=bool(audit_results) and all(r['status']=='PASS' for r in audit_results)
        config=dict(mode='settling_tune',profile=TUNE_PROFILE,sim_freq=SIM_FREQ,
                    initial_height_m=TUNE_START_ALT_M,trial_count=len(results),
                    raw_samples=sum(r['n_steps'] for r in results),
                    xy_parameters_from_prior_experiment={n:base[n] for n in (
                        'MAX_XY_ACCEL','APPROACH_ACCEL_M_S2','BRAKE_ACCEL_M_S2',
                        'MAX_APPROACH_SPEED_M_S','MAX_XY_TILT_DEG')},
                    initial_settings=base.copy(),repaired_baseline=start_status if 'start_status' in locals() else {},
                    initial_baseline=baseline_check,repair_status=rescued,
                    proposals=selected.copy(),
                    recommended_settings=(selected.copy() if audited else None),
                    final_cross_axis_audit_passed=audited,
                    final_audit=final_audit if 'final_audit' in locals() else {},
                    decisions=decisions,result_counts=by_status,
                    criteria=dict(yaw_overshoot_eps_deg=TUNE_YAW_OVERSHOOT_EPS_DEG,
                        altitude_overshoot_eps_m=TUNE_Z_OVERSHOOT_EPS_M,
                        xy_overshoot_eps_m=TUNE_XY_OVERSHOOT_EPS_M,
                        yaw_angle_tolerance_deg=TUNE_YAW_ANGLE_TOL_DEG,
                        yaw_rate_tolerance_deg_s=TUNE_YAW_RATE_TOL_DEG_S,
                        altitude_tolerance_m=TUNE_Z_POSITION_TOL_M,
                        altitude_speed_tolerance_m_s=TUNE_Z_SPEED_TOL_M_S),
                    caveat='All PASS only for tested 240Hz deterministic simulated cases; XY values were previous XY-only passes.')
        report=[
            'CF2X YAW + ALTITUDE SETTLING TUNING, 240 Hz',
            f'Initial altitude: {TUNE_START_ALT_M} m; profile: {TUNE_PROFILE}',
            f'Completed trials: {len(results)}, raw samples: {config["raw_samples"]}',
            f'Status counts: {by_status}',
            'Existing XY values: '+str(config['xy_parameters_from_prior_experiment']),
            '\nBaseline: '+str(baseline_check),
            '\nRepaired baseline: '+str(config['repaired_baseline']),
            '\nParameter decisions:',
        ]
        report.extend(str(item) for item in decisions)
        report.extend(['\nFinal combined audit:',str(config['final_audit']),
                       f'PASS_ALL_AXES={audited}',
                       'Recommended='+(str(config['recommended_settings']) if audited else 'NONE: combined validation failed'),
                       'Unaccepted candidate values must not be treated as verified.',
                       'Yaw torques are bounded by CF2X motor RPM/physical allocation.',
                       'PASS allows numerical thresholds; this is not proof of mathematically zero overshoot.'])
        reportpath.write_text('\n'.join(report)+'\n',encoding='utf-8')
        def json_safe(v):
            if isinstance(v,dict):return {str(k):json_safe(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [json_safe(x) for x in v]
            if isinstance(v,float) and not np.isfinite(v):return None
            if isinstance(v,np.generic):return json_safe(v.item())
            return v
        cfgpath.write_text(json.dumps(json_safe(config),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        try:_tune_plot(results,plotpath)
        except Exception as exc:print('[plot failed]',repr(exc),flush=True)
        with ZipFile(zippath,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as zf:
            for path in (rawpath,trialpath,reportpath,cfgpath,plotpath):
                if path.exists():zf.write(path,arcname=path.name)
        print(f'[COMPLETE] trial_count={len(results)}, combined PASS={audited}, file={zippath}',flush=True)
    if error is not None:raise RuntimeError('정착 튜닝 중단. 압축파일에 부분 로그가 남아 있음.') from error
    return config


# =============================================================================
# 10. 실측 최고 성능 주변 10개 다변수 조합 비교 (단일 실행)
# =============================================================================
# 출발점: 702회 settling_tune 결과의 proposals. 아래 값들은 단독축에서
# 통과했을 뿐, 복합 ±90deg에서 오버슈트가 있어 통합 승인된 설정은 아니다.
# 이 모드는 제어 구조/PD 토크 믹싱을 변경하지 않는다. 오직 설정값의 조합만 비교한다.
# 중요: 유효한 후보가 없으면 '최고 성능으로 검증된 1개'를 만들어내지 않는다.
#       대신 가장 근접한 진단 후보 하나를 표시하여 다음 실험 방향을 남긴다.

COMBO_START_ALT_M = 150.0
COMBO_EXTRA_AUDIT_REPEATS = 1   # 통과 후보만 동일 환경 재현성 확인; 독립 외란 검증은 아님
COMBO_GUI = False
COMBO_REAL_TIME = False
COMBO_SEED = {
    # 이전 545회 XY 단독 최적화 및 702회 Yaw/Z 단독 최적화에서 나온 잠정값
    'MAX_XY_ACCEL': 3.0,
    'APPROACH_ACCEL_M_S2': 2.7,
    'BRAKE_ACCEL_M_S2': 2.05,
    'MAX_APPROACH_SPEED_M_S': 3.6875,
    'MAX_XY_TILT_DEG': 25.0,
    'MAX_YAW_RATE_DEG_S': 60.0,
    'MAX_YAW_ACCEL_DEG_S2': 120.0,
    'YAW_BRAKE_ACCEL_DEG_S2': 40.0,
    'YAW_TORQUE_RESERVE': 0.75,
    'YAW_RATE_TRACK_K': 5.0,
    'YAW_TERMINAL_RATE_GAIN': 2.5,
    'YAW_STOP_LEAD_S': 0.0,
    'YAW_STOP_MARGIN_DEG': 0.20,
    'YAW_USE_STOP_GUARD': True,
    'MAX_VERTICAL_ACCEL': 7.5,
    'KP_ALT': 4.0,
    'KD_ALT': 2.5,
    'ALT_LOOKAHEAD_S': 0.275,
}

# 각각 완성된 파라미터 조합 1개를 뜻함 (Cartesian product가 아님).
# ±90deg 동시비행에서 Yaw 각도 드리프트가 관측되어 Yaw 감쇠/제동 여유를 집중 탐색.
# XY 브레이크 계획을 줄일 경우 XY 도착 시간이 늘거나 오버슈트가 생길 수도 있음.
COMBO_VARIATIONS = (
    ('C01_seed', '지난 실험 잠정 최적값 (복합 비행 실패한 대조군)', {}),
    ('C02_yaw_early', 'Yaw 선행 제동 + 약한 종단 속도',
     {'YAW_TERMINAL_RATE_GAIN':2.1,'YAW_BRAKE_ACCEL_DEG_S2':35.,
      'YAW_STOP_LEAD_S':0.06,'YAW_STOP_MARGIN_DEG':0.30}),
    ('C03_yaw_damping', 'Yaw 속도 추종 감쇠 강화',
     {'YAW_RATE_TRACK_K':7.,'YAW_TERMINAL_RATE_GAIN':2.2,
      'YAW_STOP_LEAD_S':0.10,'YAW_STOP_MARGIN_DEG':0.35}),
    ('C04_yaw_strong', 'Yaw 감쇠 강화 + 보수적인 제동 계획',
     {'YAW_RATE_TRACK_K':9.,'YAW_TERMINAL_RATE_GAIN':1.8,
      'YAW_BRAKE_ACCEL_DEG_S2':35.,'YAW_STOP_LEAD_S':0.12,
      'YAW_STOP_MARGIN_DEG':0.40}),
    ('C05_xy_yaw', '수평 제동 계획 축소 + Yaw 감쇠 강화',
     {'BRAKE_ACCEL_M_S2':1.7,'YAW_RATE_TRACK_K':7.,
      'YAW_TERMINAL_RATE_GAIN':2.2,'YAW_STOP_LEAD_S':0.08,
      'YAW_STOP_MARGIN_DEG':0.35}),
    ('C06_xy_yaw_safe', '수평·Yaw 제동 여유 확대',
     {'BRAKE_ACCEL_M_S2':1.5,'MAX_YAW_RATE_DEG_S':55.,
      'YAW_RATE_TRACK_K':8.,'YAW_TERMINAL_RATE_GAIN':1.8,
      'YAW_STOP_LEAD_S':0.12,'YAW_STOP_MARGIN_DEG':0.40}),
    ('C07_xy_more_brake', '수평 제동 조기화 + Yaw 정밀 감쇠',
     {'BRAKE_ACCEL_M_S2':1.2,'MAX_YAW_RATE_DEG_S':55.,
      'YAW_RATE_TRACK_K':9.,'YAW_TERMINAL_RATE_GAIN':1.6,
      'YAW_STOP_LEAD_S':0.15,'YAW_STOP_MARGIN_DEG':0.45}),
    ('C08_conservative', '빠른 속도보다 목표 통과 억제 우선',
     {'BRAKE_ACCEL_M_S2':0.9,'MAX_APPROACH_SPEED_M_S':3.4,
      'MAX_YAW_RATE_DEG_S':50.,'YAW_RATE_TRACK_K':8.,
      'YAW_TERMINAL_RATE_GAIN':1.4,'YAW_STOP_LEAD_S':0.18,
      'YAW_STOP_MARGIN_DEG':0.50}),
    ('C09_fast_damping', 'XY 속도 유지 + Yaw 속도 추종 강화',
     {'MAX_APPROACH_SPEED_M_S':3.4,'YAW_RATE_TRACK_K':8.,
      'MAX_YAW_ACCEL_DEG_S2':140.,'YAW_TERMINAL_RATE_GAIN':2.2,
      'YAW_STOP_LEAD_S':0.10,'YAW_STOP_MARGIN_DEG':0.35}),
    ('C10_balanced_z', 'Yaw/XY 보정 + 고도 정착 여유',
     {'BRAKE_ACCEL_M_S2':1.7,'MAX_APPROACH_SPEED_M_S':3.4,
      'MAX_YAW_RATE_DEG_S':55.,'YAW_RATE_TRACK_K':9.,
      'YAW_TERMINAL_RATE_GAIN':2.0,'YAW_STOP_LEAD_S':0.12,
      'YAW_STOP_MARGIN_DEG':0.45,'MAX_VERTICAL_ACCEL':7.0,
      'KP_ALT':3.8,'ALT_LOOKAHEAD_S':0.30}),
)


def _combo_build_candidates():
    out=[]
    assert len(COMBO_VARIATIONS)==10
    for label,description,changes in COMBO_VARIATIONS:
        settings=dict(COMBO_SEED)
        settings.update(changes)
        assert 0.0 < settings['YAW_TORQUE_RESERVE'] <= 1.0
        assert 0.0 < settings['MAX_YAW_RATE_DEG_S']
        assert settings['BRAKE_ACCEL_M_S2'] > 0.0
        assert 0.0 < settings['MAX_VERTICAL_ACCEL']
        assert settings['MAX_XY_ACCEL'] >= max(settings['BRAKE_ACCEL_M_S2'],
                                              settings['APPROACH_ACCEL_M_S2'])
        out.append(dict(label=label,description=description,settings=settings,
                        changes=dict(changes)))
    assert len({tuple(sorted(c['settings'].items())) for c in out})==10
    return out


def _combo_scenarios():
    # 기존 23개: XY 7, Yaw 6, 고도 4, 결합 6 (이전과 완전히 같은 조건)
    families,_ = _tune_scenarios('full')
    cases=[dict(s) for key in ('xy','yaw','alt','combined') for s in families[key]]
    # 새로운 6개 스트레스: 빠른 수평 이동, ±90/135도 회전, 고도 변화 동시 발생.
    # 여전히 150m 고도에서 출발하며 모두 완전히 동일한 초기 조건을 사용.
    stress=[
        dict(name='stress_long_p90_zup',goal=(18.,-6.),yaw=90.,dz=5.,kind='combined'),
        dict(name='stress_long_m90_zdown',goal=(-18.,6.),yaw=-90.,dz=-5.,kind='combined'),
        dict(name='stress_ne_p135',goal=(10.,10.),yaw=135.,dz=0.,kind='combined'),
        dict(name='stress_sw_m135',goal=(-10.,-10.),yaw=-135.,dz=0.,kind='combined'),
        dict(name='stress_x_p90_zup12',goal=(10.,0.),yaw=90.,dz=12.,kind='combined'),
        dict(name='stress_x_m90_zdown12',goal=(-10.,0.),yaw=-90.,dz=-12.,kind='combined'),
    ]
    cases.extend(stress)
    assert len(cases)==29 and len({s['name'] for s in cases})==29
    return cases


def _combo_evaluate(label, rows):
    """공정 비교: 모든 후보는 동일한 29개 케이스를 완료해야 함.

    하나라도 OVER/TIMEOUT/SATURATION/ABORT 발생하면 '승인' 불가.
    점수는 성공한 후보끼리만 비교하며, 4개의 시험군 평균을 가중 평균.
    """
    from collections import Counter
    n=len(rows)
    counts=dict(Counter(r['status'] for r in rows))
    family_times={}
    for family in ('xy','yaw','alt','combined'):
        group=[r for r in rows if r['kind']==family]
        ts=[float(r['settle_time_s'])-START_HOVER_S for r in group
            if r['status']=='PASS' and r['settle_time_s'] is not None]
        family_times[family]=(float(np.mean(ts)) if len(ts)==len(group) and len(group)>0 else None)
    weights={'xy':0.15,'yaw':0.25,'alt':0.20,'combined':0.40}
    score=(sum(weights[k]*v for k,v in family_times.items())
           if all(v is not None for v in family_times.values()) else None)
    over_xy=max((float(r['max_goal_overshoot_m']) for r in rows),default=0.)
    over_yaw=max((float(r['max_yaw_overshoot_deg']) for r in rows),default=0.)
    over_z=max((float(r['max_alt_overshoot_m']) for r in rows),default=0.)
    max_speed=max((float(r['max_xy_speed_m_s']) for r in rows),default=0.)
    min_rpm=max((float(r['max_rpm']) for r in rows),default=0.)
    sat=sum(int(r['motor_saturation_steps']) for r in rows)
    # 실패했을 때 가까운 후보를 찾는 참고용. PASS 판정과 결코 혼용하지 않음.
    normalized_excess=sum(
        max(0.,float(r['max_goal_overshoot_m'])-TUNE_XY_OVERSHOOT_EPS_M)/TUNE_XY_OVERSHOOT_EPS_M
        + max(0.,float(r['max_yaw_overshoot_deg'])-TUNE_YAW_OVERSHOOT_EPS_DEG)/TUNE_YAW_OVERSHOOT_EPS_DEG
        + max(0.,float(r['max_alt_overshoot_m'])-TUNE_Z_OVERSHOOT_EPS_M)/TUNE_Z_OVERSHOOT_EPS_M
        for r in rows)
    approved=(n==29 and counts.get('PASS',0)==29)
    return dict(candidate=label,eligible=approved,pass_count=counts.get('PASS',0),
                trial_count=n,status_counts=counts,
                weighted_mean_time_s=score,
                mean_xy_s=family_times['xy'],mean_yaw_s=family_times['yaw'],
                mean_alt_s=family_times['alt'],mean_combined_s=family_times['combined'],
                max_xy_overshoot_m=over_xy,max_yaw_overshoot_deg=over_yaw,
                max_alt_overshoot_m=over_z,max_xy_speed_m_s=max_speed,
                max_motor_rpm=min_rpm,motor_saturation_steps=sat,
                normalized_excess_score=normalized_excess)


def run_combo_search():
    """10개 멀티 파라미터 조합을 240Hz로 실험하고 합격한 것 중 1개 선정.

    출력: logs/drone_motion_combo_bundle.zip
    별도 접두사를 사용하여 과거 settling_tune 로그를 삭제하지 않음.
    """
    import json
    from collections import Counter
    from zipfile import ZipFile,ZIP_DEFLATED
    candidates=_combo_build_candidates()
    scenarios=_combo_scenarios()
    logstem='drone_motion_combo'
    rawpath=LOG_DIR/f'{logstem}.csv'
    trialpath=LOG_DIR/f'{logstem}_trials.csv'
    candidatepath=LOG_DIR/f'{logstem}_candidates.csv'
    txtpath=LOG_DIR/f'{logstem}.txt'
    cfgpath=LOG_DIR/f'{logstem}_config.json'
    plotpath=LOG_DIR/f'{logstem}.png'
    zippath=LOG_DIR/f'{logstem}_bundle.zip'
    # 원래 코드의 전역 설정은 시험마다 복원한다.
    backup={k:globals()[k] for k in COMBO_SEED}
    old_gui,old_time=GUI,REAL_TIME_PLAYBACK
    original_height=TUNE_START_ALT_M
    results=[];by_candidate={};evaluations=[];audits={}
    error=None;env=None;trial_counter=[0]
    started=time.perf_counter()
    print(f'[COMBO] 10 candidates x {len(scenarios)} scenarios = '
          f'{10*len(scenarios)} screening trials; altitude={COMBO_START_ALT_M}m; 240Hz',flush=True)
    try:
        globals()['GUI']=COMBO_GUI
        globals()['REAL_TIME_PLAYBACK']=COMBO_REAL_TIME
        globals()['TUNE_START_ALT_M']=COMBO_START_ALT_M
        env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
                       initial_xyzs=np.array([[0.,0.,COMBO_START_ALT_M]]),
                       initial_rpys=np.zeros((1,3)),pyb_freq=SIM_FREQ,
                       ctrl_freq=SIM_FREQ,gui=COMBO_GUI,user_debug_gui=False)
        with rawpath.open('w',newline='',encoding='utf-8-sig') as rf, \
             trialpath.open('w',newline='',encoding='utf-8-sig') as tf:
            raw_writer=csv.DictWriter(rf,fieldnames=_TUNE_RAW_COLUMNS)
            trial_writer=csv.DictWriter(tf,fieldnames=_TUNE_TRIAL_COLUMNS)
            raw_writer.writeheader();trial_writer.writeheader()
            for i,candidate in enumerate(candidates,1):
                for key,value in candidate['settings'].items():globals()[key]=value
                label=candidate['label']
                rows=_tune_run_group(env,scenarios,raw_writer,trial_writer,
                                     trial_counter,'candidate',label,i,0)
                results.extend(rows);by_candidate[label]=rows
                summary=_combo_evaluate(label,rows)
                evaluations.append(summary)
                print(f'[COMBO {i:02d}/10] {label} | PASS {summary["pass_count"]}/29 '
                      f'| worstYaw={summary["max_yaw_overshoot_deg"]:.4f}deg '
                      f'| eligible={summary["eligible"]} '
                      f'| score={summary["weighted_mean_time_s"]}',flush=True)
                rf.flush();tf.flush()
            eligible=[r for r in evaluations if r['eligible']]
            winner=min(eligible,key=lambda x:(x['weighted_mean_time_s'],
                      x['max_yaw_overshoot_deg'],x['candidate'])) if eligible else None
            # 합격자가 없으면 우승자가 없다고 기록한다. 진단 최선은 별개.
            diagnostic=min(evaluations,key=lambda x:(-x['pass_count'],
                x['normalized_excess_score'],x['max_yaw_overshoot_deg'],x['candidate']))
            if winner:
                wsettings=next(c['settings'] for c in candidates if c['label']==winner['candidate'])
                for key,value in wsettings.items():globals()[key]=value
                for rep in range(1,COMBO_EXTRA_AUDIT_REPEATS+1):
                    audit_rows=_tune_run_group(env,scenarios,raw_writer,trial_writer,
                                              trial_counter,'winner_audit',winner['candidate'],
                                              next(i for i,c in enumerate(candidates,1) if c['label']==winner['candidate']),rep)
                    results.extend(audit_rows)
                    audits[f'repeat_{rep}']=_combo_evaluate(winner['candidate'],audit_rows)
                    print(f'[WINNER AUDIT {rep}] PASS {audits[f"repeat_{rep}"]["pass_count"]}/29',flush=True)
                    rf.flush();tf.flush()
                if not all(a['eligible'] for a in audits.values()):
                    print('[WARNING] Winner failed repeat audit. No validated recommendation.',flush=True)
                    winner=None
    except Exception as ex:
        error=ex
        print('[COMBO ERROR] Partial raw/trial CSV remains: '+repr(ex),flush=True)
    finally:
        if env is not None:env.close()
        for key,value in backup.items():globals()[key]=value
        globals()['GUI']=old_gui
        globals()['REAL_TIME_PLAYBACK']=old_time
        globals()['TUNE_START_ALT_M']=original_height
        eligible=[r for r in evaluations if r['eligible']]
        # 사후 분기: 사용자에게 오해를 주지 않도록 불완전 실행은 유효 추천 불가
        complete=(error is None and len(evaluations)==len(candidates)
                  and all(x['trial_count']==len(scenarios) for x in evaluations))
        winners=[x for x in eligible if complete and
                 all(a['eligible'] for a in audits.values())]
        approved_winner=(min(winners,key=lambda x:(x['weighted_mean_time_s'],
                          x['max_yaw_overshoot_deg'],x['candidate']))
                         if winners and (not COMBO_EXTRA_AUDIT_REPEATS or audits) else None)
        diagnostic=(min(evaluations,key=lambda x:(-x['pass_count'],
                    x['normalized_excess_score'],x['max_yaw_overshoot_deg'],x['candidate']))
                    if evaluations else None)
        settings_by_label={c['label']:c['settings'] for c in candidates}
        fieldnames=('candidate','eligible','pass_count','trial_count','weighted_mean_time_s',
                    'mean_xy_s','mean_yaw_s','mean_alt_s','mean_combined_s',
                    'max_xy_overshoot_m','max_yaw_overshoot_deg','max_alt_overshoot_m',
                    'max_xy_speed_m_s','max_motor_rpm','motor_saturation_steps',
                    'normalized_excess_score')
        with candidatepath.open('w',newline='',encoding='utf-8-sig') as cf:
            wr=csv.DictWriter(cf,fieldnames=fieldnames,extrasaction='ignore')
            wr.writeheader();wr.writerows(evaluations)
        cfg=dict(mode='combo_search',complete=complete,
                 source='settling_tune 702 trials; proposals not combined-approved',
                 model='CF2X',sim_hz=SIM_FREQ,initial_altitude_m=COMBO_START_ALT_M,
                 candidate_count=len(candidates),scenario_count=len(scenarios),
                 expected_screening_trials=len(candidates)*len(scenarios),
                 actual_trials=len(results),raw_steps=sum(r['n_steps'] for r in results),
                 candidates=candidates,scenarios=scenarios,evaluations=evaluations,
                 winner=approved_winner['candidate'] if approved_winner else None,
                 recommended_settings=(settings_by_label[approved_winner['candidate']]
                                       if approved_winner else None),
                 best_diagnostic_if_no_winner=diagnostic['candidate'] if diagnostic and not approved_winner else None,
                 winner_audits=audits,
                 ranking_rule='ALL 29 PASS first, then 0.15*XY + 0.25*Yaw + 0.20*Z + 0.40*combined mean settle after 1s hover',
                 thresholds={'xy_m':TUNE_XY_OVERSHOOT_EPS_M,
                             'yaw_deg':TUNE_YAW_OVERSHOOT_EPS_DEG,'alt_m':TUNE_Z_OVERSHOOT_EPS_M},
                 caveat='Deterministic simulated 240Hz tests only. Pass threshold != mathematically zero overshoot. Original controller unchanged.')
        cfgpath.write_text(json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        lines=['CF2X 10 candidate parameter combinations, 240Hz',
               f'Complete={complete}; scenarios={len(scenarios)}; trials={len(results)}',
               'XY/Yaw/Z previous best are only standalone-validated.',
               'No changes to original roll/pitch/yaw/altitude controller algorithms.',
               'Only ALL scenarios PASS candidates can win. Thresholds are nonzero measurement tolerances.',
               'Score: 15% XY + 25% Yaw + 20% altitude + 40% combined mean settle time (after 1s hover).',
               f'Winner={approved_winner["candidate"] if approved_winner else "NONE"}',
               f'Closest diagnostic (NOT APPROVED)={diagnostic["candidate"] if diagnostic and not approved_winner else "not needed"}',
               '', 'Candidate table:']
        for c in evaluations:
            lines.append(f'{c["candidate"]}: pass={c["pass_count"]}/{c["trial_count"]}, '
                         f'score={c["weighted_mean_time_s"]}, '
                         f'maxYawOver={c["max_yaw_overshoot_deg"]:.6g}deg, '
                         f'maxXYOver={c["max_xy_overshoot_m"]:.6g}m, '
                         f'maxZOver={c["max_alt_overshoot_m"]:.6g}m, '
                         f'failures={c["status_counts"]}')
        lines+=['','Winner audit: '+str(audits),
                'Warning: no accepted winner means NOT enough evidence to choose safe combined settings.',
                'This sweep does not yet fix the Euler-yaw/body-rate coupling from prior logs.',
                'A deterministic reset does not test disturbances, noise, or external hardware.']
        txtpath.write_text('\n'.join(lines)+'\n',encoding='utf8')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            labels=[r['candidate'] for r in evaluations]
            fig,axes=plt.subplots(3,1,figsize=(13,11),constrained_layout=True)
            axes[0].bar(labels,[r['pass_count'] for r in evaluations]);axes[0].axhline(len(scenarios),ls=':')
            axes[0].set(title='PASS count (all 29 required)',ylabel='PASS trials')
            axes[1].bar(labels,[r['max_yaw_overshoot_deg'] for r in evaluations]);axes[1].axhline(TUNE_YAW_OVERSHOOT_EPS_DEG,ls=':')
            axes[1].set(title='Maximum Yaw target crossing (lower is better)',ylabel='Degrees')
            axes[2].bar(labels,[r['weighted_mean_time_s'] if r['weighted_mean_time_s'] is not None else 0 for r in evaluations])
            axes[2].set(title='Weighted mean time (shown only for all-PASS candidates)',ylabel='Seconds')
            for ax in axes:
                ax.grid(axis='y',alpha=.25)
                ax.tick_params(axis='x',rotation=20)
            fig.savefig(plotpath,dpi=125);plt.close(fig)
        except Exception as ex:
            print('[COMBO PLOT] skipped: ',repr(ex),flush=True)
        with ZipFile(zippath,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as out:
            for path in (rawpath,trialpath,candidatepath,txtpath,cfgpath,plotpath):
                if path.exists():out.write(path,arcname=path.name)
        print(f'[COMBO END] Winner={approved_winner["candidate"] if approved_winner else "NONE"}; '
              f'closest={diagnostic["candidate"] if diagnostic else "NONE"}; archive={zippath}',flush=True)
    if error is not None:raise RuntimeError('Combo sweep incomplete; inspect partial logs') from error
    return cfg


# =============================================================================
# 11. C08 Yaw Euler/body-Z coupling A/B: 단일 파일 독립 실행
# =============================================================================
# 원형 제어기 vs 좌표계 보상 제어기. 동일한 C08, 초기 조건, 모든 케이스를 유지한다.
# 4개 기존 실패 조건 집중 재시험 -> 29개 회귀 검증 -> 조건부 재감사(29개).
# 정확한 무오버슈트 보장은 아니며, 유한한 240Hz 표본/허용값으로 PASS 판정.
COUPLING_GUI = False
COUPLING_START_ALT_M = 150.0
COUPLING_RETEST_WINNER = True
COUPLING_FOCUS_NAMES = (
    'joint_nw_p90', 'joint_se_m90',
    'stress_long_p90_zup', 'stress_long_m90_zdown',
)
COUPLING_EXTRA_COLUMNS = [
    'yaw_variant','yaw_euler_rate_deg_s','yaw_body_r_deg_s',
    'yaw_coupling_rate_deg_s','yaw_r_ref_next_deg_s',
    'yaw_psi_ref_next_deg_s','yaw_torque_next_Nm',
]


def _coupling_failure_comparison(scenarios, results):
    """시나리오 단위 A/B 비교. 완료하지 못한 시험은 별도 표시."""
    lookup={(r['stage'],r['parameter'],r['scenario']):r for r in results}
    table=[]
    for scenario in scenarios:
        for stage in ('focus','full'):
            old=lookup.get((stage,'legacy',scenario['name']))
            new=lookup.get((stage,'compensated',scenario['name']))
            if old is None and new is None:continue
            def read(r,key):return r[key] if r is not None else None
            delta=(new['max_yaw_overshoot_deg']-old['max_yaw_overshoot_deg']
                   if new is not None and old is not None else None)
            table.append(dict(stage=stage,scenario=scenario['name'],kind=scenario['kind'],
                              legacy_status=read(old,'status'),compensated_status=read(new,'status'),
                              legacy_yaw_overshoot_deg=read(old,'max_yaw_overshoot_deg'),
                              compensated_yaw_overshoot_deg=read(new,'max_yaw_overshoot_deg'),
                              change_yaw_overshoot_deg=delta,
                              legacy_xy_overshoot_m=read(old,'max_goal_overshoot_m'),
                              compensated_xy_overshoot_m=read(new,'max_goal_overshoot_m'),
                              legacy_alt_overshoot_m=read(old,'max_alt_overshoot_m'),
                              compensated_alt_overshoot_m=read(new,'max_alt_overshoot_m'),
                              legacy_settle_s=read(old,'settle_time_s'),
                              compensated_settle_s=read(new,'settle_time_s'),
                              legacy_motor_saturation_steps=read(old,'motor_saturation_steps'),
                              compensated_motor_saturation_steps=read(new,'motor_saturation_steps')))
    return table


def run_yaw_coupling_ab():
    """Single invocation, C08 vs compensated, all logs and an upload-ready ZIP."""
    import json
    from zipfile import ZipFile,ZIP_DEFLATED
    from collections import Counter
    root=LOG_DIR
    root.mkdir(parents=True,exist_ok=True)
    prefix='drone_motion_coupling'
    paths={ext:root/f'{prefix}{ext}' for ext in (
        '.csv','_trials.csv','_comparison.csv','_config.json','.txt','.png','_bundle.zip')}
    scenarios=_combo_scenarios()
    focus=[s for s in scenarios if s['name'] in COUPLING_FOCUS_NAMES]
    assert len(scenarios)==29 and len(focus)==4
    c08=next(x for x in _combo_build_candidates() if x['label']=='C08_conservative')
    settings=dict(c08['settings'])
    # C08 parameters are exact copies from the previously measured combo sweep.
    backup={key:globals()[key] for key in settings}
    previous={'GUI':GUI,'REAL_TIME_PLAYBACK':REAL_TIME_PLAYBACK,
              'TUNE_START_ALT_M':TUNE_START_ALT_M}
    env=None;error=None;rows=[];counter=[0];runs={};audits={};winner=None
    start=time.perf_counter()
    print('[COUPLING AB] C08 legacy vs compensated, 4 focused + 29 full cases per variant',flush=True)
    try:
        for key,value in settings.items():globals()[key]=value
        globals()['GUI']=COUPLING_GUI
        globals()['REAL_TIME_PLAYBACK']=False
        globals()['TUNE_START_ALT_M']=COUPLING_START_ALT_M
        env=CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
                       initial_xyzs=np.array([[0.,0.,COUPLING_START_ALT_M]]),
                       initial_rpys=np.zeros((1,3)),pyb_freq=SIM_FREQ,
                       ctrl_freq=SIM_FREQ,gui=COUPLING_GUI,user_debug_gui=False)
        with paths['.csv'].open('w',newline='',encoding='utf-8-sig') as rf, \
             paths['_trials.csv'].open('w',newline='',encoding='utf-8-sig') as tf:
            rw=csv.DictWriter(rf,fieldnames=_TUNE_RAW_COLUMNS+COUPLING_EXTRA_COLUMNS)
            tw=csv.DictWriter(tf,fieldnames=_TUNE_TRIAL_COLUMNS)
            rw.writeheader();tw.writeheader()
            for variant in ('legacy','compensated'):
                for stage,selected in (('focus',focus),('full',scenarios)):
                    group=_tune_run_group(env,selected,rw,tw,counter,stage,variant,
                                          0.0,0,yaw_mode=variant)
                    rows.extend(group)
                    summary=_combo_evaluate(variant,group) if stage=='full' else None
                    runs[(variant,stage)]=summary if summary is not None else {
                        'pass_count':sum(r['status']=='PASS' for r in group),
                        'trial_count':len(group),
                        'max_yaw_overshoot_deg':max(r['max_yaw_overshoot_deg'] for r in group)}
                    print(f'[AB {variant:11s} {stage:5s}] '
                          f"PASS={runs[(variant,stage)]['pass_count']}/{len(group)} "
                          f"max yaw over={runs[(variant,stage)]['max_yaw_overshoot_deg']:.6f}deg",
                          flush=True)
                    rf.flush();tf.flush()
            # Qualification requires ALL 29 PASS and ALL 4 focused tests PASS.
            eligible=[variant for variant in ('legacy','compensated')
                      if runs[(variant,'full')]['eligible']
                      and runs[(variant,'focus')]['pass_count']==len(focus)]
            if eligible:
                winner=min(eligible,key=lambda v:(runs[(v,'full')]['weighted_mean_time_s'],v))
                if COUPLING_RETEST_WINNER:
                    group=_tune_run_group(env,scenarios,rw,tw,counter,'audit',winner,
                                          0.0,1,yaw_mode=winner)
                    rows.extend(group)
                    audits[winner]=_combo_evaluate(winner,group)
                    print(f"[AUDIT {winner}] PASS={audits[winner]['pass_count']}/29",flush=True)
                    if not audits[winner]['eligible']:
                        winner=None
                rf.flush();tf.flush()
            else:
                winner=None
    except Exception as exc:
        error=exc
        print('[COUPLING ERROR] Partial CSV kept:',repr(exc),flush=True)
    finally:
        if env is not None:env.close()
        for key,value in backup.items():globals()[key]=value
        for key,value in previous.items():globals()[key]=value
        complete=(error is None and len([r for r in rows if r['stage']=='full'])==58
                  and len([r for r in rows if r['stage']=='focus'])==8)
        fullstats={variant:runs.get((variant,'full')) for variant in ('legacy','compensated')}
        focusstats={variant:runs.get((variant,'focus')) for variant in ('legacy','compensated')}
        # Do not label an incomplete experiment approved.
        approved_winner=winner if complete and winner else None
        comparisons=_coupling_failure_comparison(scenarios,rows)
        with paths['_comparison.csv'].open('w',newline='',encoding='utf-8-sig') as cf:
            fields=list(comparisons[0]) if comparisons else [
                'stage','scenario','kind','legacy_status','compensated_status']
            writer=csv.DictWriter(cf,fieldnames=fields)
            writer.writeheader();writer.writerows(comparisons)
        config=dict(mode='yaw_coupling_ab',model='CF2X',complete=complete,
                    sim_hz=SIM_FREQ,initial_altitude_m=COUPLING_START_ALT_M,
                    setting_source='C08_conservative from 10-candidate combo test',
                    fixed_settings=settings,focus_names=COUPLING_FOCUS_NAMES,
                    scenarios=scenarios,variants=['legacy','compensated'],
                    trial_count=len(rows),raw_rows=sum(r['n_steps'] for r in rows),
                    full_evaluation=fullstats,focus_evaluation=focusstats,
                    winner_audit=audits,winner=approved_winner,
                    recommended_settings=(settings if approved_winner else None),
                    winning_controller_mode=approved_winner,
                    overshoot_tolerances=dict(xy_m=TUNE_XY_OVERSHOOT_EPS_M,
                        yaw_deg=TUNE_YAW_OVERSHOOT_EPS_DEG,alt_m=TUNE_Z_OVERSHOOT_EPS_M),
                    settling_yaw_rate_basis='Euler yaw rate for BOTH variants',
                    caveats=['Sampled 240 Hz thresholds do not prove exact zero overshoot.',
                             'Deterministic simulator comparison, not hardware validation.',
                             'A/B changes yaw rate feedback and body-r reference only.'])
        paths['_config.json'].write_text(json.dumps(config,indent=2,ensure_ascii=False,
                                          allow_nan=False)+'\n',encoding='utf-8')
        lines=[
            'CF2X C08 Yaw Euler/body-rate coupling A/B, 240Hz',
            f'Complete={complete}; trials={len(rows)}; raw rows={sum(r["n_steps"] for r in rows)}',
            f'Fixed C08 parameters={settings}',
            f'Focus cases={list(COUPLING_FOCUS_NAMES)}',
            'Legacy: original yaw_predictive_control(angle, BODY r).',
            'Compensated: yaw predictive trajectory uses true Euler psi_dot; body r_ref=(psi_ref*cos(theta)-q*sin(phi))/cos(phi).',
            'Both variants settled only if Euler psi_dot (not body r) within 0.5deg/s.',
            'PASS: position/rate settle + XY overshoot <=1mm, Yaw overshoot <=0.025deg, altitude <=5mm, no motor clipping.',
            f'Full 29 stats={fullstats}',f'Focus 4 stats={focusstats}',
            f'Audits={audits}',f'Approved winner={approved_winner or "NONE"}',
            'No winner -> neither variant satisfied full criteria. Do NOT claim full combined validation.',
            'Crucial: log RPMs describe applied motor commands at this time; *_next fields are commands for NEXT step.',
            'A/B uses the same initial resets and scenarios; no disturbances or noise added.',
            '', 'Focused cases side-by-side:']
        for line in comparisons:
            if line['stage']=='focus':
                lines.append(f"  {line['scenario']}: {line['legacy_status']} -> {line['compensated_status']}, "
                             f"Yaw overshoot {line['legacy_yaw_overshoot_deg']} -> "
                             f"{line['compensated_yaw_overshoot_deg']} deg")
        paths['.txt'].write_text('\n'.join(lines)+'\n',encoding='utf8')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
            names=list(COUPLING_FOCUS_NAMES)
            x=np.arange(len(names));w=0.36
            for offset,v in ((-w/2,'legacy'),(w/2,'compensated')):
                d={r['scenario']:r for r in rows if r['stage']=='focus' and r['parameter']==v}
                axes[0].bar(x+offset,[d[n]['max_yaw_overshoot_deg'] if n in d else 0 for n in names],w,label=v)
            axes[0].axhline(TUNE_YAW_OVERSHOOT_EPS_DEG,color='black',ls='--',lw=1)
            axes[0].set(xticks=x,xticklabels=names,ylabel='Yaw overshoot [deg]',title='4 previously failing combined scenarios')
            axes[0].legend();axes[0].tick_params(axis='x',rotation=12)
            labels=['legacy','compensated']
            axes[1].bar(labels,[fullstats[v]['pass_count'] if fullstats[v] else 0 for v in labels])
            axes[1].axhline(29,color='black',ls='--',lw=1)
            axes[1].set(ylabel='PASS/29',title='Full regression test (all 29 required)')
            axes[2].bar(labels,[fullstats[v]['max_yaw_overshoot_deg'] if fullstats[v] else 0 for v in labels])
            axes[2].axhline(TUNE_YAW_OVERSHOOT_EPS_DEG,color='black',ls='--',lw=1)
            axes[2].set(ylabel='Worst Yaw overshoot [deg]',title='Full 29 cases')
            for a in axes:a.grid(axis='y',alpha=.25)
            fig.savefig(paths['.png'],dpi=120);plt.close(fig)
        except Exception as exc:
            print('Chart skipped:',repr(exc),flush=True)
        with ZipFile(paths['_bundle.zip'],'w',compression=ZIP_DEFLATED,
                     compresslevel=3,allowZip64=True) as zf:
            for ending in ('.csv','_trials.csv','_comparison.csv','_config.json','.txt','.png'):
                f=paths[ending]
                if f.exists():zf.write(f,arcname=f.name)
        print(f'[COUPLING AB COMPLETE] full={complete} winner={approved_winner}, '
              f"zip={paths['_bundle.zip']}",flush=True)
    if error is not None:
        raise RuntimeError('Yaw coupling A/B incomplete. Inspect saved bundle.') from error
    return config

if __name__=='__main__':
    main()
