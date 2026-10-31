"""CF2X 고정 목표점 거리 기반 고속 접근 + 예측 제동 (240 Hz).

목적: 계획된 시간별 궤적 없이 (0,0,100)m -> (10,10,100)m 이동 및 정지.
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
MAX_APPROACH_SPEED_M_S = 2.5   # [m/s] 접근 목표 속도 상한
APPROACH_ACCEL_M_S2 = 0.90    # [m/s²] 진행 방향 가속
BRAKE_ACCEL_M_S2 = 0.80       # [m/s²] 진행 방향 제동 계획 및 명령 상한
MAX_XY_ACCEL = 1.0           # [m/s²] 수평 가속도 명령 벡터 크기 제한
MAX_XY_TILT_DEG = 12.0       # [deg] 목표 기울기 한계
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
    """[각도 오차 -> 허용 각속도 -> 목표 각가속도 -> Yaw 토크].

    목표에 가까워질수록 v_ref = sqrt(2*a_brake*|e|)가 작아진다.
    목표에 접근하는 실제 각속도가 v_ref보다 크면 반대 토크로 감속한다.
    미검증 제어기: 회전 관성/토크 여유와 선형 근사에 기반한다.
    """
    error = float(np.arctan2(np.sin(target_yaw_rad-yaw_rad),
                             np.cos(target_yaw_rad-yaw_rad)))
    max_acc = np.deg2rad(MAX_YAW_ACCEL_DEG_S2)
    # 실제 가용 토크가 줄면 제동계획도 보수적으로 축소한다.
    physically_available_braking = min(torque_positive_limit, torque_negative_limit) / yaw_inertia
    braking = max(1e-8, min(np.deg2rad(YAW_BRAKE_ACCEL_DEG_S2),
                           physically_available_braking))
    rate_limit = np.deg2rad(MAX_YAW_RATE_DEG_S)
    target_rate = float(np.sign(error) * min(rate_limit,
                             np.sqrt(2.0 * braking * abs(error))))
    # 작은 각도 / 낮은 속도에서는 토크를 0으로 유지하며 불필요한 떨림 감소
    arrived = (abs(error) <= np.deg2rad(YAW_ARRIVAL_ANGLE_DEG)
               and abs(yaw_rate_rad_s) <= np.deg2rad(YAW_ARRIVAL_RATE_DEG_S))
    if arrived:
        target_rate = 0.0
    raw_accel = YAW_RATE_TRACK_K * (target_rate - yaw_rate_rad_s)
    accel_cmd = float(np.clip(raw_accel, -max_acc, max_acc))
    requested_torque = yaw_inertia * accel_cmd
    torque = float(np.clip(requested_torque,
                           -torque_negative_limit, torque_positive_limit))
    actual_accel_cmd = torque / yaw_inertia
    # 과거 CSV 형식 호환: 앞 두 열은 각각 프로파일 구동/각속도 제동 토크 성분
    drive_torque = yaw_inertia * YAW_RATE_TRACK_K * target_rate
    damping_torque = -yaw_inertia * YAW_RATE_TRACK_K * yaw_rate_rad_s
    mode = (0 if arrived else
            2 if (abs(target_rate) < abs(yaw_rate_rad_s) and target_rate*yaw_rate_rad_s >= 0)
            else 1)  # 0=정착, 1=속도 추종, 2=제동
    return (error, drive_torque, damping_torque, torque,
            target_rate, actual_accel_cmd, braking, mode,
            torque_positive_limit, torque_negative_limit)


def altitude_pd(z_m, vz_m_s, target_z_m, mass_kg, gravity_m_s2,
                roll_rad, pitch_rad):
    """고도 오차/수직속도로 네 모터의 총추력[N]을 계산한다.

    m*z_ddot = T*cos(roll)*cos(pitch) - m*g (단순 무항력 모델)
    고도 PD는 토크가 아닌 가속도를 계산한 후 질량으로 힘을 환산한다.
    """
    error = target_z_m - z_m
    p_acc = KP_ALT * error
    d_acc = -KD_ALT * vz_m_s
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


if __name__ == '__main__':
    main()
