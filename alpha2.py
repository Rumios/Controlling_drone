"""CF2X 3대 Leader–Follower 군집 실험 (단일 파일).
호버링 -> 대형 형성 -> 동시 출발 -> 경유점 선회 -> 군집 평가
"""

import csv
import json
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# ================== 설정 및 제어 이득 ==================

SIM_FREQ = 240
DURATION = 20.0
NUM_DRONES = 3
LEADER_ID = 0
INITIAL_XYZ_M = (0.0, 0.0, 150.0)
INITIAL_RPY_DEG = (0.0, 0.0, 0.0)
LEADER_TARGET = (10.0, 10.0, 150.0)
TARGET_YAW_DEG = 0.0
MIN_DRONE_DISTANCE = 0.50  # [m] monitoring threshold; no collision avoidance
FORMATION_BACK_M = 0.75  # leader -> followers: distance behind along travel direction
FORMATION_SIDE_M = 0.40  # left/right spacing from centerline
INITIAL_SIDE_M = 1.20  # initial follower positions on a line (not yet triangle)
FORMATION_ERROR_TOL_M = 0.10
FORMATION_SPEED_TOL_M_S = 0.12
HOVER_POSITION_TOL_M = 0.10
HOVER_SPEED_TOL_M_S = 0.10
HOVER_ANGLE_TOL_DEG = 2.0
HOVER_RATE_TOL_DEG_S = 5.0
# 대형 형성 완료 여부는 시간 대신 위치·속도·간격 오차로만 판정한다.

PHASE_HOVER, PHASE_FORM, PHASE_MOVE = 0, 1, 2
GUI = True
REAL_TIME_PLAYBACK = True
CAMERA_MODE = 'follow'  # 'free' or 'follow'
CAMERA_DISTANCE = 8.0  # 가까운 거리에서 세 드론의 대형 관찰
CAMERA_YAW = 45.0
CAMERA_PITCH = -55.0  # 대형을 위에서 보기 쉬운 각도
CAMERA_UPDATE_HZ = 30  # GUI 업데이트 부하 감소 (제어는 240Hz 유지)
CAMERA_FOLLOW_TAU_S = 0.35  # 카메라 중심의 지수 평활화 시정수
DRAW_TRAIL = True
TRAIL_HZ = 30
CONSOLE_HZ = 1
SAVE_PLOT = True

# ----- Original C08 parameters (kept unchanged) -----
KP_XY, KD_XY = 0.8, 1.8
MAX_XY_ACCEL = 3.0
APPROACH_ACCEL_M_S2 = 2.7
BRAKE_ACCEL_M_S2 = 0.9
MAX_APPROACH_SPEED_M_S = 3.4
MAX_XY_TILT_DEG = 25.0
SPEED_TRACK_GAIN = 2.0
LATERAL_DAMP_GAIN = 2.0
BRAKE_RESPONSE_S = 0.18
BRAKE_DISTANCE_MARGIN_M = 0.12
TERMINAL_ENTER_DIST_M = 0.40
TERMINAL_EXIT_DIST_M = 0.65
XY_POSITION_TOL_M = 0.10
XY_SPEED_TOL_M_S = 0.10
ARRIVAL_STABLE_S = 0.50
SPEED_SCALE = 1.0
NAV_HOVER, NAV_APPROACH, NAV_TERMINAL, NAV_HOLD = 0, 1, 2, 3
KP_ROLL, KD_ROLL = 0.00040, 0.000095
KP_PITCH, KD_PITCH = 0.00040, 0.000095
GAIN_SCALE = 1.0
MAX_YAW_RATE_DEG_S = 50.0
MAX_YAW_ACCEL_DEG_S2 = 120.0
YAW_BRAKE_ACCEL_DEG_S2 = 40.0
YAW_RATE_TRACK_K = 8.0
YAW_TORQUE_RESERVE = 0.75
YAW_ARRIVAL_ANGLE_DEG = 0.15
YAW_ARRIVAL_RATE_DEG_S = 0.25
YAW_USE_STOP_GUARD = True
YAW_STOP_LEAD_S = 0.18
YAW_STOP_MARGIN_DEG = 0.50
YAW_TERMINAL_RATE_GAIN = 1.4
KP_ALT, KD_ALT = 4.0, 2.5
ALT_LOOKAHEAD_S = 0.275
MAX_VERTICAL_ACCEL = 7.5
ENABLE_TILT_COMPENSATION = True
MIN_TILT_COS = 0.5
ABORT_TILT_DEG = 35.0
ABORT_RATE_DEG_S = 900.0
ABORT_MIN_Z_M = 60.0

# 두 경유점을 연속 통과해 이동하면서 실제 방향 전환을 검증한다.
# 첫 경유점은 PTP 종료점이 아니라 방향 전환 지점이다.
WAYPOINTS = [
    np.array([10.0, 10.0, 150.0]),
    np.array([20.0, 10.0, 150.0]),
]
WAYPOINT_SWITCH_RADIUS_M = 1.5

# 선회 속도/가속도 제한. Follower별 상이한 목표 속도/가속도를 생성한다.
FORMATION_MAX_YAW_RATE_RAD_S = np.deg2rad(22.0)  # 대형 회전각속도 (기체 Yaw 아님)
FORMATION_MAX_YAW_ACCEL_RAD_S2 = np.deg2rad(35.0)
FORMATION_SLOW_ERROR_M = 0.10
FORMATION_STOP_ERROR_M = 0.30
LEADER_MIN_SPEED_SCALE = 0.15
LEADER_SPEED_SCALE_RATE_S = 0.8  # 급격한 지령 변화 억제
DISTANCE_SLOW_MARGIN_M = 0.18

# 포메이션 좌표계: 앞(+forward), 왼쪽(+left)
START_XYZ = np.array(INITIAL_XYZ_M, dtype=float)
FIRST_FORWARD = WAYPOINTS[0][:2] - START_XYZ[:2]
FIRST_FORWARD /= np.linalg.norm(FIRST_FORWARD)
FIRST_LEFT = np.array([-FIRST_FORWARD[1], FIRST_FORWARD[0]])

def initial_slot(back_m, side_m):
    return np.r_[-back_m * FIRST_FORWARD + side_m * FIRST_LEFT, 0.0]

INITIAL_XYZS = np.array([
    START_XYZ,
    START_XYZ + initial_slot(0, INITIAL_SIDE_M),
    START_XYZ + initial_slot(0, -INITIAL_SIDE_M),
])

# PROJECT_ROOT를 단일 진입점 기준으로 유지하기 위해 상위 실행 파일 위치 이용
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == 'examples':
    PROJECT_ROOT = PROJECT_ROOT.parent
LOG_DIR = PROJECT_ROOT / 'logs'


# ================== C08 저수준 제어 ==================

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


def xy_position_pd(pos_xy, velocity_xy, target_xy):
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


def point_to_point_guidance(pos_xy, velocity_xy, target_xy, speed_scale=SPEED_SCALE):
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
    delta = np.asarray(target_xy, dtype=float) - p_xy
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
    """항법 상태는 거리·속도 오차로 전환하고 종단 정착은 별도 확인한다.

    FOLLOW(접근)/TERMINAL 전환 및 홀드/외란 복귀는 모두 거리와 속도가 기준.
    목표점을 지나친 경우라도 방향이 재계산되며, 터미널 PD가 복구한다.
    """
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


def point_to_point_controller(pos_xy, velocity_xy, state, target_xy, start_xy, speed_scale=SPEED_SCALE):
    """상태별 XY 목표 가속도. 기존 XY 위치 PD는 초기/종단 정착에 재사용."""
    p_xy = np.asarray(pos_xy, dtype=float)
    v_xy = np.asarray(velocity_xy, dtype=float)
    if state == NAV_HOVER:
        _, acc = xy_position_pd(p_xy, v_xy, np.asarray(start_xy, dtype=float))
    elif state in (NAV_TERMINAL, NAV_HOLD):
        _, acc = xy_position_pd(p_xy, v_xy, np.asarray(target_xy, dtype=float))
    else:
        acc, _ = point_to_point_guidance(p_xy, v_xy, target_xy, speed_scale)
    # Diagnostics always reflect current state, including terminal/hold.
    _, diag = point_to_point_guidance(p_xy, v_xy, target_xy, speed_scale)
    diag["acceleration_norm"] = float(np.linalg.norm(acc))
    if state != NAV_APPROACH:
        # 대기/최종 정착에는 진행 목표 속도를 주지 않는다.
        diag["target_speed"] = 0.0
        selected_goal = np.asarray(start_xy, dtype=float) if state == NAV_HOVER else np.asarray(target_xy, dtype=float)
        raw_pd_acc = KP_XY * (selected_goal - p_xy) - KD_XY * v_xy
        diag["limited"] = float(np.linalg.norm(raw_pd_acc)) > MAX_XY_ACCEL
        diag["radial_acc"] = float(np.dot(acc, (np.asarray(target_xy, dtype=float)-p_xy) /
                                          max(1e-9, np.linalg.norm(np.asarray(target_xy, dtype=float)-p_xy))))
        diag["lateral_acc"] = float(np.linalg.norm(acc - diag["radial_acc"] *
                                    (np.asarray(target_xy, dtype=float)-p_xy) / max(1e-9, np.linalg.norm(np.asarray(target_xy, dtype=float)-p_xy))))
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


def compute_drone_rpm(state, target, origin, nav, nav_stable_s, t, dt, params,
                      reference_velocity_xy=None, reference_accel_xy=None, speed_scale=SPEED_SCALE):
    """One drone's C08 controller, no shared controller states between drones."""
    pos, vel, rpy, body_rates = measure_state(state)
    rpy_deg = np.rad2deg(rpy)
    if (not np.all(np.isfinite(np.r_[pos, vel, rpy, body_rates]))
            or abs(rpy_deg[0]) > ABORT_TILT_DEG
            or abs(rpy_deg[1]) > ABORT_TILT_DEG
            or np.max(np.abs(np.rad2deg(body_rates))) > ABORT_RATE_DEG_S
            or pos[2] < ABORT_MIN_Z_M):
        raise RuntimeError(f'Unsafe state at t={t:.3f}s: pos={pos}, RPY={rpy_deg}')

    # 움직이는 Leader 기준의 상대속도로 Follower의 PTP 오차를 계산한다.
    rel_velocity_xy = vel[:2].copy()
    if reference_velocity_xy is not None:
        rel_velocity_xy -= np.asarray(reference_velocity_xy, dtype=float)
    distance_xy = float(np.linalg.norm(target[:2] - pos[:2]))
    speed_xy = float(np.linalg.norm(rel_velocity_xy))
    nav, nav_stable_s = update_navigation_state(
        nav, t, distance_xy, speed_xy, nav_stable_s, dt
    )
    acc_xy, _ = point_to_point_controller(
        pos[:2], rel_velocity_xy, nav, target[:2], origin[:2], speed_scale
    )
    if reference_accel_xy is not None:
        # Leader와 동시에 출발해 속도를 맞출 수 있도록 상위 수준 가속도 전달.
        # 최종 상한은 기존 C08의 MAX_XY_ACCEL로 유지한다.
        acc_xy += np.asarray(reference_accel_xy, dtype=float)
        a_norm = float(np.linalg.norm(acc_xy))
        if a_norm > MAX_XY_ACCEL:
            acc_xy *= MAX_XY_ACCEL / a_norm
    alt = altitude_pd(pos[2], vel[2], float(target[2]),
                      params['mass'], params['g'], rpy[0], rpy[1])
    roll_ref, pitch_ref, acc_xy = xy_acceleration_to_attitude(acc_xy, rpy[2], alt[3], params['g'])
    roll = roll_pd(rpy[0], body_rates[0], roll_ref)
    pitch = pitch_pd(rpy[1], body_rates[1], pitch_ref)
    pos_lim, neg_lim = yaw_torque_limits(
        alt[4], roll[3], pitch[3], params['arm'],
        params['kf'], params['km'], params['max_f']
    )
    yaw, _ = yaw_predictive_control_coupled(
        rpy[2], rpy, body_rates, np.deg2rad(TARGET_YAW_DEG),
        params['izz'], pos_lim, neg_lim
    )
    rpm, saturated = allocate_motor_rpm(
        roll[3], pitch[3], yaw[3], alt[4],
        params['arm'], params['kf'], params['km'], params['max_f']
    )
    if not np.all(np.isfinite(rpm)):
        raise RuntimeError(f'Nonfinite RPM at t={t:.3f}s')
    return rpm.reshape(4), saturated, nav, nav_stable_s, acc_xy



# ================== Leader 경로·속도·회전 제어 ==================

def wrap_angle(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))

@dataclass
class LeaderCommand:
    goal: np.ndarray
    heading: float
    omega: float
    alpha: float
    speed_scale: float
    waypoint_index: int

class LeaderPlanner:
    def __init__(self):
        self.index = 0
        self.heading = float(np.arctan2(FIRST_FORWARD[1], FIRST_FORWARD[0]))
        self.omega = 0.0
        self.speed_scale = 1.0
        self.radius = float(np.hypot(FORMATION_BACK_M, FORMATION_SIDE_M))

    def update(self, leader_pos, formation_errors, min_distance, dt):
        # 마지막 목표는 도착 오차로 판단하며 이전 경유점은 통과 반경으로 전환.
        while self.index < len(WAYPOINTS)-1 and (
            np.linalg.norm(WAYPOINTS[self.index][:2] - leader_pos[:2]) <= WAYPOINT_SWITCH_RADIUS_M
        ):
            self.index += 1
            print(f'[ROUTE] waypoint switched to {self.index+1}/{len(WAYPOINTS)}', flush=True)
        goal = WAYPOINTS[self.index]
        direction = goal[:2] - leader_pos[:2]
        desired = self.heading if np.linalg.norm(direction) < 1e-6 else float(np.arctan2(direction[1], direction[0]))
        # 대형 회전은 개별 드론 Yaw와 분리. 큰 방향 점프를 각가속도/각속도로 제한.
        err = wrap_angle(desired - self.heading)
        max_omega = min(FORMATION_MAX_YAW_RATE_RAD_S,
                        np.sqrt(MAX_XY_ACCEL / max(self.radius, 1e-6)) * 0.5)
        # 정지거리 기반으로 가속과 감속을 계획해 목표 각도 오버슈트를 줄인다.
        desired_omega = np.sign(err) * min(max_omega,
                             np.sqrt(2*FORMATION_MAX_YAW_ACCEL_RAD_S2*abs(err)))
        previous = self.omega
        self.omega += float(np.clip(desired_omega-self.omega,
                        -FORMATION_MAX_YAW_ACCEL_RAD_S2*dt, FORMATION_MAX_YAW_ACCEL_RAD_S2*dt))
        # 마지막 작은 수치 오차가 생겨도 목표 각도를 넘어가지 않게 제한한다.
        delta = float(np.clip(self.omega*dt, -abs(err), abs(err))) if abs(err)>1e-9 else 0.0
        if delta*err < 0: delta=0.0
        self.heading = wrap_angle(self.heading + delta)
        if abs(err)<1e-6: self.omega=0.0
        alpha = (self.omega-previous)/dt

        # 대형 오차 / 실제 근접 거리 / 대형 선회에 필요한 가속도 고려해 Leader 속도를 낮춘다.
        e = max(formation_errors.values())
        if e <= FORMATION_SLOW_ERROR_M:
            error_scale = 1.0
        else:
            error_scale = np.interp(e, [FORMATION_SLOW_ERROR_M, FORMATION_STOP_ERROR_M],
                                    [1.0, LEADER_MIN_SPEED_SCALE])
        distance_scale = np.interp(min_distance,
                                   [MIN_DRONE_DISTANCE, MIN_DRONE_DISTANCE+DISTANCE_SLOW_MARGIN_M],
                                   [LEADER_MIN_SPEED_SCALE, 1.0])
        # 자전 운동학만 고려한 보수적 감속; 장애물 회피/충돌 방지를 보장하지 않음.
        rotational_accel = abs(alpha)*self.radius + self.omega**2*self.radius
        rotation_scale = np.clip((MAX_XY_ACCEL-rotational_accel)/MAX_XY_ACCEL,
                                  LEADER_MIN_SPEED_SCALE, 1.0)
        requested = float(np.clip(min(error_scale, distance_scale, rotation_scale),
                                  LEADER_MIN_SPEED_SCALE, 1.0))
        self.speed_scale += float(np.clip(requested-self.speed_scale,
                        -LEADER_SPEED_SCALE_RATE_S*dt, LEADER_SPEED_SCALE_RATE_S*dt))
        return LeaderCommand(goal.copy(), self.heading, self.omega, alpha,
                             self.speed_scale, self.index)


# ================== Follower 대형·회전 운동학 ==================

FOLLOWER_SLOTS = {1: (-FORMATION_BACK_M, FORMATION_SIDE_M),
                  2: (-FORMATION_BACK_M, -FORMATION_SIDE_M)}
J = np.array([[0.0, -1.0], [1.0, 0.0]])

def relative_xy(forward, drone_id):
    back, side = FOLLOWER_SLOTS[drone_id]
    left = J @ np.asarray(forward, dtype=float)
    return back * np.asarray(forward, dtype=float) + side * left

def targets_and_references(positions, leader_velocity, leader_accel,
                           heading, omega, alpha, phase, leader_goal):
    """(목표 위치[3,3], 목표 수평속도[3,2], 목표 수평가속도[3,2])."""
    targets = INITIAL_XYZS.copy()
    target_vel = np.zeros((NUM_DRONES, 2))
    target_acc = np.zeros((NUM_DRONES, 2))
    if phase == 0:
        return targets, target_vel, target_acc
    leader_position = positions[LEADER_ID]
    targets[LEADER_ID] = INITIAL_XYZS[LEADER_ID] if phase == 1 else leader_goal
    forward = np.array([np.cos(heading), np.sin(heading)])
    # 대형 형성 단계에는 Leader 고정 + 첫 진행 방향 적용.
    for drone_id in FOLLOWER_SLOTS:
        r = relative_xy(forward, drone_id)
        targets[drone_id] = leader_position + np.r_[r, 0.0]
        if phase == 2:
            # 이동하면서 회전하는 강체 대형의 운동학
            # v_i = v_L + omega * J*r
            # a_i = a_L + alpha * J*r - omega^2*r
            target_vel[drone_id] = leader_velocity[:2] + omega * (J @ r)
            target_acc[drone_id] = leader_accel + alpha * (J @ r) - omega**2 * r
    return targets, target_vel, target_acc

def swarm_metrics(positions, heading):
    distances = [(i, j, float(np.linalg.norm(positions[i]-positions[j])))
                 for i in range(NUM_DRONES) for j in range(i+1, NUM_DRONES)]
    forward = np.array([np.cos(heading), np.sin(heading)])
    errors = {i: float(np.linalg.norm(positions[i]-positions[LEADER_ID]
                   - np.r_[relative_xy(forward, i), 0.0]))
              for i in FOLLOWER_SLOTS}
    return distances, errors

def validate_geometry():
    """명령 대형 자체의 최소 간격 검증 (실제 비행 충돌 방지는 아님)."""
    theta = np.arctan2(1.0, 1.0)
    f = np.array([np.cos(theta), np.sin(theta)])
    slots = np.array([[0.,0.], relative_xy(f,1), relative_xy(f,2)])
    min_gap = min(np.linalg.norm(slots[i]-slots[j])
                  for i in range(NUM_DRONES) for j in range(i+1, NUM_DRONES))
    if min_gap <= MIN_DRONE_DISTANCE:
        raise ValueError(f'Formation desired spacing {min_gap:.3f} <= minimum distance threshold')


# ================== 결과 저장·그래프 ==================

COLUMNS = [
    'time_s', 'phase', 'drone_id', 'role', 'nav_mode',
    'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
    'roll_deg', 'pitch_deg', 'yaw_deg',
    'target_x_m', 'target_y_m', 'target_z_m',
    'target_error_3d_m', 'formation_error_3d_m',
    'min_pair_distance_m', 'distance_violation', 'motor_saturated',
    'motor_1_rpm', 'motor_2_rpm', 'motor_3_rpm', 'motor_4_rpm',
]


def save_plot(csv_path, png_path):
    if not SAVE_PLOT:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[PLOT] matplotlib unavailable; skipping plot')
        return
    data = np.genfromtxt(csv_path, delimiter=',', names=True, encoding='utf-8-sig')
    if data.size == 0:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for i in range(NUM_DRONES):
        d = data[data['drone_id'] == i]
        axes[0, 0].plot(d['x_m'], d['y_m'], label=f'Drone {i}')
        axes[0, 1].plot(d['time_s'], d['target_error_3d_m'], label=f'Drone {i}')
        if i != LEADER_ID:
            axes[1, 0].plot(d['time_s'], d['formation_error_3d_m'], label=f'Follower {i}')
    route = np.array([INITIAL_XYZ_M] + [tuple(w) for w in WAYPOINTS])
    axes[0, 0].plot(route[:,0], route[:,1], 'k--', label='Leader route')
    axes[0, 0].set(xlabel='X [m]', ylabel='Y [m]', title='XY trajectories')
    axes[0, 0].axis('equal')
    axes[0, 1].set(xlabel='Time [s]', ylabel='3D error [m]', title='Target errors')
    axes[1, 0].set(xlabel='Time [s]', ylabel='3D error [m]', title='Formation errors')
    d0 = data[data['drone_id'] == LEADER_ID]
    axes[1, 1].plot(d0['time_s'], d0['min_pair_distance_m'], label='Minimum actual distance')
    axes[1, 1].axhline(MIN_DRONE_DISTANCE, linestyle='--', label='Threshold')
    axes[1, 1].set(xlabel='Time [s]', ylabel='Distance [m]', title='Inter-drone distance')
    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=130)
    plt.close(fig)




# ================== 시뮬레이션 및 단계 관리 ==================

def create_env():
    return CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=NUM_DRONES,
        initial_xyzs=INITIAL_XYZS.copy(),
        initial_rpys=np.tile(np.deg2rad(INITIAL_RPY_DEG), (NUM_DRONES, 1)),
        pyb_freq=SIM_FREQ, ctrl_freq=SIM_FREQ, gui=GUI, user_debug_gui=False,
    )

def main():
    validate_geometry()
    if DURATION <= 0:
        raise ValueError('DURATION must be positive')
    if set(FOLLOWER_SLOTS) != set(range(NUM_DRONES)) - {LEADER_ID}:
        raise ValueError('Every follower must have one fixed offset')
    if MIN_DRONE_DISTANCE <= 0:
        raise ValueError('MIN_DRONE_DISTANCE must be positive')
    for i in range(NUM_DRONES):
        for j in range(i + 1, NUM_DRONES):
            if np.linalg.norm(INITIAL_XYZS[i] - INITIAL_XYZS[j]) < MIN_DRONE_DISTANCE:
                raise ValueError(f'Initial distance below threshold: {i}-{j}')
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = LOG_DIR / 'swarm_motion.csv'
    report_path = LOG_DIR / 'swarm_report.json'
    plot_path = LOG_DIR / 'swarm_motion.png'
    env = None
    error = None
    steps = 0
    min_distance_seen = float('inf')
    distance_violations = 0
    sum_formation_error = np.zeros(NUM_DRONES)
    max_formation_error = np.zeros(NUM_DRONES)
    max_rpm = np.zeros(NUM_DRONES)
    saturations = np.zeros(NUM_DRONES, dtype=int)
    nav_modes = [NAV_HOVER] * NUM_DRONES
    nav_stable_s = [0.0] * NUM_DRONES
    trails = [None] * NUM_DRONES
    last_warn_t = -10.0
    phase = PHASE_HOVER
    planner = LeaderPlanner()
    heading = planner.heading
    omega = 0.0
    alpha = 0.0
    speed_scale = 1.0
    last_waypoint_index = 0
    move_start_s = None
    goal_reached_s = None
    formation_started_s = None
    hover_completed_s = None
    last_sample_t = 0.0
    last_positions = None
    last_velocities = None
    segment_origins = INITIAL_XYZS.copy()
    move_samples = 0
    start_wall = time.perf_counter()

    try:
        env = create_env()
        mg = float(env.GRAVITY)
        params = dict(mass=float(env.M), g=mg / float(env.M),
                      arm=float(env.L / np.sqrt(2.0)), kf=float(env.KF),
                      km=float(env.KM), izz=float(env.J[2, 2]),
                      max_f=float(env.KF * env.MAX_RPM ** 2))
        if min(params.values()) <= 0:
            raise ValueError('Invalid physical constants')
        action = np.full((NUM_DRONES, 4), np.sqrt(mg / (4 * params['kf'])))
        prev_saturated = np.zeros(NUM_DRONES, dtype=bool)
        obs, _ = env.reset()
        camera_target = INITIAL_XYZS.mean(axis=0).copy()
        camera_interval = max(1, round(SIM_FREQ / CAMERA_UPDATE_HZ))
        camera_dt = camera_interval / SIM_FREQ
        camera_alpha = (1.0 if CAMERA_FOLLOW_TAU_S <= 0 else
                        1.0 - np.exp(-camera_dt / CAMERA_FOLLOW_TAU_S))
        if GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW,
                cameraPitch=CAMERA_PITCH,
                cameraTargetPosition=camera_target.tolist(),
                physicsClientId=env.CLIENT,
            )
        dt = 1.0 / SIM_FREQ
        with raw_path.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS)
            writer.writeheader()
            for k in range(int(round(DURATION * SIM_FREQ))):
                applied = action.copy()
                applied_saturated = prev_saturated.copy()
                obs, _, terminated, truncated, _ = env.step(applied)
                t = (k + 1) * dt
                states = np.asarray(obs, dtype=float)
                if states.shape[0] != NUM_DRONES:
                    raise RuntimeError(f'Expected {NUM_DRONES} drone states, got {states.shape}')
                positions = states[:, :3]
                velocities = states[:, 10:13]
                pair_distances, formation_errors = swarm_metrics(positions, heading)
                step_min_distance = min(distance for _, _, distance in pair_distances)

                # 단계 전환은 경과 시간이 아닌 현재 측정 오차로만 결정한다.
                # 세 드론 모두 초기 좌표에서 정지·수평 상태이면 호버링 성공.
                if phase == PHASE_HOVER:
                    hover_position_error = np.linalg.norm(positions - INITIAL_XYZS, axis=1)
                    hover_speed = np.linalg.norm(velocities, axis=1)
                    hover_rpy = np.rad2deg(states[:, 7:10])
                    hover_body_rates = np.array([measure_state(st)[3] for st in states])
                    hover_rate_deg = np.abs(np.rad2deg(hover_body_rates))
                    hover_ready = (
                        np.all(hover_position_error <= HOVER_POSITION_TOL_M)
                        and np.all(hover_speed <= HOVER_SPEED_TOL_M_S)
                        and np.all(np.abs(hover_rpy[:, :2]) <= HOVER_ANGLE_TOL_DEG)
                        and np.all(hover_rate_deg <= HOVER_RATE_TOL_DEG_S)
                        and step_min_distance >= MIN_DRONE_DISTANCE
                    )
                    if hover_ready:
                        hover_completed_s = t
                        formation_started_s = t
                        phase = PHASE_FORM
                        nav_modes = [NAV_APPROACH] * NUM_DRONES
                        nav_stable_s = [0.0] * NUM_DRONES
                        print(f'[PHASE] HOVER complete / FORM begins: t={t:.3f}s', flush=True)
                elif phase == PHASE_FORM:
                    max_formation_err = max(formation_errors.values())
                    max_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
                    # Leader가 초기 위치에 멈춰 있는 것도 함께 확인한다.
                    leader_hold_error = float(np.linalg.norm(positions[LEADER_ID] - INITIAL_XYZS[LEADER_ID]))
                    formed = (
                        max_formation_err <= FORMATION_ERROR_TOL_M
                        and max_speed <= FORMATION_SPEED_TOL_M_S
                        and leader_hold_error <= FORMATION_ERROR_TOL_M
                        and step_min_distance >= MIN_DRONE_DISTANCE
                    )
                    if formed:
                        phase = PHASE_MOVE
                        move_start_s = t
                        segment_origins = positions.copy()
                        nav_modes = [NAV_APPROACH] * NUM_DRONES
                        nav_stable_s = [0.0] * NUM_DRONES
                        print(f'[PHASE] FORM complete / MOVE: all 3 start at t={t:.3f}s', flush=True)

                if phase == PHASE_MOVE:
                    cmd = planner.update(positions[LEADER_ID], formation_errors, step_min_distance, dt)
                    heading, omega, alpha = cmd.heading, cmd.omega, cmd.alpha
                    speed_scale = cmd.speed_scale
                    if cmd.waypoint_index != last_waypoint_index:
                        last_waypoint_index = cmd.waypoint_index
                        nav_modes[LEADER_ID] = NAV_APPROACH
                        nav_stable_s[LEADER_ID] = 0.0
                    pair_distances, formation_errors = swarm_metrics(positions, heading)
                # 이동 완료 시각은 군집 전체가 목표 대형에서 정지했을 때 최초 기록한다.
                if phase == PHASE_MOVE and goal_reached_s is None:
                    leader_error = float(np.linalg.norm(positions[LEADER_ID] - np.asarray(WAYPOINTS[-1])))
                    formation_max = max(formation_errors.values())
                    speed_max = float(np.max(np.linalg.norm(velocities, axis=1)))
                    if (last_waypoint_index == len(WAYPOINTS)-1
                            and leader_error <= FORMATION_ERROR_TOL_M
                            and formation_max <= FORMATION_ERROR_TOL_M
                            and speed_max <= FORMATION_SPEED_TOL_M_S
                            and step_min_distance >= MIN_DRONE_DISTANCE):
                        goal_reached_s = t
                        print(f'[GOAL] Group reached and settled: t={t:.3f}s', flush=True)

                leader_goal = WAYPOINTS[last_waypoint_index] if phase == PHASE_MOVE else START_XYZ
                targets, reference_velocities, reference_accelerations = targets_and_references(
                    positions, velocities[LEADER_ID], np.zeros(2),
                    heading, omega, alpha, phase, leader_goal)
                min_distance_seen = min(min_distance_seen, step_min_distance)
                violation = step_min_distance < MIN_DRONE_DISTANCE
                distance_violations += int(violation)
                if violation and t - last_warn_t >= 1.0:
                    print(f'[DISTANCE WARNING] t={t:.2f}s, minimum={step_min_distance:.3f}m')
                    last_warn_t = t

                next_action = np.empty_like(action)
                next_saturated = np.empty_like(prev_saturated)
                leader_accel_xy = np.zeros(2, dtype=float)
                for i in range(NUM_DRONES):
                    following = phase == PHASE_MOVE and i != LEADER_ID
                    # Leader에서 계산한 XY 가속도에 회전 운동학 가속도를 더한다.
                    if following:
                        r = relative_xy(np.array([np.cos(heading), np.sin(heading)]), i)
                        reference_accelerations[i] = (leader_accel_xy + alpha*(J @ r) - omega**2*r)
                    rpm, sat, nav_modes[i], nav_stable_s[i], accel_xy = compute_drone_rpm(
                        states[i], targets[i], segment_origins[i],
                        nav_modes[i], nav_stable_s[i], t, dt, params,
                        reference_velocity_xy=reference_velocities[i] if following else None,
                        reference_accel_xy=reference_accelerations[i] if following else None,
                        speed_scale=speed_scale if i == LEADER_ID and phase == PHASE_MOVE else SPEED_SCALE,
                    )
                    if i == LEADER_ID:
                        leader_accel_xy = accel_xy.copy()
                    next_action[i] = rpm
                    next_saturated[i] = sat
                    pos, vel, rpy, _ = measure_state(states[i])
                    f_error = 0.0 if i == LEADER_ID else formation_errors[i]
                    if i != LEADER_ID and phase == PHASE_MOVE:
                        sum_formation_error[i] += f_error
                        max_formation_error[i] = max(max_formation_error[i], f_error)
                    max_rpm[i] = max(max_rpm[i], float(np.max(applied[i])))
                    saturations[i] += int(applied_saturated[i])
                    writer.writerow(dict(
                        time_s=t, phase=phase, drone_id=i,
                        role='leader' if i == LEADER_ID else 'follower',
                        nav_mode=nav_modes[i],
                        x_m=pos[0], y_m=pos[1], z_m=pos[2],
                        vx_m_s=vel[0], vy_m_s=vel[1], vz_m_s=vel[2],
                        roll_deg=np.rad2deg(rpy[0]), pitch_deg=np.rad2deg(rpy[1]),
                        yaw_deg=np.rad2deg(rpy[2]),
                        target_x_m=targets[i, 0], target_y_m=targets[i, 1],
                        target_z_m=targets[i, 2],
                        target_error_3d_m=float(np.linalg.norm(targets[i] - pos)),
                        formation_error_3d_m=f_error,
                        min_pair_distance_m=step_min_distance,
                        distance_violation=int(violation), motor_saturated=int(applied_saturated[i]),
                        motor_1_rpm=applied[i, 0], motor_2_rpm=applied[i, 1],
                        motor_3_rpm=applied[i, 2], motor_4_rpm=applied[i, 3],
                    ))
                action = next_action
                prev_saturated = next_saturated
                steps += 1
                last_sample_t = t
                last_positions = positions.copy()
                last_velocities = velocities.copy()
                move_samples += int(phase == PHASE_MOVE)

                if GUI and DRAW_TRAIL and k % max(1, round(SIM_FREQ / TRAIL_HZ)) == 0:
                    colors = ([0.2, 0.8, 0.2], [0.1, 0.4, 1.0], [1.0, 0.4, 0.1])
                    for i in range(NUM_DRONES):
                        if trails[i] is not None:
                            p.addUserDebugLine(trails[i].tolist(), positions[i].tolist(),
                                               lineColorRGB=colors[i], lineWidth=2,
                                               physicsClientId=env.CLIENT)
                        trails[i] = positions[i].copy()
                if GUI and CAMERA_MODE == 'follow' and k % camera_interval == 0:
                    # 군집 중심을 따라가되 순간적인 위치 변화로 카메라가 튀지 않도록 평활화.
                    # 사용자가 마우스로 변경한 카메라 거리/각도는 유지한다.
                    desired_center = positions.mean(axis=0)
                    camera_target += camera_alpha * (desired_center - camera_target)
                    cam = p.getDebugVisualizerCamera(physicsClientId=env.CLIENT)
                    p.resetDebugVisualizerCamera(
                        cameraDistance=float(cam[10]),
                        cameraYaw=float(cam[8]), cameraPitch=float(cam[9]),
                        cameraTargetPosition=camera_target.tolist(),
                        physicsClientId=env.CLIENT,
                    )
                if k % max(1, round(SIM_FREQ / CONSOLE_HZ)) == 0:
                    print(f't={t:6.2f}s | phase={phase} | Leader XYerr={np.linalg.norm(targets[0,:2]-positions[0,:2]):.3f}m '
                          f'| F1={formation_errors[1]:.3f}m F2={formation_errors[2]:.3f}m '
                          f'| minDist={step_min_distance:.3f}m')
                if REAL_TIME_PLAYBACK:
                    remaining = t - (time.perf_counter() - start_wall)
                    if remaining > 0:
                        time.sleep(remaining)
                if np.any(terminated) or np.any(truncated):
                    print('[SIM] terminated/truncated')
                    break
    except Exception as exc:
        error = str(exc)
        print(f'[ERROR] {error}')
    finally:
        if env is not None:
            env.close()
        # 단계별 실제 소요 시간은 측정 시각의 차이로 계산한다.
        hover_duration_s = hover_completed_s
        formation_duration_s = (move_start_s - formation_started_s
                                if move_start_s is not None and formation_started_s is not None
                                else None)
        movement_duration_s = (last_sample_t - move_start_s
                               if move_start_s is not None else None)
        goal_travel_duration_s = (goal_reached_s - move_start_s
                                  if goal_reached_s is not None and move_start_s is not None
                                  else None)
        leader_final_error_m = (float(np.linalg.norm(last_positions[LEADER_ID] - WAYPOINTS[-1]))
                                if last_positions is not None else None)
        report = {
            'status': 'ERROR' if error else 'FINISHED', 'error': error,
            'controller': 'C08 PTP + yaw coupling compensation',
            'swarm_mode': 'hover -> formation -> simultaneous leader follower PTP',
            'move_start_s': move_start_s, 'formation_started_s': formation_started_s,
            'phase_durations_s': {
                'hover': hover_duration_s,
                'formation': formation_duration_s,
                'move_observed': movement_duration_s,
                'move_until_group_arrival': goal_travel_duration_s,
            },
            'leader_final_error_m': leader_final_error_m,
            'movement_started': move_start_s is not None,
            'group_goal_reached': goal_reached_s is not None,

            'num_drones': NUM_DRONES, 'leader_id': LEADER_ID,
            'leader_target_m': WAYPOINTS[-1].tolist(),
            'waypoints_m': [w.tolist() for w in WAYPOINTS],
            'leader_final_speed_scale': speed_scale,
            'leader_heading_deg': float(np.rad2deg(heading)),
            'minimum_distance_threshold_m': MIN_DRONE_DISTANCE,
            'minimum_observed_distance_m': min_distance_seen if steps else None,
            'distance_violation_steps': distance_violations,
            'duration_simulated_s': steps / SIM_FREQ,
            'formation_error_statistics_phase': 'move_only',
            'formation_error_mean_m': {str(i): float(sum_formation_error[i] / move_samples)
                                       if move_samples else None for i in FOLLOWER_SLOTS},
            'formation_error_max_m': {str(i): float(max_formation_error[i])
                                      for i in FOLLOWER_SLOTS},
            'max_motor_rpm': max_rpm.tolist(),
            'motor_saturation_steps': saturations.tolist(),
            'offsets_m': {str(i): list(relative_xy(FIRST_FORWARD, i)) + [0.] for i in FOLLOWER_SLOTS},
            'forward_unit_xy': FIRST_FORWARD.tolist(),
            'phase_labels': {'0': 'hover', '1': 'formation', '2': 'move'},
            'warning': 'Minimum distance is monitored only; collision avoidance not implemented.',
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if raw_path.exists():
            try:
                save_plot(raw_path, plot_path)
            except Exception as exc:
                print('[PLOT] failed:', exc)
        print('[OUTPUT]', raw_path, report_path, plot_path)
        def fmt_seconds(value):
            return f'{value:.3f}s' if value is not None else '미완료'

        print('\n========== SWARM STAGE SUMMARY ==========', flush=True)
        print(f'Hover (오차 조건 충족): {fmt_seconds(hover_duration_s)}')
        print(f'Formation (대형 형성): {fmt_seconds(formation_duration_s)}')
        print(f'Move (군집 도착까지):   {fmt_seconds(goal_travel_duration_s)}')
        print(f'Move (전체 관찰시간):   {fmt_seconds(movement_duration_s)}')
        if leader_final_error_m is not None:
            print(f'Leader 최종 목표 오차: {leader_final_error_m:.3f}m')
        print(f'최소 드론 간 거리:     {min_distance_seen:.3f}m')
        print(f'거리 위반 샘플 수:     {distance_violations}')
        print(f'최종 경유점:         {last_waypoint_index+1}/{len(WAYPOINTS)}')
        print(f'군집 최종 진행방향: {np.rad2deg(heading):.1f}°')
        for i in FOLLOWER_SLOTS:
            if move_samples:
                print(f'Follower {i} 평균/최대 대형 오차: '
                      f'{sum_formation_error[i]/move_samples:.3f}m / {max_formation_error[i]:.3f}m')
        if phase != PHASE_MOVE:
            print('[주의] 조건을 만족하지 않아 MOVE가 시작되지 않았습니다.')
        print('=========================================')
    if error:
        raise RuntimeError(error)
    return report




if __name__ == "__main__":
    main()
