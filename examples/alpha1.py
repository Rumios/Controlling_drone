"""CF2X 목표점/경유점 자율이동 — 검증된 C08 + Yaw 결합 보상 (240 Hz).

- 단일 목표점: WAYPOINTS에 (x,y,z,yaw_deg) 한 개 입력 (기본, 먼저 검증할 모드)
- 경유점 경로: 여러 개 입력하면 모든 축 정착 후 다음 경유점으로 이동 (새 기능, 미검증)
- 경로 탐색(A*/RRT), 장애물 회피, 실기체 안전 보증은 구현하지 않음.
- 테스트한 제어기의 Roll/Pitch/XY/Yaw/고도 수식과 C08 값은 유지.
- 모터 RPM 직접 계산, 0..MAX_RPM 제한, CSV 240Hz 기록.

Ubuntu 22.04 / gym-pybullet-drones / PyBullet, Python 3.
"""
import csv
import json
import math
import time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel

# =========================== 1. 설정 ===========================
SIM_FREQ = 240
DURATION = 35.0   # 최종 목표에서 계속 호버링하는 관찰 시간 포함
START_HOVER_S = 1.0
INITIAL_XYZ_M = (0.0, 0.0, 150.0)
INITIAL_RPY_DEG = (0.0, 0.0, 0.0)

# 우선 기존 검증과 유사한 단일 목표점. 여러 개 입력하면 순차 경유점 이동.
# 항목 형식: (X[m], Y[m], Z[m], Yaw[deg]) — 절대 월드 좌표계.
WAYPOINTS = [
    (10.0, 10.0, 150.0, 0.0),
    # (10.0, -10.0, 150.0, 90.0),  # 경유점 실험용 (아직 연속비행 미검증)
    # (0.0, 0.0, 150.0, 0.0),
]

GUI = True
REAL_TIME_PLAYBACK = True
CAMERA_MODE = 'free'  # 'free' / 'follow'
CAMERA_DISTANCE = 23.0
CAMERA_YAW = 45.0
CAMERA_PITCH = -35.0
CAMERA_FOLLOW_TAU_S = 0.04
CAMERA_UPDATE_HZ = 60
DRAW_TRAIL = True
TRAIL_HZ = 30
CONSOLE_HZ = 1
SAVE_PLOT = True

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == 'examples':
    PROJECT_ROOT = PROJECT_ROOT.parent
LOG_DIR = PROJECT_ROOT / 'logs'

# ---------------- C08 실측 A/B 통과 설정 (변경하지 않음) ----------------
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

# 지난 A/B 실험의 엄격한 관측/평가 기준.
SETTLE_XY_M = 0.10
SETTLE_XY_SPEED_M_S = 0.10
SETTLE_YAW_DEG = 0.25
SETTLE_YAW_RATE_DEG_S = 0.50
SETTLE_Z_M = 0.10
SETTLE_Z_SPEED_M_S = 0.10
STABLE_REQUIRED_S = 0.45
POST_SETTLE_OBSERVE_S = 0.55
XY_OVERSHOOT_EPS_M = 0.001
YAW_OVERSHOOT_EPS_DEG = 0.025
ALT_OVERSHOOT_EPS_M = 0.005
ABORT_TILT_DEG = 35.0
ABORT_RATE_DEG_S = 900.0
ABORT_MIN_Z_M = 60.0

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

# ===================== 3. 순차 목표점 관리자 =====================

def wrap_deg(value):
    return (float(value)+180.0) % 360.0 - 180.0


class RouteManager:
    """STOP-AND-GO. 각 경유점은 위치+속도+Yaw+고도 정착 뒤 전환.

    waypoint 변경은 경로 관리만 담당하고, C08의 저수준 제어기는 수정하지 않는다.
    """
    def __init__(self, waypoints, start_xyz, start_yaw_deg):
        if not waypoints:
            raise ValueError('WAYPOINTS는 최소 하나가 필요합니다.')
        self.goals = [tuple(map(float, w)) for w in waypoints]
        if not all(len(w)==4 and np.all(np.isfinite(w)) for w in self.goals):
            raise ValueError('경유점은 유한한 (x,y,z,yaw_deg) 네 값이어야 합니다.')
        self.index=0
        self.origin=np.asarray(start_xyz, dtype=float).copy()
        self.origin_yaw=float(start_yaw_deg)
        self.nav=NAV_HOVER
        self.nav_xy_stable=0.0
        self.stable_since=None
        self.arrived_since=None
        self.completed=[]
        self.finished=False
        self.started_at=START_HOVER_S
        self.metrics=self.new_metrics()

    @staticmethod
    def new_metrics():
        return dict(xy_overshoot_m=0.0, yaw_overshoot_deg=0.0,
                    alt_overshoot_m=0.0, max_xy_speed_m_s=0.0,
                    max_yaw_abs_rate_deg_s=0.0, max_rpm=0.0,
                    motor_saturation_steps=0, n_steps=0)

    @property
    def goal(self):
        return self.goals[self.index]

    def goal_yaw_unwrapped(self, current_yaw_unwrapped_deg):
        # shortest turn from segment starting yaw; fixed for this segment
        return self.origin_yaw + wrap_deg(self.goal[3] - self.origin_yaw)

    def update(self, t, dt, pos, vel, rpy_deg, euler_rate_deg_s,
               yaw_unwrapped_deg, rpm, motor_saturated):
        """Post-physics observation -> waypoint completion/transition decision."""
        if self.finished:
            return None
        goal=np.asarray(self.goal[:3],dtype=float)
        gxy=goal[:2]
        dist=float(np.linalg.norm(gxy-pos[:2]))
        speed=float(np.linalg.norm(vel[:2]))
        yaw_target_unwrapped=self.goal_yaw_unwrapped(yaw_unwrapped_deg)
        yaw_error=abs(yaw_target_unwrapped-yaw_unwrapped_deg)
        z_error=abs(goal[2]-pos[2])
        same=(dist<=SETTLE_XY_M and speed<=SETTLE_XY_SPEED_M_S and
              yaw_error<=SETTLE_YAW_DEG and abs(euler_rate_deg_s)<=SETTLE_YAW_RATE_DEG_S and
              z_error<=SETTLE_Z_M and abs(float(vel[2]))<=SETTLE_Z_SPEED_M_S)
        # Overshoot uses segment starting point and movement direction, not world origin.
        disp=gxy-self.origin[:2]
        length=float(np.linalg.norm(disp))
        if length>1e-10:
            axis=disp/length
            self.metrics['xy_overshoot_m']=max(self.metrics['xy_overshoot_m'],
                0.0,float(np.dot(pos[:2]-self.origin[:2],axis))-length)
        yaw_dir=np.sign(yaw_target_unwrapped-self.origin_yaw)
        if yaw_dir:
            self.metrics['yaw_overshoot_deg']=max(self.metrics['yaw_overshoot_deg'],
                                                  0.0,float(yaw_dir*(yaw_unwrapped_deg-yaw_target_unwrapped)))
        z_dir=np.sign(goal[2]-self.origin[2])
        if z_dir:
            self.metrics['alt_overshoot_m']=max(self.metrics['alt_overshoot_m'],
                                                0.0,float(z_dir*(pos[2]-goal[2])))
        self.metrics['max_xy_speed_m_s']=max(self.metrics['max_xy_speed_m_s'],speed)
        self.metrics['max_yaw_abs_rate_deg_s']=max(self.metrics['max_yaw_abs_rate_deg_s'],abs(euler_rate_deg_s))
        self.metrics['max_rpm']=max(self.metrics['max_rpm'],float(np.max(rpm)))
        self.metrics['motor_saturation_steps']+=int(motor_saturated)
        self.metrics['n_steps']+=1
        if t<START_HOVER_S:
            return None
        if same:
            if self.stable_since is None:self.stable_since=t
            if self.arrived_since is None and t-self.stable_since>=STABLE_REQUIRED_S:
                self.arrived_since=t
        else:
            self.stable_since=None
            self.arrived_since=None
        if self.arrived_since is None or t-self.arrived_since<POST_SETTLE_OBSERVE_S:
            return None
        # A complete waypoint requires ALL axes stable over continuous observation.
        m=dict(self.metrics)
        if m['motor_saturation_steps']:
            status='SATURATED'
        elif m['xy_overshoot_m']>XY_OVERSHOOT_EPS_M:
            status='XY_OVERSHOOT'
        elif m['yaw_overshoot_deg']>YAW_OVERSHOOT_EPS_DEG:
            status='YAW_OVERSHOOT'
        elif m['alt_overshoot_m']>ALT_OVERSHOOT_EPS_M:
            status='ALT_OVERSHOOT'
        else:status='PASS'
        summary=dict(index=self.index+1,goal_x=goal[0],goal_y=goal[1],
                     goal_z=goal[2],goal_yaw_deg=self.goal[3],
                     status=status,settle_time_s=float(self.arrived_since),
                     reached_time_s=float(t),
                     segment_duration_s=float(t-self.started_at),**m)
        self.completed.append(summary)
        return summary

    def advance(self, pos, yaw_unwrapped_deg, t):
        """Return True if a new segment was activated; False for final waypoint."""
        if self.index+1==len(self.goals):
            self.nav=NAV_HOLD
            self.finished=True
            return False
        self.index+=1
        self.origin=np.asarray(pos,dtype=float).copy()
        self.origin_yaw=float(yaw_unwrapped_deg)
        self.started_at=t
        self.metrics=self.new_metrics()
        self.stable_since=None
        self.arrived_since=None
        self.nav=NAV_APPROACH
        self.nav_xy_stable=0.0
        return True


# ========================== 4. 로깅/평가 ==========================
COLUMNS=[
    'time_s','waypoint_index','nav_mode',
    'x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s',
    'roll_deg','pitch_deg','yaw_unwrapped_deg',
    'p_body_deg_s','q_body_deg_s','r_body_deg_s','yaw_euler_rate_deg_s',
    'yaw_coupling_rate_deg_s','yaw_body_rate_ref_deg_s',
    'target_x_m','target_y_m','target_z_m','target_yaw_deg',
    'xy_distance_m','xy_speed_m_s','yaw_error_deg','z_error_m',
    'xy_speed_ref_m_s','closing_speed_m_s','brake_guard',
    'ideal_stop_distance_m','brake_margin_m',
    'xy_ax_cmd_m_s2','xy_ay_cmd_m_s2','roll_ref_deg','pitch_ref_deg',
    'yaw_torque_Nm','yaw_control_mode','alt_acc_cmd_m_s2','total_thrust_cmd_N',
    'motor_saturated','motor_1_rpm','motor_2_rpm','motor_3_rpm','motor_4_rpm'
]
WAYPOINT_COLUMNS=['index','goal_x','goal_y','goal_z','goal_yaw_deg','status',
                  'settle_time_s','reached_time_s','segment_duration_s',
                  'xy_overshoot_m','yaw_overshoot_deg','alt_overshoot_m',
                  'max_xy_speed_m_s','max_yaw_abs_rate_deg_s','max_rpm',
                  'motor_saturation_steps','n_steps']


def _create_env():
    return CtrlAviary(drone_model=DroneModel.CF2X,num_drones=1,
                      initial_xyzs=np.array([INITIAL_XYZ_M],dtype=float),
                      initial_rpys=np.deg2rad(np.array([INITIAL_RPY_DEG],dtype=float)),
                      pyb_freq=SIM_FREQ,ctrl_freq=SIM_FREQ,
                      gui=GUI,user_debug_gui=False)


def _draw_goal(client, goal, old_ids):
    for i in old_ids:
        p.removeUserDebugItem(i,physicsClientId=client)
    if not GUI:return []
    x,y,z,_=goal
    return [p.addUserDebugLine([x-.35,y,z],[x+.35,y,z],lineColorRGB=[0.,.7,0.],
                               lineWidth=2,physicsClientId=client),
            p.addUserDebugLine([x,y-.35,z],[x,y+.35,z],lineColorRGB=[0.,.7,0.],
                               lineWidth=2,physicsClientId=client)]


def _plot(raw_path, wp_path, plot_path):
    if not SAVE_PLOT:return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    data=np.genfromtxt(raw_path,delimiter=',',names=True,encoding='utf-8-sig',invalid_raise=False)
    if data.size==0:return
    fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    t=data['time_s']
    axs[0,0].plot(data['x_m'],data['y_m'],label='Actual XY')
    route=np.asarray([INITIAL_XYZ_M[:2],*[w[:2] for w in WAYPOINTS]],dtype=float)
    axs[0,0].plot(route[:,0],route[:,1],'o--',label='Waypoints')
    axs[0,0].set(xlabel='X [m]',ylabel='Y [m]',title='XY route');axs[0,0].axis('equal')
    axs[0,1].plot(t,data['xy_distance_m'],label='XY target distance')
    axs[0,1].plot(t,data['xy_speed_m_s'],label='XY speed [m/s]')
    axs[0,1].set(xlabel='Time [s]',title='Target distance / speed')
    axs[1,0].plot(t,data['yaw_unwrapped_deg'],label='Yaw')
    axs[1,0].plot(t,data['target_yaw_deg'],label='Yaw target')
    axs[1,0].set(xlabel='Time [s]',ylabel='Yaw [deg]',title='Compensated yaw')
    axs[1,1].plot(t,data['z_m'],label='Altitude')
    axs[1,1].plot(t,data['target_z_m'],label='Altitude target')
    axs[1,1].set(xlabel='Time [s]',ylabel='Z [m]',title='Altitude')
    for ax in axs.flat:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(plot_path,dpi=130);plt.close(fig)


def main():
    if DURATION <= START_HOVER_S:
        raise ValueError('DURATION은 START_HOVER_S보다 커야 합니다.')
    if len(WAYPOINTS)<1:
        raise ValueError('WAYPOINTS를 하나 이상 입력하세요.')
    LOG_DIR.mkdir(parents=True,exist_ok=True)
    raw_path=LOG_DIR/'drone_motion.csv'
    wp_path=LOG_DIR/'drone_motion_waypoints.csv'
    report_path=LOG_DIR/'drone_motion_goal.txt'
    config_path=LOG_DIR/'drone_motion_config.json'
    plot_path=LOG_DIR/'drone_motion.png'
    bundle_path=LOG_DIR/'drone_motion_bundle.zip'
    env=None
    route=RouteManager(WAYPOINTS, INITIAL_XYZ_M, INITIAL_RPY_DEG[2])
    waypoint_events=[]
    error=None
    max_abs_roll=0.0;max_abs_pitch=0.0
    rows_count=0
    start_wall=time.perf_counter()
    try:
        env=_create_env()
        # CF2X model enforces RPM; GRAVITY in CtrlAviary is weight force [N].
        mg=float(env.GRAVITY);mass=float(env.M);g=mg/mass
        arm=float(env.L/np.sqrt(2.0));kf=float(env.KF);km=float(env.KM)
        izz=float(env.J[2,2]);max_f=kf*float(env.MAX_RPM)**2
        if min(mass,kf,km,izz,max_f)<=0:raise ValueError('유효하지 않은 물리 계수')
        action=np.full((1,4),np.sqrt(mg/(4.*kf)),dtype=float)
        prev_saturated=False
        obs,_=env.reset()
        state=np.asarray(obs[0],dtype=float)
        yaw_prev=float(np.rad2deg(state[9]));yaw_unwrapped=yaw_prev
        cam_target=np.array(INITIAL_XYZ_M,dtype=float)
        draw_ids=[]
        if GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=env.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=CAMERA_DISTANCE,
                cameraYaw=CAMERA_YAW,cameraPitch=CAMERA_PITCH,
                cameraTargetPosition=[5.,5.,INITIAL_XYZ_M[2]],physicsClientId=env.CLIENT)
            draw_ids=_draw_goal(env.CLIENT,route.goal,[])
        dt=1./SIM_FREQ
        with raw_path.open('w',newline='',encoding='utf-8-sig') as rawfile:
            writer=csv.DictWriter(rawfile,fieldnames=COLUMNS)
            writer.writeheader()
            for i in range(int(round(DURATION*SIM_FREQ))):
                applied=action.copy();sat_applied=prev_saturated
                obs,_,terminated,truncated,_=env.step(applied)
                state=np.asarray(obs[0],dtype=float)
                t=(i+1)*dt
                pos,vel,rpy,body_rates=measure_state(state)
                rpy_deg=np.rad2deg(rpy)
                yaw_unwrapped+=wrap_deg(rpy_deg[2]-yaw_prev)
                yaw_prev=float(rpy_deg[2])
                euler_rate,coupling_rate,_=yaw_kinematic_terms(rpy,body_rates)
                yaw_rate_deg=float(np.rad2deg(euler_rate))
                max_abs_roll=max(max_abs_roll,abs(float(rpy_deg[0])))
                max_abs_pitch=max(max_abs_pitch,abs(float(rpy_deg[1])))
                if (not np.all(np.isfinite(np.r_[pos,vel,rpy,body_rates,applied.flatten()]))
                    or abs(rpy_deg[0])>ABORT_TILT_DEG or abs(rpy_deg[1])>ABORT_TILT_DEG
                    or max(abs(np.rad2deg(body_rates)))>ABORT_RATE_DEG_S
                    or pos[2]<ABORT_MIN_Z_M):
                    raise RuntimeError(f'비정상 상태/보호조건 발생: t={t:.4f}s, pos={pos}, RPY={rpy_deg}')
                # Manage one waypoint at a time. Previous applied motor RPM is logged.
                active=route.goal
                if t>=START_HOVER_S and route.nav==NAV_HOVER:
                    route.nav=NAV_APPROACH
                target_xy=np.asarray(active[:2],dtype=float)
                target_z=float(active[2] if t>=START_HOVER_S else INITIAL_XYZ_M[2])
                target_yaw=float(active[3] if t>=START_HOVER_S else INITIAL_RPY_DEG[2])
                goal_dist=float(np.linalg.norm(target_xy-pos[:2]))
                speed=float(np.linalg.norm(vel[:2]))
                route.nav,route.nav_xy_stable=update_navigation_state(
                    route.nav,t,goal_dist,speed,route.nav_xy_stable,dt)
                # In HOVER track start XY; otherwise latest selected waypoint.
                acc_xy,nav_diag=point_to_point_controller(
                    pos[:2],vel[:2],route.nav,target_xy,route.origin[:2],SPEED_SCALE)
                alt=altitude_pd(pos[2],vel[2],target_z,mass,g,rpy[0],rpy[1])
                roll_ref,pitch_ref,acc_xy=xy_acceleration_to_attitude(acc_xy,rpy[2],alt[3],g)
                roll=roll_pd(rpy[0],body_rates[0],roll_ref)
                pitch=pitch_pd(rpy[1],body_rates[1],pitch_ref)
                pos_lim,neg_lim=yaw_torque_limits(
                    alt[4],roll[3],pitch[3],arm,kf,km,max_f)
                yaw,ydiag=yaw_predictive_control_coupled(
                    rpy[2],rpy,body_rates,np.deg2rad(target_yaw),izz,pos_lim,neg_lim)
                action,prev_saturated=allocate_motor_rpm(
                    roll[3],pitch[3],yaw[3],alt[4],arm,kf,km,max_f)
                if not np.all(np.isfinite(action)):
                    raise RuntimeError('비유한 모터 RPM 명령')
                writer.writerow(dict(
                    time_s=t,waypoint_index=route.index+1,nav_mode=route.nav,
                    x_m=pos[0],y_m=pos[1],z_m=pos[2],
                    vx_m_s=vel[0],vy_m_s=vel[1],vz_m_s=vel[2],
                    roll_deg=rpy_deg[0],pitch_deg=rpy_deg[1],
                    yaw_unwrapped_deg=yaw_unwrapped,
                    p_body_deg_s=np.rad2deg(body_rates[0]),
                    q_body_deg_s=np.rad2deg(body_rates[1]),
                    r_body_deg_s=np.rad2deg(body_rates[2]),
                    yaw_euler_rate_deg_s=yaw_rate_deg,
                    yaw_coupling_rate_deg_s=np.rad2deg(coupling_rate),
                    yaw_body_rate_ref_deg_s=np.rad2deg(ydiag['target_body_yaw_rate_rad_s']),
                    target_x_m=target_xy[0],target_y_m=target_xy[1],
                    target_z_m=target_z,target_yaw_deg=target_yaw,
                    xy_distance_m=goal_dist,xy_speed_m_s=speed,
                    yaw_error_deg=wrap_deg(target_yaw-yaw_unwrapped),z_error_m=target_z-pos[2],
                    xy_speed_ref_m_s=nav_diag['target_speed'],
                    closing_speed_m_s=nav_diag['closing_speed'],
                    brake_guard=int(nav_diag['emergency_brake'] and route.nav==NAV_APPROACH),
                    ideal_stop_distance_m=nav_diag['ideal_stop_distance'],
                    brake_margin_m=nav_diag['margin'],
                    xy_ax_cmd_m_s2=acc_xy[0],xy_ay_cmd_m_s2=acc_xy[1],
                    roll_ref_deg=np.rad2deg(roll_ref),pitch_ref_deg=np.rad2deg(pitch_ref),
                    yaw_torque_Nm=yaw[3],yaw_control_mode=yaw[7],
                    alt_acc_cmd_m_s2=alt[3],total_thrust_cmd_N=alt[4],
                    motor_saturated=int(sat_applied),
                    motor_1_rpm=applied[0,0],motor_2_rpm=applied[0,1],
                    motor_3_rpm=applied[0,2],motor_4_rpm=applied[0,3],
                ))
                rows_count+=1
                # Initial hover only; a stop-and-go intermediate waypoint is
                # activated AFTER all four goal dimensions have settled.
                event=route.update(t,dt,pos,vel,rpy_deg,yaw_rate_deg,
                                   yaw_unwrapped,applied,sat_applied)
                if event and (not waypoint_events or waypoint_events[-1]['index']!=event['index']):
                    waypoint_events.append(event)
                    print(f"[WAYPOINT {event['index']}/{len(WAYPOINTS)}] "
                          f"{event['status']} at t={event['reached_time_s']:.3f}s "
                          f"(xy-over={event['xy_overshoot_m']:.6f}m, "
                          f"yaw-over={event['yaw_overshoot_deg']:.6f}deg, "
                          f"z-over={event['alt_overshoot_m']:.6f}m)",flush=True)
                    changed=route.advance(pos,yaw_unwrapped,t)
                    if changed and GUI:draw_ids=_draw_goal(env.CLIENT,route.goal,draw_ids)
                if GUI and CAMERA_MODE=='follow' and i%max(1,round(SIM_FREQ/CAMERA_UPDATE_HZ))==0:
                    camera_data=p.getDebugVisualizerCamera(physicsClientId=env.CLIENT)
                    cam_dt=max(1,round(SIM_FREQ/CAMERA_UPDATE_HZ))/SIM_FREQ
                    alpha_cam=(1.0 if CAMERA_FOLLOW_TAU_S<=0 else
                               1.0-np.exp(-cam_dt/CAMERA_FOLLOW_TAU_S))
                    cam_target=cam_target+alpha_cam*(pos-cam_target)
                    p.resetDebugVisualizerCamera(cameraDistance=float(camera_data[10]),
                        cameraYaw=float(camera_data[8]),cameraPitch=float(camera_data[9]),
                        cameraTargetPosition=cam_target.tolist(),physicsClientId=env.CLIENT)
                if GUI and DRAW_TRAIL and i%max(1,round(SIM_FREQ/TRAIL_HZ))==0:
                    if 'old_line_pos' in locals():
                        p.addUserDebugLine(old_line_pos.tolist(),pos.tolist(),
                            lineColorRGB=[1.,.2,.2],lineWidth=2,physicsClientId=env.CLIENT)
                    old_line_pos=pos.copy()
                if i%max(1,round(SIM_FREQ/CONSOLE_HZ))==0:
                    print(f't={t:6.2f} | WP {route.index+1}/{len(WAYPOINTS)} '
                          f'| XYerr={goal_dist:.3f}m speed={speed:.3f}m/s '
                          f'| Yawerr={wrap_deg(target_yaw-yaw_unwrapped):+.3f}deg '
                          f'| Zerr={target_z-pos[2]:+.4f}m | nav={route.nav}')
                if REAL_TIME_PLAYBACK:
                    remaining=t-(time.perf_counter()-start_wall)
                    if remaining>0:time.sleep(remaining)
                if terminated or truncated:
                    print('[SIM] env terminated / truncated')
                    break
    except Exception as exc:
        error=str(exc)
        print(f'[ERROR] {error}',flush=True)
    finally:
        if env is not None:env.close()
        # Incomplete final leg is not silently omitted from the summary.
        if len(waypoint_events)<len(WAYPOINTS) and not route.finished:
            remaining_goal=route.goal
            metrics=dict(route.metrics)
            waypoint_events.append(dict(index=route.index+1,
                goal_x=float(remaining_goal[0]),goal_y=float(remaining_goal[1]),
                goal_z=float(remaining_goal[2]),goal_yaw_deg=float(remaining_goal[3]),
                status='ABORT' if error else 'TIMEOUT',
                settle_time_s='',reached_time_s='',segment_duration_s='',**metrics))
        with wp_path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=WAYPOINT_COLUMNS);w.writeheader();w.writerows(waypoint_events)
        cfg=dict(mode='navigation_c08',sim_hz=SIM_FREQ,
                 initial_xyz_m=INITIAL_XYZ_M,waypoints=WAYPOINTS,
                 controller='C08_conservative + coupled Euler-Yaw feedback',
                 tested_reference='29/29 and repeat 29/29 PASS in separate A/B rig',
                 route_mode='stop_and_go',route_waypoints_validated=False,
                 parameters={name:globals()[name] for name in (
                     'MAX_XY_ACCEL','APPROACH_ACCEL_M_S2','BRAKE_ACCEL_M_S2',
                     'MAX_APPROACH_SPEED_M_S','MAX_XY_TILT_DEG',
                     'MAX_YAW_RATE_DEG_S','MAX_YAW_ACCEL_DEG_S2','YAW_BRAKE_ACCEL_DEG_S2',
                     'YAW_RATE_TRACK_K','YAW_TORQUE_RESERVE',
                     'YAW_STOP_LEAD_S','YAW_STOP_MARGIN_DEG','YAW_TERMINAL_RATE_GAIN',
                     'KP_ALT','KD_ALT','ALT_LOOKAHEAD_S','MAX_VERTICAL_ACCEL')},
                 status='ERROR' if error else 'FINISHED',error=error,
                 visited_waypoints=sum(x['status'] not in ('TIMEOUT','ABORT') for x in waypoint_events),waypoints_total=len(WAYPOINTS),
                 raw_rows=rows_count)
        config_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        lines=['CF2X navigation C08 + compensated yaw',
               f'Waypoints reached {sum(x["status"] not in ("TIMEOUT","ABORT") for x in waypoint_events)}/{len(WAYPOINTS)}',
               f'240Hz logged rows {rows_count}',
               f'Max |roll| {max_abs_roll:.4f} deg, max |pitch| {max_abs_pitch:.4f} deg',
               'Each PASS means <=1mm XY, <=0.025deg yaw, <=5mm Z overshoot and no motor saturation,',
               'plus XY, yaw, Z position/velocity settling for 0.45s and extra 0.55s observation.',
               'Previous 29/29 PASS applies to C08 trial rig; this navigation integration is not yet validated.',
               'Multi-waypoint transitions are experimental; no obstacle avoidance or time-optimal path planning.',
               f'Error: {error}', '', 'Waypoint results:']
        for r in waypoint_events:
            settle_str=(f"{r['settle_time_s']:.4f}s" if isinstance(r['settle_time_s'],float)
                        else str(r['settle_time_s']) or 'unreached')
            lines.append(f"WP{r['index']}: {r['status']}, settle t={settle_str} "
                         f"max XY-over={r['xy_overshoot_m']:.6f}m "
                         f"Yaw-over={r['yaw_overshoot_deg']:.6f}deg "
                         f"Z-over={r['alt_overshoot_m']:.6f}m")
        report_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        try:_plot(raw_path,wp_path,plot_path)
        except Exception as ex:print('Plot failed:',repr(ex))
        with ZipFile(bundle_path,'w',compression=ZIP_DEFLATED,compresslevel=4) as zf:
            for path in (raw_path,wp_path,report_path,config_path,plot_path):
                if path.exists():zf.write(path,arcname=path.name)
        print(f'[OUTPUT] {bundle_path}',flush=True)
    if error:raise RuntimeError(error)
    return waypoint_events


if __name__=='__main__':
    main()
