"""CF2X 저수준 Roll/Pitch/고도 PD + Yaw 예측 제동 제어 (240 Hz).

배치: <project>/examples/pid.py  ->  로그: <project>/logs/
기본 실험: Roll=Pitch=0°, Yaw=+45° -> 0°, 고도 100m 유지 (Yaw 단독 검증).
기존 Roll/Pitch/고도 PD와 CF2X 믹싱은 유지하고 Yaw만 물리 기반 속도 계획기로 교체한다.
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
INITIAL_RPY_DEG = np.array([0.0, 45.0, 45.0])  # Yaw 단독 검증; 이후 [40,40,45] 통합 실험
TARGET_RPY_DEG = np.array([0.0, 0.0, 0.0])
INITIAL_ALTITUDE = 100.0
TARGET_ALTITUDE = 100.0
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
CAMERA_DISTANCE = 0.10
CAMERA_YAW = 45
CAMERA_PITCH = -35
SHOW_ATTITUDE_LABEL = True
REAL_TIME_PLAYBACK = True

# 스크립트 경로: project/examples/pid.py, 결과 경로: project/logs/
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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
def new_run_index():
    idx = 1
    while any((LOG_DIR / f"{prefix}_{idx}.{ext}").exists() for prefix, ext in (
        ("roll_pitch_yaw_alt_pd", "csv"), ("roll_pitch_yaw_alt_eval", "csv"),
        ("roll_pitch_yaw_alt_eval", "txt"),
        ("roll_pitch_yaw_alt_motion_eval", "csv"))):
        idx += 1
    return idx


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


def evaluate_and_save(data, run):
    windows = [
        ("initial_recovery", "초기 복귀", data[data[:, 0] <= INITIAL_EVAL_END + 1e-9]),
        ("long_term_hold", "장시간 유지", data[data[:, 0] > INITIAL_EVAL_END + 1e-9]),
        ("full_run", "전체 실행", data),
    ]
    report = [
        "======= Roll/Pitch/Altitude PD + Yaw 예측 제동 평가 =======",
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
    txt_path = LOG_DIR / f"roll_pitch_yaw_alt_eval_{run}.txt"
    csv_path = LOG_DIR / f"roll_pitch_yaw_alt_eval_{run}.csv"
    txt_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    with csv_path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    motion_path = LOG_DIR / f"roll_pitch_yaw_alt_motion_eval_{run}.csv"
    with motion_path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(motion_rows[0]))
        writer.writeheader()
        writer.writerows(motion_rows)
    print("\n".join(report))
    print(f"평가 TXT: {txt_path}\n평가 CSV: {csv_path}"
          f"\n운동 평가 CSV: {motion_path}")


def main():
    run = new_run_index()
    init_rpy = np.deg2rad(INITIAL_RPY_DEG)
    target_rpy = np.deg2rad(TARGET_RPY_DEG)
    dt = 1.0 / SIM_FREQ
    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, INITIAL_ALTITUDE]]),
        initial_rpys=init_rpy.reshape(1, 3),
        pyb_freq=SIM_FREQ, ctrl_freq=SIM_FREQ, gui=GUI,
        user_debug_gui=False,
    )
    # env.GRAVITY는 힘[N]=m*g. 기존 코드의 호버 추력 그대로 사용.
    gravity_force = float(env.GRAVITY)
    mass = float(env.M)
    gravity_acc = gravity_force / mass
    arm = float(env.L / np.sqrt(2.0))
    max_motor_thrust = float(env.KF * env.MAX_RPM ** 2)
    km = float(env.KM)
    if km <= 0.0:
        raise ValueError("CF2X Yaw 모터 토크 계수 KM이 0 이하입니다.")

    action = np.full((1, 4), np.sqrt((gravity_force / 4.0) / env.KF))
    initial_roll = roll_pd(init_rpy[0], 0., target_rpy[0])
    initial_pitch = pitch_pd(init_rpy[1], 0., target_rpy[1])
    initial_alt = altitude_pd(INITIAL_ALTITUDE, 0., TARGET_ALTITUDE,
                              mass, gravity_acc, init_rpy[0], init_rpy[1])
    yaw_inertia = float(env.J[2, 2])
    if yaw_inertia <= 0:
        raise ValueError("Yaw 관성모멘트 Izz가 0 이하입니다.")
    positive_limit, negative_limit = yaw_torque_limits(
        initial_alt[4], initial_roll[3], initial_pitch[3],
        arm, env.KF, km, max_motor_thrust)
    initial_yaw = yaw_predictive_control(
        init_rpy[2], 0.0, target_rpy[2], yaw_inertia, positive_limit, negative_limit)
    log = [[
        0., *INITIAL_RPY_DEG, 0., 0., 0.,
        np.rad2deg(initial_roll[0]), np.rad2deg(initial_pitch[0]), INITIAL_ALTITUDE,
        *initial_roll[1:], *initial_pitch[1:], *action[0], 0.,
        0., initial_alt[0], initial_alt[1], initial_alt[2], initial_alt[3],
        initial_alt[4], initial_alt[5], float(initial_alt[6]), float(initial_alt[7]),
        np.rad2deg(initial_yaw[0]), *initial_yaw[1:4],
        0., 0.,  # 초기 x, y
        0., 0., 0.,  # 초기 월드 좌표계 vx, vy, vz
        float("nan"), float("nan"), float("nan"),  # 초기 가속도는 아직 차분 불가
        np.rad2deg(initial_yaw[4]), np.rad2deg(initial_yaw[5]),
        np.rad2deg(initial_yaw[6]), float("nan"), float(initial_yaw[7]),
        initial_yaw[8], initial_yaw[9],
    ]]
    prev_world_vel = np.zeros(3, dtype=float)  # 시작 조건: 정지 상태
    prev_yaw_rate = 0.0
    filtered_yaw_acc = 0.0

    if GUI:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW,
            cameraPitch=CAMERA_PITCH,
            cameraTargetPosition=[0, 0, INITIAL_ALTITUDE],
            physicsClientId=env.CLIENT,
        )
    text_id = -1
    start = time.time()
    try:
        for i in range(int(DURATION * SIM_FREQ)):
            obs, _, terminated, truncated, _ = env.step(action)
            pos, world_vel, rpy, body_rates = measure_state(obs[0])
            # a_world[k] = (v_world[k] - v_world[k-1]) / (1/240s)
            # 센서의 직접 가속도 측정이 아니라 속도 기반 유한차분 값이며 중력 포함 운동 가속도.
            world_acc = (world_vel - prev_world_vel) / dt
            prev_world_vel = world_vel.copy()
            roll = roll_pd(rpy[0], body_rates[0], target_rpy[0])
            pitch = pitch_pd(rpy[1], body_rates[1], target_rpy[1])
            alt = altitude_pd(pos[2], world_vel[2], TARGET_ALTITUDE,
                              mass, gravity_acc, rpy[0], rpy[1])
            # 각속도 차분으로 Yaw 각가속도를 추정하되 저역통과 필터로 잡음 완화.
            raw_yaw_acc = (body_rates[2] - prev_yaw_rate) / dt
            filtered_yaw_acc += dt / (YAW_ACCEL_FILTER_TAU_S + dt) * (
                raw_yaw_acc - filtered_yaw_acc)
            prev_yaw_rate = float(body_rates[2])
            positive_limit, negative_limit = yaw_torque_limits(
                alt[4], roll[3], pitch[3], arm, env.KF, km, max_motor_thrust)
            yaw = yaw_predictive_control(
                rpy[2], body_rates[2], target_rpy[2], yaw_inertia,
                positive_limit, negative_limit)
            action, saturated = allocate_motor_rpm(
                roll[3], pitch[3], yaw[3], alt[4], arm, env.KF, km, max_motor_thrust)
            t = (i + 1) * dt
            log.append([
                t, *np.rad2deg(rpy), *np.rad2deg(body_rates),
                np.rad2deg(roll[0]), np.rad2deg(pitch[0]), pos[2],
                *roll[1:], *pitch[1:], *action[0], float(saturated),
                world_vel[2], alt[0], alt[1], alt[2], alt[3],
                alt[4], alt[5], float(alt[6]), float(alt[7]),
                np.rad2deg(yaw[0]), *yaw[1:4],
                pos[0], pos[1],
                *world_vel, *world_acc,
                np.rad2deg(yaw[4]), np.rad2deg(yaw[5]),
                np.rad2deg(yaw[6]), np.rad2deg(filtered_yaw_acc),
                float(yaw[7]), yaw[8], yaw[9],
            ])
            if GUI and i % 4 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=CAMERA_DISTANCE, cameraYaw=CAMERA_YAW,
                    cameraPitch=CAMERA_PITCH, cameraTargetPosition=pos,
                    physicsClientId=env.CLIENT)
                if SHOW_ATTITUDE_LABEL:
                    text_id = p.addUserDebugText(
                        text=(f"R {np.rad2deg(rpy[0]):+.1f} | P {np.rad2deg(rpy[1]):+.1f} "
                              f"| Y {np.rad2deg(rpy[2]):+.1f} | Z {pos[2]:.2f}m"),
                        textPosition=[0, 0, 0.1], textColorRGB=[0, 0, 0],
                        textSize=1.1, parentObjectUniqueId=int(env.DRONE_IDS[0]),
                        replaceItemUniqueId=text_id, physicsClientId=env.CLIENT)
            if i % 24 == 0:
                print(f"t={t:5.2f}s | Roll={np.rad2deg(rpy[0]):+8.3f}°"
                      f" | Pitch={np.rad2deg(rpy[1]):+8.3f}°"
                      f" | Yaw={np.rad2deg(rpy[2]):+8.3f}°"
                      f" | Z={pos[2]:8.3f}m"
                      f" | vx/vy/vz={world_vel[0]:+.6f}/{world_vel[1]:+.6f}/{world_vel[2]:+.6f}m/s"
                      f" | ax/ay/az={world_acc[0]:+.6f}/{world_acc[1]:+.6f}/{world_acc[2]:+.6f}m/s²"
                      f" | yaw_ref={np.rad2deg(yaw[4]):+.3f}deg/s"
                      f" | yaw_mode={yaw[7]} | saturated={saturated}")
            if REAL_TIME_PLAYBACK:
                remaining = t - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if terminated or truncated:
                break
    finally:
        env.close()
        # 저장 전에 열 수를 확인해, 헤더와 어긋난 CSV가 생성되지 않도록 한다.
        bad_rows = [(index, len(record)) for index, record in enumerate(log)
                    if len(record) != len(HEADER)]
        if bad_rows:
            raise ValueError(
                f"CSV 로그 열 개수 불일치: HEADER={len(HEADER)}개, "
                f"첫 오류 (행 번호, 열 개수)={bad_rows[:5]}"
            )
        log_path = LOG_DIR / f"roll_pitch_yaw_alt_pd_{run}.csv"
        with log_path.open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(log)
        print(f"원본 CSV: {log_path}")
    evaluate_and_save(np.asarray(log, dtype=float), run)
    print(f"로그 폴더: {LOG_DIR}")


if __name__ == "__main__":
    main()
