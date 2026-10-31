"""CF2X Roll + Pitch 통합 PD 자세 제어 (PyBullet / 240 Hz).

실험: 초기 Roll=+40°, Pitch=+40° -> 두 축 0°.
제어기와 추력 배분을 직접 구현한다. 고도/Yaw/위치 제어는 미구현.
"""

import csv
import time
from pathlib import Path

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# =============== 1. 실험 및 제어 설정 ===============
SIM_FREQ = 240
DURATION = 30.0
INITIAL_EVAL_END = 5.0  # [0, 5] 초기 자세 복귀, (5, 30] 장시간 유지 평가
INITIAL_RPY_DEG = np.array([40.0, 40.0, 0.0])
TARGET_RPY_DEG = np.array([0.0, 0.0, 0.0])
INITIAL_ALTITUDE = 100.0   # 고도 제어가 없으므로 충돌 방지용 높은 초기값

# Roll/Pitch 단일 축에서 검증한 이득. 통합 제어 성능은 아직 검증되지 않았다.
KP = np.array([0.00020, 0.00020])
KD = np.array([0.000095, 0.000095])
SETTLING_TOL_DEG = 0.5

# UI 설정: 이전 실험에서 확인한 근접 카메라를 유지.
GUI = True
CAMERA_DISTANCE = 0.10
CAMERA_YAW = 45
CAMERA_PITCH = -35
SHOW_ATTITUDE_LABEL = True

# 튜닝 실험 때만 변경. 예: 1.25, 1.5 (안정성/포화 확인 필수)
GAIN_SCALE = 1.0
REAL_TIME_PLAYBACK = True

# 프로젝트 루트/logs 에 저장 (스크립트 위치: 프로젝트 루트/examples/pid.py)
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEADER = [
    "time_s", "roll_deg", "pitch_deg", "yaw_deg",
    "roll_rate_deg_s", "pitch_rate_deg_s", "yaw_rate_deg_s",
    "roll_error_deg", "pitch_error_deg", "altitude_m",
    "roll_p_torque_Nm", "roll_d_torque_Nm", "roll_torque_Nm",
    "pitch_p_torque_Nm", "pitch_d_torque_Nm", "pitch_torque_Nm",
    "motor_1_rpm", "motor_2_rpm", "motor_3_rpm", "motor_4_rpm",
    "motor_saturated",
]


def new_run_index():
    index = 1
    while (LOG_DIR / f"roll_pitch_pd_{index}.csv").exists():
        index += 1
    return index


def measure_state(state):
    """관측 벡터에서 Euler 자세와 몸체 좌표계 각속도를 읽는다.

    state[13:16]은 월드 프레임 각속도이다. 회전행렬 전치로 몸체축 변환.
    """
    rpy = state[7:10]
    rot = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
    body_rates = rot.T @ state[13:16]
    return np.asarray(state[:3]), rpy, body_rates


def attitude_pd(rpy, body_rates, target_rpy):
    """Roll/Pitch 목표 토크를 각각 계산한다. Yaw는 제어하지 않는다."""
    err = target_rpy[:2] - rpy[:2]
    proportional = (KP * GAIN_SCALE) * err
    derivative = -(KD * GAIN_SCALE) * body_rates[:2]
    torque = proportional + derivative
    return err, proportional, derivative, torque


def allocate_motor_rpm(roll_torque, pitch_torque, hover_force, arm, kf,
                       max_motor_thrust):
    """CF2X의 [M1,M2,M3,M4] 순서에 Roll/Pitch 토크를 합산한다.

    tau_roll  = arm * (-F1-F2+F3+F4)
    tau_pitch = arm * (-F1+F2+F3-F4)
    모터군 차등 추력 delta=tau/(4*arm); 총 기본 추력은 유지.
    """
    roll_delta = roll_torque / (4.0 * arm)
    pitch_delta = pitch_torque / (4.0 * arm)

    thrust = (np.full(4, hover_force)
              + roll_delta * np.array([-1., -1., 1., 1.])
              + pitch_delta * np.array([-1., 1., 1., -1.]))
    saturated = bool(np.any((thrust < 0.0) | (thrust > max_motor_thrust)))
    thrust = np.clip(thrust, 0.0, max_motor_thrust)
    rpm = np.sqrt(thrust / kf)
    return rpm.reshape(1, 4), saturated


def settling_time(t, angles_deg, target_deg, tol_deg):
    """두 축이 동시에 오차 허용범위를 계속 만족하는 최초 시각."""
    ok = np.all(np.abs(angles_deg - target_deg) <= tol_deg, axis=1)
    indices = np.flatnonzero(~ok)
    first = int(indices[-1] + 1) if len(indices) else 0
    return float(t[first]) if first < len(t) else float("nan")


def rise_time(t, angle_deg, target_deg):
    """초기 오차의 90% -> 10%까지 감소한 시간 (기존 단일 축 정의)."""
    initial_error = abs(angle_deg[0] - target_deg)
    if initial_error <= 0:
        return float("nan")
    distance = np.abs(angle_deg - target_deg)
    a = np.flatnonzero(distance <= 0.9 * initial_error)
    b = np.flatnonzero(distance <= 0.1 * initial_error)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    valid_b = b[b >= a[0]]
    return float(t[valid_b[0]] - t[a[0]]) if len(valid_b) else float("nan")


def evaluate_and_save(data, index):
    """초기 복귀와 장시간 자세 유지를 별도로 평가하고 txt+CSV로 보존."""
    initial_mask = data[:, 0] <= INITIAL_EVAL_END + 1e-9
    hold_mask = data[:, 0] > INITIAL_EVAL_END + 1e-9
    windows = [
        ("initial_recovery", "초기 복귀", data[initial_mask]),
        ("long_term_hold", "장시간 유지", data[hold_mask]),
        ("full_run", "전체 실행", data),
    ]
    result_rows = []
    report = [
        "======= Roll + Pitch 통합 PD 구간별 평가 =======",
        f"초기 자세: {INITIAL_RPY_DEG.tolist()} deg / 목표: {TARGET_RPY_DEG.tolist()} deg",
        f"제어 주파수: {SIM_FREQ}Hz / KP={KP.tolist()} / KD={KD.tolist()} / gain_scale={GAIN_SCALE}",
        f"평가 구간: 초기 0~{INITIAL_EVAL_END:.2f}s, 유지 {INITIAL_EVAL_END:.2f}~{DURATION:.2f}s",
        f"정착 허용 오차: 각 축 ±{SETTLING_TOL_DEG}deg",
        "주의: 초기 정착 시간은 초기 구간 안에서 지속되는 경우만 인정한다.",
        "      이후 구간의 정착 시간은 초기 복귀 성능에 포함하지 않는다.",
    ]

    for key, label, chunk in windows:
        if len(chunk) == 0:
            report.append(f"\n[{label}] 데이터 없음")
            continue
        t = chunk[:, 0]
        ang = chunk[:, 1:3]
        rates = chunk[:, 4:6]
        err = np.abs(ang - TARGET_RPY_DEG[:2])
        within = np.all(err <= SETTLING_TOL_DEG, axis=1)
        jointly_settled = settling_time(t, ang, TARGET_RPY_DEG[:2], SETTLING_TOL_DEG)
        report.append(f"\n[{label}] t={t[0]:.4f}~{t[-1]:.4f}s, {len(chunk)} samples")

        for axis, j in (("Roll", 0), ("Pitch", 1)):
            first_entry = np.flatnonzero(err[:, j] <= SETTLING_TOL_DEG)
            first_entry_t = float(t[first_entry[0]]) if len(first_entry) else float('nan')
            axis_settle = settling_time(t, ang[:, j:j+1], TARGET_RPY_DEG[j:j+1], SETTLING_TOL_DEG)
            max_rate_i = int(np.argmax(np.abs(rates[:, j])))
            metrics = {
                "rise_time_s": rise_time(t, ang[:, j], TARGET_RPY_DEG[j]) if key == "initial_recovery" else float('nan'),
                "first_entry_s": first_entry_t,
                "settling_time_s": axis_settle,
                "max_abs_rate_deg_s": float(abs(rates[max_rate_i, j])),
                "max_rate_at_s": float(t[max_rate_i]),
                "mae_deg": float(np.mean(err[:, j])),
                "max_abs_error_deg": float(np.max(err[:, j])),
                "final_error_deg": float(err[-1, j]),
            }
            result_rows.append({"section": key, "axis": axis, **metrics})
            report.append(
                f" {axis:5s} | 최초 ±{SETTLING_TOL_DEG}deg 진입={first_entry_t:.4f}s"
                f" | 정착={axis_settle:.4f}s | 최대 각속도={metrics['max_abs_rate_deg_s']:.3f}deg/s"
                f" (t={metrics['max_rate_at_s']:.4f}s)"
                f" | MAE={metrics['mae_deg']:.4f}deg | 최대 오차={metrics['max_abs_error_deg']:.4f}deg"
                f" | 최종 오차={metrics['final_error_deg']:.6f}deg"
            )
            if key == "initial_recovery":
                report.append(f"         상승 시간={metrics['rise_time_s']:.4f}s")
        report.append(
            f" 동시 정착={jointly_settled:.4f}s"
            f" | 양축 허용오차 이탈 샘플={int(np.count_nonzero(~within))}/{len(chunk)}"
            f" | 모터 포화={int(np.count_nonzero(chunk[:, -1]))}스텝"
            f" | 고도 변화={chunk[-1, 9]-chunk[0, 9]:+.4f}m"
            f" | Yaw 변화={chunk[-1, 3]-chunk[0, 3]:+.4f}deg"
        )

    report += ["", "해석: 강제 외란을 주지 않아도 장시간 유지 구간에서 큰 각속도가", "      발생하면 자세/각속도/모터 RPM의 해당 시점 로그를 조사해야 한다.", "==========================================="]
    text_path = LOG_DIR / f"roll_pitch_eval_{index}.txt"
    csv_path = LOG_DIR / f"roll_pitch_eval_{index}.csv"
    text_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if result_rows:
        with csv_path.open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result_rows[0].keys()))
            writer.writeheader()
            writer.writerows(result_rows)
    print("\n".join(report))
    print(f"평가 TXT 저장: {text_path}")
    print(f"평가 CSV 저장: {csv_path}")


def main():
    run = new_run_index()
    target_rpy = np.deg2rad(TARGET_RPY_DEG)
    initial_rpy = np.deg2rad(INITIAL_RPY_DEG)
    dt = 1.0 / SIM_FREQ

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, INITIAL_ALTITUDE]]),
        initial_rpys=initial_rpy.reshape(1, 3),
        pyb_freq=SIM_FREQ,
        ctrl_freq=SIM_FREQ,
        gui=GUI,
        user_debug_gui=False,
    )

    hover_motor_force = env.GRAVITY / 4.0
    arm = env.L / np.sqrt(2.0)
    max_motor_thrust = env.KF * env.MAX_RPM ** 2
    action = np.full((1, 4), np.sqrt(hover_motor_force / env.KF))
    first_error = target_rpy[:2] - initial_rpy[:2]
    initial_p = (KP * GAIN_SCALE) * first_error

    # 첫 표본을 포함해야 성능 계산이 최초 자세를 정확하게 기준으로 삼는다.
    log = [[0.0, *INITIAL_RPY_DEG, 0.0, 0.0, 0.0,
            *(TARGET_RPY_DEG[:2] - INITIAL_RPY_DEG[:2]), INITIAL_ALTITUDE,
            initial_p[0], 0., initial_p[0],
            initial_p[1], 0., initial_p[1],
            *action[0], 0.]]

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
            state = obs[0]
            position, rpy, body_rates = measure_state(state)
            error, p_torque, d_torque, torque = attitude_pd(rpy, body_rates, target_rpy)
            action, saturated = allocate_motor_rpm(
                torque[0], torque[1], hover_motor_force,
                arm, env.KF, max_motor_thrust,
            )
            sim_t = (i + 1) * dt
            log.append([
                sim_t, *np.rad2deg(rpy), *np.rad2deg(body_rates),
                *np.rad2deg(error), position[2],
                p_torque[0], d_torque[0], torque[0],
                p_torque[1], d_torque[1], torque[1],
                *action[0], float(saturated),
            ])

            # 60Hz 추적: 확인된 가까운 거리에서 드론 Roll/Pitch 자세를 관찰.
            if GUI and i % 4 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=CAMERA_DISTANCE,
                    cameraYaw=CAMERA_YAW,
                    cameraPitch=CAMERA_PITCH,
                    cameraTargetPosition=position,
                    physicsClientId=env.CLIENT,
                )
                if SHOW_ATTITUDE_LABEL:
                    # GUI 내 드론 위에 현재 Roll/Pitch 값을 표시한다.
                    text_id = p.addUserDebugText(
                        text=f"Roll {np.rad2deg(rpy[0]):+.1f} deg | Pitch {np.rad2deg(rpy[1]):+.1f} deg",
                        textPosition=[0.0, 0.0, 0.10],
                        textColorRGB=[0, 0, 0],
                        textSize=1.1,
                        parentObjectUniqueId=int(env.DRONE_IDS[0]),
                        replaceItemUniqueId=text_id,
                        physicsClientId=env.CLIENT,
                    )

            if i % 24 == 0:
                print(f"t={sim_t:.2f}s | Roll={np.rad2deg(rpy[0]):+.3f}°"
                      f" | Pitch={np.rad2deg(rpy[1]):+.3f}°"
                      f" | Yaw={np.rad2deg(rpy[2]):+.3f}°"
                      f" | Z={position[2]:.3f}m | saturated={saturated}")
            if REAL_TIME_PLAYBACK:
                remaining = sim_t - (time.time() - start)
                if remaining > 0:
                    time.sleep(remaining)
            if terminated or truncated:
                break
    finally:
        env.close()
        path = LOG_DIR / f"roll_pitch_pd_{run}.csv"
        with path.open("x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerows(log)
        print(f"CSV 저장: {path}")

    evaluate_and_save(np.asarray(log, dtype=float), run)
    print(f"로그 폴더: {LOG_DIR}")


if __name__ == "__main__":
    main()
