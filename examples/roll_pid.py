"""CF2X Roll 단독 PD 제어 실험 (240 Hz)."""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# ==========================================
# 0. 실험 설정
# ==========================================
LOG_DIR = (
    Path(__file__).resolve().parent.parent / "logs"
    if Path(__file__).resolve().parent.name == "examples"
    else Path(__file__).resolve().parent / "logs"
)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 5.0
SIM_FREQ = 240
INITIAL_ROLL = -45.0       # 처음 기울어진 Roll 각도 (deg)
TARGET_ROLL = 0.0        # 복귀할 목표 Roll 각도 (deg)
INITIAL_PITCH = 0.0      # 이번 실험에서는 Pitch를 처음부터 0도로 둠

# 기존 Pitch 실험에서 검증한 이득을 Roll의 초기 후보값으로 재사용.
# Roll에 대해서는 아직 검증되지 않았으므로 응답을 보고 조정해야 함.
KP = 0.00020
KI = 0.0
KD = 0.000095
INTEGRAL_LIMIT = 0.5     # 적분 오차 제한 (rad*s); 현재 KI=0으로 I 토크 없음
SETTLING_TOLERANCE = 0.5 # 정착 판정 범위 (deg)

COLUMNS = [
    "Time (s)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)",
    "Roll Rate (deg/s)", "Pitch Rate (deg/s)", "Yaw Rate (deg/s)",
    "Roll Error (deg)", "Altitude (m)", "Integral Error (rad*s)",
    "P Torque (N*m)", "I Torque (N*m)", "D Torque (N*m)",
    "Total Torque (N*m)", "Motor Saturated",
]


def get_next_run_index():
    """Roll 실험 로그 1개와 그래프 3개에 공통 번호를 부여한다."""
    index = 1
    while any((LOG_DIR / f"roll_pd_{name}_{index}.{extension}").exists()
              for name, extension in [
                  ("control", "csv"),
                  ("torque", "png"),
                  ("integral", "png"),
                  ("performance", "png"),
              ]):
        index += 1
    return index


def calculate_roll_pid(roll, roll_rate, target_roll, integral_error, dt):
    """Roll 오차와 몸체 x축 각속도로 요구 Roll 토크(N*m)를 계산한다."""
    error = target_roll - roll
    p_torque = KP * error

    integral_error = np.clip(
        integral_error + error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT
    )
    i_torque = KI * integral_error

    # 목표 Roll이 고정된 단일 축 실험에서 각속도에 반대되는 제동 토크.
    d_torque = -KD * roll_rate
    total_torque = p_torque + i_torque + d_torque
    return error, integral_error, p_torque, i_torque, d_torque, total_torque


def allocate_roll_thrust(roll_torque, hover_thrust, arm, max_motor_thrust, kf):
    """CF2X 모터 순서 [M1,M2,M3,M4]에 Roll 토크를 차등 배분한다."""
    # CF2X에서 tau_roll = arm * (-F1 - F2 + F3 + F4).
    # 각 그룹의 두 모터에 +/-delta를 주면 tau_roll = 4*arm*delta.
    delta = roll_torque / (4.0 * arm)
    requested_thrust = hover_thrust + np.array([-delta, -delta, delta, delta])

    # 각 모터의 물리적인 추력 한계를 넘는지 확인하고 RPM 변환.
    saturated = bool(np.any((requested_thrust < 0.0) |
                            (requested_thrust > max_motor_thrust)))
    actual_thrust = np.clip(requested_thrust, 0.0, max_motor_thrust)
    rpm = np.sqrt(actual_thrust / kf).reshape(1, 4)
    return rpm, saturated


def evaluate_performance(data):
    """Roll 응답의 상승 시간(90%->10%), 오버슈트, 정착 시간 평가."""
    t = data[:, 0]
    roll = data[:, 1]
    rate = data[:, 4]
    distance = np.abs(roll - TARGET_ROLL)
    initial_distance = abs(INITIAL_ROLL - TARGET_ROLL)

    rise_time = np.nan
    if initial_distance > 0:
        start = np.where(distance <= 0.9 * initial_distance)[0]
        end = np.where(distance <= 0.1 * initial_distance)[0]
        if len(start) and len(end):
            after_start = end[end >= start[0]]
            if len(after_start):
                rise_time = t[after_start[0]] - t[start[0]]

    if INITIAL_ROLL > TARGET_ROLL:
        overshoot_index = int(np.argmin(roll))
        overshoot = max(0.0, TARGET_ROLL - roll[overshoot_index])
    elif INITIAL_ROLL < TARGET_ROLL:
        overshoot_index = int(np.argmax(roll))
        overshoot = max(0.0, roll[overshoot_index] - TARGET_ROLL)
    else:
        overshoot_index, overshoot = 0, 0.0

    overshoot_percent = 100 * overshoot / initial_distance if initial_distance else 0.0

    # 정착 시간은 기록이 끝날 때까지 허용 범위를 유지한 최초 시각.
    within = distance <= SETTLING_TOLERANCE
    settling_index = next((i for i in range(len(t)) if np.all(within[i:])), None)
    settling_time = t[settling_index] if settling_index is not None else np.nan

    return {
        "rise_time": rise_time,
        "overshoot": overshoot,
        "overshoot_percent": overshoot_percent,
        "overshoot_index": overshoot_index,
        "settling_time": settling_time,
        "settling_index": settling_index,
        "max_rate": np.max(np.abs(rate)),
        "mae": np.mean(distance),
        "final_error": distance[-1],
        "saturation_count": int(np.count_nonzero(data[:, 14])),
    }


def save_graphs(data, run_index):
    """기존 Pitch 실험과 동일한 세 종류의 그래프를 Roll에 대해 저장."""
    t = data[:, 0]
    prefix = lambda name: LOG_DIR / f"roll_pd_{name}_{run_index}.png"

    # 그래프 1: P, I, D와 총 Roll 토크.
    plt.figure(figsize=(10, 5))
    for col, label in [(10, "P Torque"), (11, "I Torque"), (12, "D Torque")]:
        plt.plot(t, data[:, col], label=label)
    plt.plot(t, data[:, 13], label="Total Roll Torque", linestyle="--")
    plt.axhline(0, linestyle=":", color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N*m)")
    plt.title("Roll PD Control - Torque Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("torque"), dpi=200)

    # 그래프 2: Roll 누적 오차. KI=0이어도 적분값을 기록하여 이후 I 실험에 대비.
    plt.figure(figsize=(10, 5))
    plt.plot(t, data[:, 9], label="Integral Error")
    plt.axhline(INTEGRAL_LIMIT, linestyle="--", label="Upper Limit")
    plt.axhline(-INTEGRAL_LIMIT, linestyle="--", label="Lower Limit")
    plt.axhline(0, linestyle=":", color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral Error (rad*s)")
    plt.title("Roll PD Control - Integral Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("integral"), dpi=200)

    # 그래프 3: Roll 응답과 정착 범위, 오버슈트, 정착 시간.
    metrics = evaluate_performance(data)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, data[:, 1], label="Actual Roll", linewidth=2)
    ax.axhline(TARGET_ROLL, color="black", linestyle="--", label="Target Roll")
    ax.axhspan(TARGET_ROLL - SETTLING_TOLERANCE,
               TARGET_ROLL + SETTLING_TOLERANCE,
               color="green", alpha=0.12,
               label=f"Settling Band (+/-{SETTLING_TOLERANCE} deg)")
    if metrics["overshoot"] > 0:
        j = metrics["overshoot_index"]
        ax.scatter(t[j], data[j, 1], color="red", s=60, zorder=5,
                   label="Maximum Overshoot")
    if metrics["settling_index"] is not None:
        ax.axvline(metrics["settling_time"], color="purple", linestyle=":",
                   label=f'Settling Time: {metrics["settling_time"]:.2f}s')
    summary = (
        f'Rise Time: {metrics["rise_time"]:.3f} s\n'
        f'Overshoot: {metrics["overshoot"]:.3f} deg '
        f'({metrics["overshoot_percent"]:.2f}%)\n'
        f'Settling Time: {metrics["settling_time"]:.3f} s\n'
        f'Final Error: {metrics["final_error"]:.5f} deg\n'
        f'Saturated Steps: {metrics["saturation_count"]}'
    )
    ax.text(0.98, 0.73, summary, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Roll (deg)")
    ax.set_title("Roll Control - Performance Evaluation")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("performance"), dpi=200)

    print("\n========== Roll PD 성능 평가 ==========")
    print(f'상승 시간: {metrics["rise_time"]:.4f} s')
    print(f'최대 오버슈트: {metrics["overshoot"]:.4f} deg '
          f'({metrics["overshoot_percent"]:.2f}%)')
    print(f'정착 시간 (±{SETTLING_TOLERANCE}도): {metrics["settling_time"]:.4f} s')
    print(f'최대 각속도: {metrics["max_rate"]:.4f} deg/s')
    print(f'평균 절대 오차: {metrics["mae"]:.4f} deg')
    print(f'최종 오차: {metrics["final_error"]:.6f} deg')
    print(f'모터 추력 포화 스텝: {metrics["saturation_count"]}')
    print("=====================================\n")
    plt.show()


def main():
    run_index = get_next_run_index()
    print(f"Roll 실험 번호: {run_index}")

    # Roll만 기울이고 Pitch/Yaw는 0도에서 시작한다.
    # 높이 100m는 고도 제어가 없는 동안 지면 충돌을 피하기 위한 실험 조건.
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 100.0]]),
        initial_rpys=np.array([[
            np.deg2rad(INITIAL_ROLL),
            np.deg2rad(INITIAL_PITCH),
            0.0,
        ]]),
        pyb_freq=SIM_FREQ,
        ctrl_freq=SIM_FREQ,
        gui=True,
        user_debug_gui=False,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.10, cameraYaw=45, cameraPitch=-35,
        cameraTargetPosition=[0, 0, 100], physicsClientId=env.CLIENT,
    )

    motor_hover_thrust = env.GRAVITY / 4.0
    hover_rpm = np.sqrt(motor_hover_thrust / env.KF)
    action = np.full((1, 4), hover_rpm)
    arm = env.L / np.sqrt(2.0)  # CF2X 모터의 Roll 축 유효 레버암
    max_motor_thrust = env.KF * env.MAX_RPM**2
    target_roll = np.deg2rad(TARGET_ROLL)
    dt = 1.0 / SIM_FREQ
    integral_error = 0.0
    start_time = time.time()

    # 초기 측정치를 첫 행으로 기록하고, 이후에는 240Hz로 제어 결과를 저장한다.
    initial_p_torque = KP * np.deg2rad(TARGET_ROLL - INITIAL_ROLL)
    log_data = [[
        0.0, INITIAL_ROLL, INITIAL_PITCH, 0.0,
        0.0, 0.0, 0.0,
        TARGET_ROLL - INITIAL_ROLL, 100.0, 0.0,
        initial_p_torque, 0.0, 0.0, initial_p_torque, 0.0,
    ]]

    try:
        for i in range(int(DURATION * SIM_FREQ)):
            # 현재 명령 RPM을 한 물리 스텝 동안 적용한 후 상태를 측정한다.
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs[0]
            position = state[0:3]
            roll, pitch, yaw = state[7:10]  # Euler RPY (rad)

            # 드론 상태의 월드 프레임 각속도를 몸체 프레임으로 변환한다.
            # Roll의 D항은 몸체 x축 각속도를 사용한다.
            quaternion = state[3:7]
            rot = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            body_ang_vel = rot.T @ state[13:16]
            roll_rate, pitch_rate, yaw_rate = body_ang_vel

            # Roll PD 제어기: 목표 각도를 향한 토크 + 회전 속도 감쇠 토크.
            (roll_error, integral_error, p_torque, i_torque,
             d_torque, roll_torque) = calculate_roll_pid(
                roll, roll_rate, target_roll, integral_error, dt
            )

            # Roll 토크를 4개 모터에 배분. Pitch/Yaw 보정은 적용하지 않는다.
            action, saturated = allocate_roll_thrust(
                roll_torque, motor_hover_thrust, arm, max_motor_thrust, env.KF
            )

            simulation_time = (i + 1) * dt
            log_data.append([
                simulation_time,
                np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw),
                np.rad2deg(roll_rate), np.rad2deg(pitch_rate), np.rad2deg(yaw_rate),
                np.rad2deg(roll_error), position[2], integral_error,
                p_torque, i_torque, d_torque, roll_torque, float(saturated),
            ])

            if i % 24 == 0:
                print(
                    f"시간: {simulation_time:.2f}s | "
                    f"RPY: {np.round(np.rad2deg([roll, pitch, yaw]), 3)}° | "
                    f"Roll 각속도: {np.rad2deg(roll_rate):.3f}°/s | "
                    f"높이: {position[2]:.3f}m | "
                    f"Roll 토크: {roll_torque:.8f} N*m | "
                    f"추력 포화: {saturated}"
                )

            # 카메라는 60Hz로 드론을 추적; 드론은 240Hz로 제어.
            # 카메라는 60Hz로 드론 위치를 추적하며 확대 거리 유지
            if i % 4 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.10,  # 매번 동일한 확대 거리 유지
                    cameraYaw=45,
                    cameraPitch=-35,
                    cameraTargetPosition=position,
                    physicsClientId=env.CLIENT,
                )

            env.render()
            elapsed = time.time() - start_time
            if simulation_time > elapsed:
                time.sleep(simulation_time - elapsed)
            if terminated or truncated:
                break
    finally:
        env.close()

        # 실행이 중간에 종료되어도 지금까지의 데이터를 고유한 이름으로 보존한다.
        csv_path = LOG_DIR / f"roll_pd_control_{run_index}.csv"
        with open(csv_path, "x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(log_data)
        print(f"CSV 저장 완료: {csv_path}")

    data = np.asarray(log_data, dtype=float)
    save_graphs(data, run_index)
    print(f"Roll 실험 {run_index} 저장 완료: {LOG_DIR}")


if __name__ == "__main__":
    main()
