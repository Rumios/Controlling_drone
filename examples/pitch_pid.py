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
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" if Path(__file__).resolve().parent.name == "examples" else Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 5
SIM_FREQ = 240
INITIAL_PITCH = -45.0
TARGET_PITCH = 0.0

# P: 현재 각도 오차 / I: 누적 오차 / D: Pitch 회전 속도
KP = 0.00020
KI = 0.0
KD = 0.000095
INTEGRAL_LIMIT = 0.5  # rad*s
SETTLING_TOLERANCE = 0.5  # deg


def get_next_run_index():
    """CSV와 그래프 3개에 동일한 실행 번호를 부여하고 이전 결과를 보존한다."""
    index = 1
    while any((LOG_DIR / f"{name}_{index}.{extension}").exists() for name, extension in [
        ("drone_pid_control", "csv"),
        ("drone_pid_torque", "png"),
        ("drone_pid_integral", "png"),
        ("drone_pid_performance", "png"),
    ]):
        index += 1
    return index


def evaluate_performance(data):
    """Pitch 응답의 오버슈트, 정착 시간, 상승 시간 등을 계산한다."""
    t = data[:, 0]
    pitch = data[:, 2]
    rate = data[:, 5]
    distance = np.abs(pitch - TARGET_PITCH)
    initial_distance = abs(INITIAL_PITCH - TARGET_PITCH)

    # 상승 시간: 처음으로 초기 오차의 90%와 10%에 도달한 시점의 차이
    rise_time = np.nan
    if initial_distance > 0:
        start = np.where(distance <= 0.9 * initial_distance)[0]
        end = np.where(distance <= 0.1 * initial_distance)[0]
        if len(start) and len(end):
            after_start = end[end >= start[0]]
            if len(after_start):
                rise_time = t[after_start[0]] - t[start[0]]

    # 오버슈트: 처음 위치의 반대쪽으로 목표를 넘은 최대 각도
    if INITIAL_PITCH > TARGET_PITCH:
        overshoot_index = int(np.argmin(pitch))
        overshoot = max(0.0, TARGET_PITCH - pitch[overshoot_index])
    elif INITIAL_PITCH < TARGET_PITCH:
        overshoot_index = int(np.argmax(pitch))
        overshoot = max(0.0, pitch[overshoot_index] - TARGET_PITCH)
    else:
        overshoot_index, overshoot = 0, 0.0
    overshoot_percent = 100 * overshoot / initial_distance if initial_distance else 0.0

    # 정착 시간: 이후 '모든' 측정치가 ±허용 오차 안에 머무는 최초 시점
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
    }


def save_graphs(data, run_index):
    t = data[:, 0]
    prefix = lambda name: LOG_DIR / f"drone_pid_{name}_{run_index}.png"

    # 그래프 1: P/I/D/전체 토크를 하나의 그래프에서 비교
    plt.figure(figsize=(10, 5))
    for col, label in [(10, "P Torque"), (11, "I Torque"), (12, "D Torque")]:
        plt.plot(t, data[:, col], label=label)
    plt.plot(t, data[:, 13], label="Total Torque", linestyle="--")
    plt.axhline(0, linestyle=":", color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N*m)")
    plt.title("Pitch PID Control - Torque Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("torque"), dpi=200)

    # 그래프 2: 시간에 따른 누적 오차 (KI=0이어도 누적값은 기록됨)
    plt.figure(figsize=(10, 5))
    plt.plot(t, data[:, 9], label="Integral Error")
    plt.axhline(INTEGRAL_LIMIT, linestyle="--", label="Upper Limit")
    plt.axhline(-INTEGRAL_LIMIT, linestyle="--", label="Lower Limit")
    plt.axhline(0, linestyle=":", color="gray")
    plt.xlabel("Time (s)")
    plt.ylabel("Integral Error (rad*s)")
    plt.title("Pitch PID Control - Integral Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("integral"), dpi=200)

    # 그래프 3: 실제 Pitch, 오버슈트, 정착 시간 시각화
    metrics = evaluate_performance(data)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(t, data[:, 2], label="Actual Pitch", linewidth=2)
    ax.axhline(TARGET_PITCH, color="black", linestyle="--", label="Target Pitch")
    ax.axhspan(TARGET_PITCH - SETTLING_TOLERANCE,
               TARGET_PITCH + SETTLING_TOLERANCE,
               color="green", alpha=0.12,
               label=f"Settling Band (+/-{SETTLING_TOLERANCE} deg)")
    if metrics["overshoot"] > 0:
        j = metrics["overshoot_index"]
        ax.scatter(t[j], data[j, 2], color="red", s=60, zorder=5,
                   label="Maximum Overshoot")
    if metrics["settling_index"] is not None:
        ax.axvline(metrics["settling_time"], color="purple", linestyle=":",
                   label=f'Settling Time: {metrics["settling_time"]:.2f}s')
    summary = (
        f'Rise Time: {metrics["rise_time"]:.3f} s\n'
        f'Overshoot: {metrics["overshoot"]:.3f} deg '
        f'({metrics["overshoot_percent"]:.2f}%)\n'
        f'Settling Time: {metrics["settling_time"]:.3f} s\n'
        f'Final Error: {metrics["final_error"]:.5f} deg'
    )
    ax.text(0.98, 0.73, summary, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Pitch Control - Performance Evaluation")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(prefix("performance"), dpi=200)

    print("\n========== Pitch PID 성능 평가 ==========")
    print(f'상승 시간: {metrics["rise_time"]:.4f} s')
    print(f'최대 오버슈트: {metrics["overshoot"]:.4f} deg ({metrics["overshoot_percent"]:.2f}%)')
    print(f'정착 시간 (±{SETTLING_TOLERANCE}도): {metrics["settling_time"]:.4f} s')
    print(f'최대 각속도: {metrics["max_rate"]:.4f} deg/s')
    print(f'평균 절대 오차: {metrics["mae"]:.4f} deg')
    print(f'최종 오차: {metrics["final_error"]:.6f} deg')
    print("=========================================\n")
    plt.show()


def main():
    # 실행 시점에 번호 한 번만 정함: CSV와 그래프를 같은 번호로 묶는다.
    run_index = get_next_run_index()
    print(f"실험 번호: {run_index}")

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 100.0]]),
        initial_rpys=np.array([[0.0, np.deg2rad(INITIAL_PITCH), 0.0]]),
        pyb_freq=SIM_FREQ,
        ctrl_freq=SIM_FREQ,
        gui=True,
        user_debug_gui=False,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env.CLIENT)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8, cameraYaw=45, cameraPitch=-35,
        cameraTargetPosition=[0, 0, 1], physicsClientId=env.CLIENT,
    )

    # 중력과 균형을 이루는 총 호버링 추력을 모터 4개에 분배
    motor_hover_thrust = env.GRAVITY / 4
    hover_rpm = np.sqrt(motor_hover_thrust / env.KF)  # F = KF*RPM^2
    action = np.full((1, 4), hover_rpm)

    # X 배치 모터의 Pitch 회전축에 대한 유효 레버암
    arm = env.L / np.sqrt(2)
    max_motor_thrust = env.KF * env.MAX_RPM**2
    target_pitch = np.deg2rad(TARGET_PITCH)
    dt = 1.0 / SIM_FREQ
    integral_error = 0.0
    start_time = time.time()

    # CSV 열: 시간, RPY, 각속도 3개, Pitch 오차, 고도, 누적 오차, P/I/D/총 토크
    initial_p_torque = KP * np.deg2rad(TARGET_PITCH - INITIAL_PITCH)
    log_data = [[
        0.0, 0.0, INITIAL_PITCH, 0.0, 0.0, 0.0, 0.0,
        TARGET_PITCH - INITIAL_PITCH, 1.0, 0.0,
        initial_p_torque, 0.0, 0.0, initial_p_torque,
    ]]

    try:
        for i in range(int(DURATION * SIM_FREQ)):
            # 현재 모터 RPM으로 물리 시뮬레이션 1스텝 진행
            obs, reward, terminated, truncated, info = env.step(action)
            state = obs[0]
            position = state[0:3]
            roll, pitch, yaw = state[7:10]  # rad

            # 월드 기준 각속도를 드론 몸체 기준 각속도로 변환
            quaternion = state[3:7]
            rot = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            body_ang_vel = rot.T @ state[13:16]
            roll_rate, pitch_rate, yaw_rate = body_ang_vel  # rad/s

            # ============= PID 제어 =============
            # 오차 e(t) = 목표각 - 현재각 (rad)
            pitch_error = target_pitch - pitch

            # P: 현재 오차가 클수록 복귀 토크가 커짐
            p_torque = KP * pitch_error

            # I: (오차 * dt)를 매 스텝 더해 적분을 수치적으로 근사
            # 오랜 시간 남는 편차를 보정하지만 진동을 키울 수도 있음
            integral_error += pitch_error * dt
            integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
            i_torque = KI * integral_error

            # D: Pitch 각속도의 반대 방향으로 제동 토크를 만듦
            # 목표각이 고정이고 Pitch 위주 회전이므로 de/dt ≈ -pitch_rate
            d_torque = -KD * pitch_rate

            # 요구되는 전체 Pitch 토크 (N*m)
            pitch_torque = p_torque + i_torque + d_torque

            # ============= 모터 제어 =============
            # 4개 모터의 차등 추력으로 목표 Pitch 토크 구현
            thrust_delta = pitch_torque / (4 * arm)
            motor_thrust = np.array([
                motor_hover_thrust - thrust_delta,
                motor_hover_thrust + thrust_delta,
                motor_hover_thrust + thrust_delta,
                motor_hover_thrust - thrust_delta,
            ])
            motor_thrust = np.clip(motor_thrust, 0, max_motor_thrust)
            action = np.sqrt(motor_thrust / env.KF).reshape(1, 4)

            simulation_time = (i + 1) * dt
            log_data.append([
                simulation_time,
                np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw),
                np.rad2deg(roll_rate), np.rad2deg(pitch_rate), np.rad2deg(yaw_rate),
                np.rad2deg(pitch_error), position[2], integral_error,
                p_torque, i_torque, d_torque, pitch_torque,
            ])

            if i % 24 == 0:
                print(
                    f"시간: {simulation_time:.2f}s | "
                    f"RPY: {np.round(np.rad2deg([roll, pitch, yaw]), 3)}° | "
                    f"Pitch 각속도: {np.rad2deg(pitch_rate):.3f}°/s | "
                    f"높이: {position[2]:.3f}m | "
                    f"PID 토크: {pitch_torque:.8f} N*m"
                )

            # 카메라 위치는 4스텝마다 갱신 (240 / 4 = 60 Hz)
            if i % 4 == 0:
                camera_info = p.getDebugVisualizerCamera(physicsClientId=env.CLIENT)
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_info[10],
                    cameraYaw=camera_info[8],
                    cameraPitch=camera_info[9],
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

        # 기존 CSV를 덮어쓰지 않고 실행 번호에 맞춰 저장
        csv_path = LOG_DIR / f"drone_pid_control_{run_index}.csv"
        with open(csv_path, "x", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time (s)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)",
                "Roll Rate (deg/s)", "Pitch Rate (deg/s)", "Yaw Rate (deg/s)",
                "Pitch Error (deg)", "Altitude (m)", "Integral Error (rad*s)",
                "P Torque (N*m)", "I Torque (N*m)", "D Torque (N*m)",
                "Total Torque (N*m)",
            ])
            writer.writerows(log_data)
        print(f"CSV 저장 완료: {csv_path}")

    # 토크, 누적 오차, Pitch 성능 평가 그래프를 같은 실행 번호로 저장
    data = np.asarray(log_data, dtype=float)
    save_graphs(data, run_index)
    print(f"실험 {run_index} 저장 완료: {LOG_DIR}")


if __name__ == "__main__":
    main()
