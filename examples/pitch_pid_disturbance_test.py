
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# ============================================================
# 1. 실험 설정
# 이 부분만 수정하면 대부분의 실험 조건을 변경할 수 있음
# ============================================================

@dataclass(frozen=True)
class Config:
    # 시뮬레이션
    DURATION: float = 7.0
    SIM_FREQ: int = 240
    GUI: bool = True
    REAL_TIME: bool = True

    # 초기 상태
    INITIAL_ALTITUDE: float = 100.0
    INITIAL_PITCH: float = 0.0
    TARGET_PITCH: float = 0.0

    # 기존 실험에서 선정한 PID 계수
    KP: float = 0.00020
    KI: float = 0.0
    KD: float = 0.000095

    # 적분항 제한 (Anti-windup)
    INTEGRAL_LIMIT: float = 0.5

    # 정착 판정 기준
    SETTLING_TOLERANCE: float = 0.5

    # 외란 설정
    DISTURBANCE_ENABLED: bool = True
    DISTURBANCE_START: float = 3.0
    DISTURBANCE_DURATION: float = 0.1

    # Y축(Pitch)에 가할 외부 토크 (N*m)
    DISTURBANCE_TORQUE: float = 0.0001

    # GUI 및 로그
    PRINT_INTERVAL: float = 0.1
    CAMERA_UPDATE_HZ: int = 60
    SHOW_GRAPHS: bool = True


CFG = Config()

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = (
    BASE_DIR.parent / "logs"
    if BASE_DIR.name == "examples"
    else BASE_DIR / "logs"
)

CSV_FIELDS = [
    "Time (s)",
    "Roll (deg)",
    "Pitch (deg)",
    "Yaw (deg)",
    "Roll Rate (deg/s)",
    "Pitch Rate (deg/s)",
    "Yaw Rate (deg/s)",
    "Pitch Error (deg)",
    "Altitude (m)",
    "Integral Error (rad*s)",
    "P Torque (N*m)",
    "I Torque (N*m)",
    "D Torque (N*m)",
    "Total Torque (N*m)",
    "Disturbance Torque (N*m)",
]


# ============================================================
# 2. 결과 파일 관리
# 이전 CSV 및 그래프를 덮어쓰지 않고 실행 번호 부여
# ============================================================

def get_next_run_index():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    names = [
        "control",
        "torque",
        "integral",
        "performance",
    ]

    index = 1

    while True:
        files = [
            LOG_DIR / f"drone_pid_{name}_{index}.png"
            for name in names[1:]
        ]

        files.append(
            LOG_DIR / f"drone_pid_control_{index}.csv"
        )

        if not any(path.exists() for path in files):
            return index

        index += 1


# ============================================================
# 3. 드론 상태 읽기
# PyBullet에서 실제 위치·자세·각속도를 직접 가져옴
# ============================================================

def read_state(env):
    drone_id = int(env.DRONE_IDS[0])

    position, quaternion = p.getBasePositionAndOrientation(
        drone_id,
        physicsClientId=env.CLIENT,
    )

    _, world_ang_vel = p.getBaseVelocity(
        drone_id,
        physicsClientId=env.CLIENT,
    )

    # Quaternion -> Euler (Roll, Pitch, Yaw)
    roll, pitch, yaw = p.getEulerFromQuaternion(quaternion)

    # 월드 좌표계 각속도 -> 드론 본체 좌표계 각속도
    rotation = np.array(
        p.getMatrixFromQuaternion(quaternion)
    ).reshape(3, 3)

    body_ang_vel = rotation.T @ np.asarray(world_ang_vel)

    return {
        "position": np.asarray(position),
        "rpy": np.array([roll, pitch, yaw]),
        "body_rates": body_ang_vel,
    }


# ============================================================
# 4. Pitch PID 제어기
# 입력: 현재 Pitch, Pitch 각속도
# 출력: 요구 Pitch 토크
# ============================================================

class PitchPID:
    def __init__(self, config):
        self.cfg = config
        self.integral_error = 0.0

    def update(self, pitch, pitch_rate, dt):
        target = np.deg2rad(self.cfg.TARGET_PITCH)

        # e(t) = 목표각 - 현재각
        error = target - pitch

        # P 제어: 현재 오차에 비례하는 복원 토크
        p_torque = self.cfg.KP * error

        # I 제어: 누적 오차 보정
        # KI=0인 현재 실험에서는 실제 토크에 영향 없음
        if self.cfg.KI != 0.0:
            self.integral_error += error * dt
            self.integral_error = float(
                np.clip(
                    self.integral_error,
                    -self.cfg.INTEGRAL_LIMIT,
                    self.cfg.INTEGRAL_LIMIT,
                )
            )

        i_torque = self.cfg.KI * self.integral_error

        # D 제어: 각속도 반대 방향으로 제동
        # 목표각이 일정하므로 D = -KD * pitch_rate
        d_torque = -self.cfg.KD * pitch_rate

        total_torque = p_torque + i_torque + d_torque

        return {
            "error": error,
            "p": p_torque,
            "i": i_torque,
            "d": d_torque,
            "total": total_torque,
            "integral": self.integral_error,
        }


# ============================================================
# 5. Pitch 토크 -> 모터 RPM
# CF2X 모터 배열에 맞춰 차등 추력을 계산
# ============================================================

def torque_to_rpm(env, pitch_torque):
    # 중력과 균형을 이루는 모터 1개당 추력
    hover_thrust = env.GRAVITY / 4.0

    # X자 배치에서 Pitch 축에 대한 유효 레버암
    arm = env.L / np.sqrt(2.0)

    # Pitch 토크를 모터 4개의 추력 차이로 변환
    thrust_delta = pitch_torque / (4.0 * arm)

    motor_thrust = np.array([
        hover_thrust - thrust_delta,
        hover_thrust + thrust_delta,
        hover_thrust + thrust_delta,
        hover_thrust - thrust_delta,
    ])

    # 모터가 낼 수 있는 추력 범위로 제한
    max_thrust = env.KF * env.MAX_RPM**2

    motor_thrust = np.clip(
        motor_thrust,
        0.0,
        max_thrust,
    )

    # F = KF * RPM^2
    rpm = np.sqrt(motor_thrust / env.KF)

    # CtrlAviary: (드론 수, 모터 수)
    return rpm.reshape(1, 4)


# ============================================================
# 6. 외란 발생기
# 설정된 구간에만 드론 본체에 외부 토크 적용
# ============================================================

def apply_disturbance(env, sim_time, config):
    if not config.DISTURBANCE_ENABLED:
        return 0.0

    start = config.DISTURBANCE_START
    end = start + config.DISTURBANCE_DURATION

    # 정확히 [start, end) 구간에만 적용
    if not (start <= sim_time < end):
        return 0.0

    torque = config.DISTURBANCE_TORQUE

    p.applyExternalTorque(
        objectUniqueId=int(env.DRONE_IDS[0]),
        linkIndex=-1,
        torqueObj=[0.0, torque, 0.0],
        flags=p.LINK_FRAME,
        physicsClientId=env.CLIENT,
    )

    return torque


# ============================================================
# 7. 카메라 갱신
# 드론을 따라가되 사용자 카메라 회전·줌 값 유지
# ============================================================

def update_camera(env, position):
    camera = p.getDebugVisualizerCamera(
        physicsClientId=env.CLIENT
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=camera[10],
        cameraYaw=camera[8],
        cameraPitch=camera[9],
        cameraTargetPosition=position,
        physicsClientId=env.CLIENT,
    )


# ============================================================
# 8. 로그 생성
# 실제 측정 상태와 PID 출력, 외란 값을 저장
# ============================================================

def make_log_row(sim_time, state, pid, disturbance):
    roll, pitch, yaw = state["rpy"]
    roll_rate, pitch_rate, yaw_rate = state["body_rates"]

    return [
        sim_time,
        np.rad2deg(roll),
        np.rad2deg(pitch),
        np.rad2deg(yaw),
        np.rad2deg(roll_rate),
        np.rad2deg(pitch_rate),
        np.rad2deg(yaw_rate),
        np.rad2deg(pid["error"]),
        state["position"][2],
        pid["integral"],
        pid["p"],
        pid["i"],
        pid["d"],
        pid["total"],
        disturbance,
    ]


def save_csv(rows, run_index):
    path = LOG_DIR / f"drone_pid_control_{run_index}.csv"

    # "x" 모드: 기존 파일이 있으면 덮어쓰지 않음
    with path.open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)

    print(f"CSV 저장: {path}")


# ============================================================
# 9. 성능 평가
# 초기 자세 복귀와 외란 복구를 구분하여 계산
# ============================================================

def permanent_settling_time(t, error, tolerance):
    """
    이후 모든 기록이 허용 범위 안에 있는 최초 시점.
    관측 종료까지의 정착 판정이므로 장기 안정성을 보장하지는 않음.
    """
    inside = np.abs(error) <= tolerance

    # 뒤에서 앞으로 검사하여 마지막 범위 이탈 위치 검색
    outside_indices = np.flatnonzero(~inside)

    if len(outside_indices) == 0:
        return float(t[0])

    last_outside = int(outside_indices[-1])

    if last_outside == len(t) - 1:
        return np.nan

    return float(t[last_outside + 1])


def evaluate_performance(data, config):
    t = data[:, 0]
    pitch = data[:, 2]
    rate = data[:, 5]

    error = pitch - config.TARGET_PITCH
    tolerance = config.SETTLING_TOLERANCE

    results = {
        "max_rate": float(np.max(np.abs(rate))),
        "mae": float(np.mean(np.abs(error))),
        "final_error": float(abs(error[-1])),
    }

    # 초기 각도가 목표와 다른 경우: 기존 복귀 실험 평가
    initial_error = config.INITIAL_PITCH - config.TARGET_PITCH

    if abs(initial_error) > 1e-9:
        distance = np.abs(error)

        start = np.flatnonzero(
            distance <= 0.9 * abs(initial_error)
        )
        end = np.flatnonzero(
            distance <= 0.1 * abs(initial_error)
        )

        rise_time = np.nan

        if len(start) and len(end):
            later = end[end >= start[0]]

            if len(later):
                rise_time = float(
                    t[later[0]] - t[start[0]]
                )

        # 초기 각도의 반대쪽으로 목표를 지나친 정도
        if initial_error > 0:
            overshoot = max(0.0, -float(np.min(error)))
        else:
            overshoot = max(0.0, float(np.max(error)))

        results.update({
            "mode": "initial",
            "rise_time": rise_time,
            "overshoot": overshoot,
            "overshoot_percent": (
                100 * overshoot / abs(initial_error)
            ),
            "settling_time": permanent_settling_time(
                t, error, tolerance
            ),
        })

    # 초기 각도가 0°인 경우: 외란을 받은 후 복구 평가
    else:
        if not config.DISTURBANCE_ENABLED:
            results["mode"] = "hold"
            return results

        disturbance_end = (
            config.DISTURBANCE_START
            + config.DISTURBANCE_DURATION
        )

        # 외란 시작부터의 응답
        affected = t >= config.DISTURBANCE_START

        # 외란 제거 이후의 응답
        recovery = t >= disturbance_end

        max_deviation = float(
            np.max(np.abs(error[affected]))
        ) if np.any(affected) else np.nan

        recovery_time = np.nan

        if np.any(recovery):
            # 외란 제거 이후 안정화에 걸린 시간
            settled_at = permanent_settling_time(
                t[recovery],
                error[recovery],
                tolerance,
            )

            if np.isfinite(settled_at):
                recovery_time = max(
                    0.0, settled_at - disturbance_end
                )

        results.update({
            "mode": "disturbance",
            "max_deviation": max_deviation,
            "recovery_time": recovery_time,
        })

    return results


def print_metrics(metrics):
    print("\n========== Pitch PD 성능 평가 ==========")

    mode = metrics["mode"]

    if mode == "initial":
        print(f'상승 시간: {metrics["rise_time"]:.4f} s')
        print(f'최대 오버슈트: {metrics["overshoot"]:.4f} deg')
        print(
            f'오버슈트율: {metrics["overshoot_percent"]:.3f}%'
        )
        print(
            f'정착 시간 (±{CFG.SETTLING_TOLERANCE}도): '
            f'{metrics["settling_time"]:.4f} s'
        )

    elif mode == "disturbance":
        print(
            f'외란 후 최대 Pitch 편차: '
            f'{metrics["max_deviation"]:.4f} deg'
        )
        print(
            f'외란 제거 후 복구 시간 '
            f'(±{CFG.SETTLING_TOLERANCE}도): '
            f'{metrics["recovery_time"]:.4f} s'
        )

    else:
        print("초기 목표 각도 유지 실험")

    print(f'최대 각속도: {metrics["max_rate"]:.4f} deg/s')
    print(f'평균 절대 오차: {metrics["mae"]:.4f} deg')
    print(f'최종 오차: {metrics["final_error"]:.6f} deg')
    print("========================================\n")


# ============================================================
# 10. 그래프 생성
# ============================================================

def save_graphs(data, run_index, config):
    t = data[:, 0]
    metrics = evaluate_performance(data, config)

    def output_path(name):
        return LOG_DIR / f"drone_pid_{name}_{run_index}.png"

    def add_disturbance_band(ax):
        if config.DISTURBANCE_ENABLED:
            ax.axvspan(
                config.DISTURBANCE_START,
                config.DISTURBANCE_START
                + config.DISTURBANCE_DURATION,
                alpha=0.15,
                color="orange",
                label="External disturbance",
            )

    # 그래프 1: P / I / D / 전체 토크 / 외란 토크
    fig, ax = plt.subplots(figsize=(11, 5))

    for column, label in [
        (10, "P Torque"),
        (11, "I Torque"),
        (12, "D Torque"),
        (13, "Control Torque"),
        (14, "External Torque"),
    ]:
        ax.plot(t, data[:, column], label=label)

    add_disturbance_band(ax)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_title("Pitch PD - Torque Components")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N*m)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path("torque"), dpi=200)

    # 그래프 2: 누적 오차
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, data[:, 9], label="Integral Error")
    ax.axhline(
        config.INTEGRAL_LIMIT,
        color="red",
        linestyle="--",
        label="Upper Limit",
    )
    ax.axhline(
        -config.INTEGRAL_LIMIT,
        color="red",
        linestyle="--",
        label="Lower Limit",
    )

    add_disturbance_band(ax)
    ax.set_title("Pitch PD - Integral Error")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Integral Error (rad*s)")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path("integral"), dpi=200)

    # 그래프 3: Pitch 응답 및 외란 복구
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        t,
        data[:, 2],
        linewidth=2,
        label="Actual Pitch",
    )

    ax.axhline(
        config.TARGET_PITCH,
        color="black",
        linestyle="--",
        label="Target Pitch",
    )

    ax.axhspan(
        config.TARGET_PITCH - config.SETTLING_TOLERANCE,
        config.TARGET_PITCH + config.SETTLING_TOLERANCE,
        color="green",
        alpha=0.12,
        label="Settling Band",
    )

    add_disturbance_band(ax)

    # 외란 복구 시간 표시
    if metrics["mode"] == "disturbance":
        recovery_time = metrics["recovery_time"]

        if np.isfinite(recovery_time):
            recovered_at = (
                config.DISTURBANCE_START
                + config.DISTURBANCE_DURATION
                + recovery_time
            )

            ax.axvline(
                recovered_at,
                color="purple",
                linestyle=":",
                label=f"Recovered at {recovered_at:.3f}s",
            )

        description = (
            f'Max Deviation: {metrics["max_deviation"]:.3f} deg\n'
            f'Recovery Time: {recovery_time:.3f} s\n'
            f'Final Error: {metrics["final_error"]:.6f} deg'
        )

    elif metrics["mode"] == "initial":
        description = (
            f'Rise Time: {metrics["rise_time"]:.3f} s\n'
            f'Overshoot: {metrics["overshoot"]:.3f} deg\n'
            f'Settling Time: {metrics["settling_time"]:.3f} s'
        )

    else:
        description = (
            f'Final Error: {metrics["final_error"]:.6f} deg'
        )

    ax.text(
        0.98,
        0.72,
        description,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

    ax.set_title("Pitch PD - Disturbance Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (deg)")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path("performance"), dpi=200)

    print_metrics(metrics)

    if config.SHOW_GRAPHS:
        plt.show()

    plt.close("all")


# ============================================================
# 11. 메인 시뮬레이션
# ============================================================

def main():
    config = CFG

    if config.SIM_FREQ <= 0 or config.DURATION <= 0:
        raise ValueError("주파수와 실험 시간은 양수여야 합니다.")

    if config.DISTURBANCE_ENABLED:
        end = (
            config.DISTURBANCE_START
            + config.DISTURBANCE_DURATION
        )
        if config.DISTURBANCE_START < 0 or end > config.DURATION:
            raise ValueError("외란 구간이 시뮬레이션 범위를 벗어납니다.")

    run_index = get_next_run_index()
    dt = 1.0 / config.SIM_FREQ

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([
            [0.0, 0.0, config.INITIAL_ALTITUDE]
        ]),
        initial_rpys=np.array([
            [0.0, np.deg2rad(config.INITIAL_PITCH), 0.0]
        ]),
        pyb_freq=config.SIM_FREQ,
        ctrl_freq=config.SIM_FREQ,
        gui=config.GUI,
        user_debug_gui=False,
    )

    controller = PitchPID(config)
    log_rows = []

    # CtrlAviary의 모터 모델에서 호버링 RPM 계산
    hover_rpm = np.sqrt((env.GRAVITY / 4.0) / env.KF)
    action = np.full((1, 4), hover_rpm)

    if config.GUI:
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            0,
            physicsClientId=env.CLIENT,
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=0.8,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[
                0, 0, config.INITIAL_ALTITUDE
            ],
            physicsClientId=env.CLIENT,
        )

    print(f"\n실험 번호: {run_index}")
    print(
        f"초기 Pitch: {config.INITIAL_PITCH}° | "
        f"목표 Pitch: {config.TARGET_PITCH}°"
    )
    print(
        f"KP={config.KP}, KI={config.KI}, KD={config.KD}"
    )
    print(
        f"외란: {config.DISTURBANCE_ENABLED} | "
        f"토크: {config.DISTURBANCE_TORQUE} N*m"
    )

    wall_start = time.perf_counter()

    print_steps = max(
        1,
        round(config.PRINT_INTERVAL * config.SIM_FREQ),
    )

    camera_steps = max(
        1,
        round(config.SIM_FREQ / config.CAMERA_UPDATE_HZ),
    )

    try:
        # 실제 물리 엔진에서 초기 상태 조회
        state = read_state(env)

        # 첫 시점 제어 출력 계산
        pid = controller.update(
            state["rpy"][1],
            state["body_rates"][1],
            0.0,
        )

        # 0초 로그: 임의의 고도·각도를 기록하지 않음
        log_rows.append(
            make_log_row(0.0, state, pid, 0.0)
        )

        steps = round(config.DURATION * config.SIM_FREQ)

        for i in range(steps):
            sim_time = i * dt

            # (1) 현재 상태에서 계산한 제어 토크를 RPM으로 변환
            action = torque_to_rpm(env, pid["total"])

            # (2) 지정된 시점에 외란 토크 추가
            # PyBullet의 다음 물리 스텝에 외부 토크가 반영됨
            external_torque = apply_disturbance(
                env,
                sim_time,
                config,
            )

            # (3) 물리 시뮬레이션 진행
            # pyb_freq == ctrl_freq이므로 한 호출에 물리 1스텝
            obs, reward, terminated, truncated, info = env.step(
                action
            )

            # (4) 새 상태 측정
            state = read_state(env)

            next_time = (i + 1) * dt

            # (5) 방금 사용한 제어 출력과 외란을 함께 기록
            log_rows.append(
                make_log_row(
                    next_time,
                    state,
                    pid,
                    external_torque,
                )
            )

            # (6) 다음 스텝에 사용할 PID 출력 계산
            pid = controller.update(
                state["rpy"][1],
                state["body_rates"][1],
                dt,
            )

            # (7) 콘솔 로그
            if i % print_steps == 0:
                pitch_deg = np.rad2deg(state["rpy"][1])
                pitch_rate_deg = np.rad2deg(
                    state["body_rates"][1]
                )

                print(
                    f"t={next_time:5.2f}s | "
                    f"Pitch={pitch_deg:9.4f}° | "
                    f"Rate={pitch_rate_deg:9.3f}°/s | "
                    f"Altitude={state['position'][2]:8.3f}m | "
                    f"External={external_torque:.7f} N*m"
                )

            # (8) 카메라 갱신
            if config.GUI and i % camera_steps == 0:
                update_camera(env, state["position"])

            if config.GUI:
                env.render()

            # (9) 실제 시간과 시뮬레이션 시간 동기화
            if config.REAL_TIME:
                elapsed = time.perf_counter() - wall_start

                if next_time > elapsed:
                    time.sleep(next_time - elapsed)

            if np.any(terminated) or np.any(truncated):
                print("환경 종료 조건 발생")
                break

    finally:
        # 비정상 종료 시에도 기록 가능한 데이터 보존
        env.close()

        if log_rows:
            save_csv(log_rows, run_index)

            data = np.asarray(log_rows, dtype=float)

            save_graphs(
                data,
                run_index,
                config,
            )

            print(f"실험 결과 저장 완료: {LOG_DIR}")


if __name__ == "__main__":
    main()
