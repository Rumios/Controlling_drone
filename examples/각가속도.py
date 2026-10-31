
import pybullet as p
import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


# ===== 기본 설정 =====
DURATION = 10
SIM_FREQ = 240

# 카메라 갱신 주기 (Hz)
CAMERA_FREQ = 60

# 목표 Pitch 각도 (도)
TARGET_PITCH = 10.0

# PID 계수
# P: 목표 각도와 현재 각도의 오차 보정
# I: 누적 오차 보정
# D: 회전 속도 억제
KP = 0.003
KI = 0.0001
KD = 0.001

# 적분 오차 최대값
INTEGRAL_LIMIT = 0.5


def main():
    # ==========================================
    # 1. 드론 시뮬레이션 환경 설정
    # ==========================================

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([
            [0.0, 0.0, 1.0]
        ]),
        pyb_freq=SIM_FREQ,
        ctrl_freq=SIM_FREQ,
        gui=True,
        user_debug_gui=False
    )

    # PyBullet 기본 UI 숨기기
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
        physicsClientId=env.CLIENT
    )

    # 초기 카메라 시점 설정
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 1],
        physicsClientId=env.CLIENT
    )

    # ==========================================
    # 2. 기본 추력 및 PID 설정
    # ==========================================

    # 드론이 호버링하는 데 필요한 총추력 (N)
    hover_thrust = env.GRAVITY

    # 모터 1개당 호버링 추력 (N)
    motor_hover_thrust = hover_thrust / 4

    # 목표 Pitch 각도를 라디안으로 변환
    target_pitch = np.deg2rad(TARGET_PITCH)

    # PID 적분 오차 초기화
    integral_error = 0.0

    # 제어 주기
    dt = 1.0 / SIM_FREQ

    # 각가속도 계산을 위한 이전 각속도
    previous_ang_vel = None

    # 첫 번째 모터 명령은 호버링 RPM으로 설정
    hover_rpm = np.sqrt(motor_hover_thrust / env.KF)
    action = np.full((1, 4), hover_rpm)

    print(f"호버링 총추력: {hover_thrust:.6f} N")
    print(f"호버링 RPM: {hover_rpm:.2f}")
    print(f"목표 Pitch: {TARGET_PITCH:.2f}°")

    # ==========================================
    # 3. 시뮬레이션 실행 설정
    # ==========================================

    total_steps = int(DURATION * SIM_FREQ)

    # 카메라 갱신 간격
    camera_interval = max(1, SIM_FREQ // CAMERA_FREQ)

    start_time = time.time()

    # ==========================================
    # 4. 시뮬레이션 반복 실행
    # ==========================================

    try:
        for i in range(total_steps):

            # 모터 RPM을 적용하고 물리 계산 진행
            obs, reward, terminated, truncated, info = env.step(action)

            # 첫 번째 드론 상태 벡터
            state = obs[0]

            # 현재 위치 (X, Y, Z)
            position = state[0:3]

            # 현재 자세 (Roll, Pitch, Yaw)
            orientation = state[7:10]

            # 현재 Pitch 각도 (rad)
            current_pitch = orientation[1]

            # 현재 각속도 (X, Y, Z, rad/s)
            angular_velocity = state[13:16].copy()

            # ==========================================
            # 5. 각속도 및 각가속도 계산
            # ==========================================

            if previous_ang_vel is None:
                angular_acceleration = np.zeros(3)
            else:
                angular_acceleration = (
                    angular_velocity - previous_ang_vel
                ) / dt

            previous_ang_vel = angular_velocity.copy()

            # ==========================================
            # 6. Pitch PID 제어
            # ==========================================

            # 목표 Pitch와 현재 Pitch의 오차
            pitch_error = target_pitch - current_pitch

            # P항: 현재 각도 오차에 비례
            p_term = KP * pitch_error

            # I항: 지금까지 발생한 오차 누적
            integral_error += pitch_error * dt

            # 적분 오차가 지나치게 커지는 현상 방지
            integral_error = np.clip(
                integral_error,
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT
            )

            i_term = KI * integral_error

            # D항: 현재 Pitch 각속도에 반대 방향으로 작용
            # 월드 좌표계의 각속도를 기체 좌표계로 변환
            quaternion = state[3:7]
            rotation_matrix = np.array(
                p.getMatrixFromQuaternion(quaternion)
            ).reshape(3, 3)

            body_ang_vel = rotation_matrix.T @ angular_velocity

            pitch_rate = body_ang_vel[1]

            d_term = -KD * pitch_rate

            # PID 출력: 목표 Pitch 회전 토크 (N·m)
            pitch_torque = p_term + i_term + d_term

            # ==========================================
            # 7. Pitch 토크를 모터 추력으로 변환
            # ==========================================

            # X형 드론의 Pitch 토크 레버암 (m)
            arm = env.L / np.sqrt(2)

            # 필요한 앞뒤 모터 추력 차이 계산
            thrust_delta = pitch_torque / (4 * arm)

            # CF2X 모터 순서:
            # 0: 전방 좌측 방향
            # 1: 후방 좌측 방향
            # 2: 후방 우측 방향
            # 3: 전방 우측 방향
            #
            # Pitch 양의 토크:
            # 앞쪽 추력을 감소시키고 뒤쪽 추력을 증가

            motor_thrust = np.array([
                motor_hover_thrust - thrust_delta,
                motor_hover_thrust + thrust_delta,
                motor_hover_thrust + thrust_delta,
                motor_hover_thrust - thrust_delta
            ])

            # 모터 추력 제한
            max_motor_thrust = env.KF * env.MAX_RPM**2

            motor_thrust = np.clip(
                motor_thrust,
                0,
                max_motor_thrust
            )

            # 추력을 RPM으로 변환
            # F = KF * RPM²
            motor_rpm = np.sqrt(motor_thrust / env.KF)

            # 다음 프레임에서 사용할 모터 명령
            action = motor_rpm.reshape(1, 4)

            # ==========================================
            # 8. 카메라 및 상태 출력
            # ==========================================

            # 카메라를 60Hz로 갱신
            if i % camera_interval == 0:

                # 현재 카메라 상태 가져오기
                camera_info = p.getDebugVisualizerCamera(
                    physicsClientId=env.CLIENT
                )

                # 사용자가 조절한 카메라 회전각과 확대 거리 유지
                camera_distance = camera_info[10]
                camera_yaw = camera_info[8]
                camera_pitch = camera_info[9]

                # 카메라가 드론을 따라가도록 위치만 갱신
                p.resetDebugVisualizerCamera(
                    cameraDistance=camera_distance,
                    cameraYaw=camera_yaw,
                    cameraPitch=camera_pitch,
                    cameraTargetPosition=position,
                    physicsClientId=env.CLIENT
                )

            # 0.5초마다 상태 출력
            if i % (SIM_FREQ // 2) == 0:

                print(
                    f"시간: {i / SIM_FREQ:.2f}s\n"
                    f"위치: {np.round(position, 3)}\n"
                    f"자세(RPY): {np.round(np.rad2deg(orientation), 2)}°\n"
                    f"목표 Pitch: {TARGET_PITCH:.2f}°\n"
                    f"현재 Pitch: {np.rad2deg(current_pitch):.2f}°\n"
                    f"Pitch 오차: {np.rad2deg(pitch_error):.2f}°\n"
                    f"Pitch 각속도: {pitch_rate:.4f} rad/s\n"
                    f"각가속도(XYZ): {np.round(angular_acceleration, 3)}\n"
                    f"P항: {p_term:.6f}\n"
                    f"I항: {i_term:.6f}\n"
                    f"D항: {d_term:.6f}\n"
                    f"PID 토크: {pitch_torque:.6f} N·m\n"
                    f"모터 추력: {np.round(motor_thrust, 5)}\n"
                    f"모터 RPM: {np.round(motor_rpm, 2)}\n"
                )

            # 시뮬레이션 상태 화면 갱신
            env.render()

            # 실제 시간과 시뮬레이션 시간 동기화
            elapsed = time.time() - start_time
            target_time = (i + 1) / SIM_FREQ

            if target_time > elapsed:
                time.sleep(target_time - elapsed)

            if terminated or truncated:
                break

    # ==========================================
    # 9. 시뮬레이션 종료
    # ==========================================

    finally:
        env.close()

    print("시뮬레이션 종료")


if __name__ == "__main__":
    main()
