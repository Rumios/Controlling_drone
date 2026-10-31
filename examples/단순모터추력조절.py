# 이 상태에서는 호버링만 가능하고 전진, 후진, 회전 등이 불가능함
# 왜?
# 만약 드론의 두 앞날개의 추력이 약해지고 뒤가 높아진다면 기울어지면서 앞으로 가는 것이 정상.
# 그러나 RPM을 높였기 때문에 기체를 앞으로 숙이려는 회전력(토크)가 지속적으로 작용
# 그렇기 때문에 꼬꾸라짐.
# 그래서 이 각을 유지시켜주는 시스템이 필요한데 현재의 각속도와 각을 계산하여 적절한 각에서 토크를 0으로 거의 유지하면 전진하게 됨.

import pybullet as p
import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel

# ===== 기본 설정 =====
DURATION = 3
SIM_FREQ = 240
MOTOR_SCALE = [0.98, 1.02, 1.02, 0.98]

def main():
    # 드론 시뮬레이션 환경 설정
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([
            [0.0, 0.0, 1.0]             # 초기 위치 (X, Y, Z), 단위 m
        ]),
        pyb_freq=SIM_FREQ,              # 물리 시뮬레이션 Hz
        ctrl_freq=SIM_FREQ,             # 모터 명령을 적용하는 Hz
        gui=True,                       # PyBullet 시각화 창 활성화
        user_debug_gui=False            # 추가 디버그 UI 비활성화
    )
    
    # PyBullet 기본 UI 숨기기 (선택)
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI,
        0,
        physicsClientId=env.CLIENT
    )

    # 카메라 시점 설정
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,              # 가까울수록 작아짐
        cameraYaw=45,                    # 옆에서 보는 각도
        cameraPitch=-35,                 # 음수일수록 위에서 내려다봄
        cameraTargetPosition=[0, 0, 1],  # 드론 초기 위치 근처
        physicsClientId=env.CLIENT
    )
    
    # RPM 계산 (중력 가속도를 쿼드콥터라서 날개 4개로 하는 것. KF는 추력 계수)
    # 아래 수식은 드론이 공중에 떠 있는데 필요한 rpm을 계산. (일단 지금은 대충 정지 비행에 필요한 수식)
    hover_rpm = np.sqrt(
        env.GRAVITY / (4 * env.KF)
    )
    
    # 모터 4개 출력 배율을 Numpy 배열로 변환
    motor_scale = np.array(MOTOR_SCALE)
    
    # 각 모터 실제 RPM 계산 (RPM * 출력) (이쪽 수식 알아야할 듯)
    motor_rpm = hover_rpm * motor_scale
    
    # (drone_num, motor_num)
    action = motor_rpm.reshape(1, 4)
    
    print(f"호버링 RPM: {hover_rpm:.2f}")
    print(f"모터 RPM: {motor_rpm}")
    
    
    # ==========================================
    # 3. 시뮬레이션 실행 설정
    # ==========================================

    # 전체 반복 횟수 = 실행 시간 × 초당 제어 횟수
    total_steps = int(DURATION * SIM_FREQ)

    # 시뮬레이션 시작 시각 저장
    start_time = time.time()

    # ==========================================
    # 4. 시뮬레이션 반복 실행
    # ==========================================

    try:
        for i in range(total_steps):

            # 모터 RPM을 적용하고 물리 계산을 진행
            # obs: 드론의 현재 상태
            # reward: 보상 (이번 실습에서는 사용하지 않음)
            # terminated: 종료 조건
            # truncated: 시간 등으로 인한 중단 조건
            # info: 추가 정보
                        # 모터 출력 적용 및 상태 갱신
            obs, reward, terminated, truncated, info = env.step(action)

            # 매 프레임 드론의 현재 위치 확인
            position = obs[0, 0:3]

            # 카메라가 드론을 실시간으로 추적
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=position,
                physicsClientId=env.CLIENT
            )

            # 1초마다 현재 드론 상태 출력
            if i % SIM_FREQ == 0:

                # 첫 번째 드론의 상태 벡터
                state = obs[0]

                # 상태 벡터의 0~2번: X, Y, Z 위치
                position = state[0:3]

                # 상태 벡터의 7~9번: Roll, Pitch, Yaw
                orientation = state[7:10]

                print(
                    f"시간: {i / SIM_FREQ:.1f}s | "
                    f"위치: {position} | "
                    f"자세: {orientation}"
                )
            
            # 시뮬레이션 상태 화면 갱신
            env.render()

            # 실제 시간과 시뮬레이션 시간 동기화
            elapsed = time.time() - start_time
            target_time = (i + 1) / SIM_FREQ

            if target_time > elapsed:
                time.sleep(target_time - elapsed)

            # 환경에서 종료를 요청하면 반복 중단
            if terminated or truncated:
                break

    # ==========================================
    # 5. 시뮬레이션 종료
    # ==========================================

    finally:
        # 오류가 발생해도 시뮬레이션 환경 정리
        env.close()

    print("시뮬레이션 종료")
    
    
    


if __name__ == "__main__":
    main()