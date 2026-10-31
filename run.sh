
#!/usr/bin/env bash

# =====================================================
# DroneLab - 드론 시뮬레이터 실행 스크립트
# =====================================================
#
# 사용 방법:
#   1. 아래 설정값을 원하는 대로 수정합니다.
#   2. 파일을 저장합니다.
#   3. 터미널에서 ./run.sh 를 실행합니다.
#
# =====================================================

# ---------------- 시뮬레이션 설정 ----------------

# 드론 개수
# 1 = 드론 한 대, 3 = 드론 세 대
DRONES=1

# 시뮬레이션 지속 시간 (초)
# 예: 60 = 1분, 120 = 2분, 300 = 5분
DURATION=120

# 장애물 표시 여부
# true = 장애물 생성
# false = 장애물 생성하지 않음
OBSTACLES=false

# ------------------------------------------------


# 이 스크립트가 위치한 폴더로 이동합니다.
# 따라서 어느 디렉터리에서 실행해도 정상 작동합니다.
cd "$(dirname "$0")" || exit 1

# Python 가상환경 확인
if [ ! -f ".venv/bin/python" ]; then
    echo "오류: Python 가상환경(.venv)이 없습니다."
    echo "먼저 가상환경을 생성하고 패키지를 설치하세요."
    exit 1
fi

# PID 예제 파일 확인
EXAMPLE="gym-pybullet-drones/gym_pybullet_drones/examples/pid.py"

if [ ! -f "$EXAMPLE" ]; then
    echo "오류: PID 예제 파일을 찾을 수 없습니다."
    exit 1
fi

echo "================================="
echo "     DroneLab Simulator"
echo "================================="
echo "드론 개수: $DRONES"
echo "실행 시간: $DURATION 초"
echo "장애물: $OBSTACLES"
echo "================================="

# 시뮬레이터 실행
.venv/bin/python "$EXAMPLE" \
    --num_drones "$DRONES" \
    --duration_sec "$DURATION" \
    --obstacles "$OBSTACLES"
