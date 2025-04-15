import numpy as np
import cv2 as cv

# 주어진 비디오 및 보정 데이터
video_file = './data/myChess2.mp4'

# 카메라 내부 보정 행렬 (K): 카메라의 초점 거리와 주점을 포함한 행렬
K = np.array([[859.06054177, 0, 973.44260266],
              [0, 859.57809559, 554.72652988],
              [0, 0, 1]])  # 보정된 카메라 내부 행렬

# 카메라 왜곡 계수 (dist_coeff): 렌즈 왜곡 보정
dist_coeff = np.array([0.02487478, -0.0966195, 0.00617931, 0.00395589, 0.08269927])

# 체스보드 패턴 크기: 10개의 열과 7개의 행
board_pattern = (10, 7)
# 체스보드 각 정사각형의 크기 (단위: 미터)
board_cellsize = 0.022
# 체스보드 코너를 찾기 위한 설정 (영상 전처리 기준)
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# 비디오 열기
video = cv.VideoCapture(video_file)  # 비디오 캡처 초기화
assert video.isOpened(), '주어진 입력을 읽을 수 없습니다: ' + video_file  # 비디오 파일 열기 확인

# 3D 박스를 위한 좌표 준비 (박스와 링의 3D 위치 설정)
x_box = [1, 7]  # x축 범위
y_box = [0, 6]  # y축 범위
z = -1.3        # z축 값 (카메라 앞쪽으로 나오게 설정)
z_box = [0, z]  # z값에 대한 범위 설정
ring_box_gap = 0.5  # 링 박스의 간격
x_ring = [x_box[0] + ring_box_gap, x_box[1] - ring_box_gap]  # 링의 x축 범위
y_ring = [y_box[0] + ring_box_gap, y_box[1] - ring_box_gap]  # 링의 y축 범위
z_ring = [z + 0.6, z + 0.3, z]  # 링의 z축 값

# 색상 정의 (BGR 순서)
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (255, 255, 255)]

# 링과 박스의 3D 점들을 저장할 리스트 초기화
ring = [[] for _ in range(3)]
box = [[] for _ in range(2)]
box_ring = [[] for _ in range(3)]

# 3D 박스 계산
i = 0
for z in z_box:
    for x in x_box:
        for y in y_box:
            box[i].append([x, y, z])
    box[i][2], box[i][3] = box[i][3], box[i][2]  # 박스의 꼭짓점 순서 수정
    box[i] = board_cellsize * np.array(box[i])  # 각 점을 미터 단위로 변환
    i += 1

# 링의 3D 좌표 계산
i = 0
for z in z_ring:
    for x in x_box:
        for y in y_box:
            box_ring[i].append([x, y, z])
    for x in x_ring:
        for y in y_ring:
            ring[i].append([x, y, z])
    box_ring[i][2], box_ring[i][3] = box_ring[i][3], box_ring[i][2]
    box_ring[i] = board_cellsize * np.array(box_ring[i])  # 링도 미터 단위로 변환
    ring[i][2], ring[i][3] = ring[i][3], ring[i][2]
    ring[i] = board_cellsize * np.array(ring[i])
    i += 1

# 3D 객체를 그리기 위한 함수들
def draw_line(pt1, pt2, color, thickness=2):
    """
    두 점을 연결하는 선을 그리는 함수
    pt1, pt2: 3D 좌표
    color: 선 색상
    thickness: 선 두께
    """
    pro_p1, _ = cv.projectPoints(pt1, rvec, tvec, K, dist_coeff)
    pro_p2, _ = cv.projectPoints(pt2, rvec, tvec, K, dist_coeff)
    for b, t in zip(pro_p1, pro_p2):
        cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), color, thickness)  # 화면에 선 그리기

def draw_poly(obj, color):
    """
    주어진 3D 객체를 화면에 그리는 함수
    obj: 3D 객체 (점들의 리스트)
    color: 객체의 색상
    """
    pro_p, _ = cv.projectPoints(obj, rvec, tvec, K, dist_coeff)
    cv.polylines(img, [np.int32(pro_p)], True, color, 2)  # 객체의 면을 그리기

def draw_text(text, position_3d, color=(0, 0, 0), scale=1.0, thickness=3):
    """
    3D 위치에 텍스트를 투영하여 화면에 출력
    position_3d: 텍스트의 3D 좌표
    color: 텍스트 색상
    scale: 텍스트 크기
    thickness: 텍스트 두께
    """
    # 3D 좌표를 2D 화면 좌표로 변환
    pts, _ = cv.projectPoints(np.array([position_3d], dtype=np.float32), rvec, tvec, K, dist_coeff)
    pt_float = pts[0][0]

    # 좌표가 유효하면 텍스트 출력
    if np.all(np.isfinite(pt_float)):
        pt = tuple(np.int32(pt_float))  # 화면 좌표로 변환
        cv.putText(img, text, pt, cv.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, color, thickness, cv.LINE_AA)

# 체스보드 코너들 (3D 위치)
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# 포즈 추정 실행 (비디오에서 프레임 읽고, PnP로 회전과 이동 벡터 계산)
while True:
    # 비디오에서 이미지 읽기
    valid, img = video.read()
    if not valid:
        break  # 비디오 끝나면 종료

    # 체스보드 코너를 찾기
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        # PnP 문제 풀기 (3D 포인트와 2D 이미지 포인트를 이용해 회전 벡터와 이동 벡터 계산)
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # AR 객체 그리기
        box_l, _ = cv.projectPoints(box[0], rvec, tvec, K, dist_coeff)
        cv.fillPoly(img, [np.int32(box_l)], color[2])  # 박스 채우기
        draw_line(box[0], box[1], color[4], 3)  # 박스를 연결하는 선 그리기

        # 링 그리기
        for i in range(3):
            draw_poly(ring[i], color[0])

        # 박스 링 그리기
        for j in range(3):
            for k in range(4):
                draw_line(ring[j][k], box_ring[j][k], color[4])

        # 텍스트 3D 위치에 그리기
        text_pos = np.array([2.2, 3.5, -0.7]) * board_cellsize  # 텍스트의 3D 위치
        draw_text("Let's Box!", text_pos, color=(0, 215, 255), scale=1.5)

    # 이미지 크기 변경 및 화면에 출력
    resized_img = cv.resize(img, None, fx=0.5, fy=0.5)
    cv.imshow('Pose Estimation (Chessboard)', resized_img)

    # 키 이벤트 대기
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()  # 스페이스바 누르면 대기
    if key == 27:  # ESC 키를 누르면 종료
        break

# 비디오 캡처 해제 및 창 닫기
video.release()
cv.destroyAllWindows()
