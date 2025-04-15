# 주요 기능
- 3D AR 객체 렌더링:
  체스보드에서 검출된 코너를 기준으로 3D 박스와 링을 투영하여 화면에 표시합니다.

- 3D 텍스트 투영: 
지정된 3D 좌표에 텍스트를 투영하여, 화면상에서 마치 3D 공간에 존재하는 것처럼 보이게 합니다.

- 카메라 포즈 추정: 
PnP(Perspective-n-Point) 문제를 해결하여, 카메라의 회전과 이동 벡터를 추정하고, 이를 바탕으로 3D 객체와 텍스트를 정확히 투영합니다.

# 프로젝트 구성
1. 체스보드 코너 찾기
cv.findChessboardCorners()를 사용하여 체스보드의 코너를 찾고, 그 위치를 2D 이미지 좌표로 반환합니다. 이를 통해 카메라의 내부 파라미터와 왜곡 계수를 기반으로 3D 공간에 맞춰 AR 객체를 그릴 수 있습니다.

2. PnP 문제 해결
cv.solvePnP()는 3D 포인트와 대응하는 2D 이미지 포인트를 사용하여 카메라의 회전 벡터(rvec)와 이동 벡터(tvec)를 계산합니다. 이를 통해 카메라와 AR 객체의 상대적 위치를 계산하고, 3D 객체를 화면에 올바르게 투영할 수 있습니다.

3. 3D 객체 투영
박스: 3D 좌표계에서 정해진 위치에 3D 박스를 생성하고, cv.projectPoints()를 사용하여 이를 2D 화면에 투영하여 선으로 연결합니다.

링: 링 형태의 3D 객체를 여러 번 반복하여 그리며, 이를 2D 화면에서 볼 수 있게 합니다.

4. 3D 텍스트 투영
draw_text() 함수는 3D 공간에서 텍스트를 2D 화면으로 변환하여, 지정된 3D 좌표에 텍스트를 배치합니다. 텍스트는 카메라의 회전과 이동에 맞춰 자동으로 위치를 조정합니다.

5. 실시간 비디오 처리
비디오 파일을 읽어 실시간으로 각 프레임을 처리하고, 추적된 3D 객체와 텍스트를 영상에 올려 AR 효과를 구현합니다.


# 비디오와 카메라 보정 파라미터 설정
video_file = './data/myChess2.mp4' <br>
K = np.array([[859.06054177, 0, 973.44260266], 
              [0, 859.57809559, 554.72652988], 
              [0, 0, 1]])  # 카메라 내부 행렬
<br>
dist_coeff = np.array([0.02487478, -0.0966195, 0.00617931, 0.00395589, 0.08269927])  # 왜곡 계수

# 체스보드 코너 찾기 및 포즈 추정
board_pattern = (10, 7)
board_cellsize = 0.022
video = cv.VideoCapture(video_file)

if not video.isOpened():
    print('비디오를 읽을 수 없습니다.')

# 3D 박스와 텍스트를 그리기 위한 함수 정의
def draw_text(text, position_3d, color=(0, 0, 0), scale=1.0, thickness=3):
    pts, _ = cv.projectPoints(np.array([position_3d], dtype=np.float32), rvec, tvec, K, dist_coeff)
    pt_float = pts[0][0]
    if np.all(np.isfinite(pt_float)):
        pt = tuple(np.int32(pt_float))
        cv.putText(img, text, pt, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)

# AR 객체와 텍스트 투영
while True:
    valid, img = video.read()
    if not valid:
        break
    
    success, img_points = cv.findChessboardCorners(img, board_pattern)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
        draw_text("Hello AR", [5, 3.5, -1.0], color=(255, 215, 0), scale=1.2)  # 텍스트 3D 위치
    cv.imshow('Pose Estimation', img)
    if cv.waitKey(10) == 27:
        break

video.release()
cv.destroyAllWindows()
