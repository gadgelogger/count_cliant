import cv2
import numpy as np
import json
from picamera2 import Picamera2

# キャリブレーションデータの読み込み
with open('/home/gadgelogger/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

# 元の解像度
original_dim = tuple(calibration_data["DIM"])  # 640x480
K = np.array(calibration_data["K"])
D = np.array(calibration_data["D"])

# 新しい解像度
new_dim = (2592, 1944)

# カメラ行列をスケーリング
scale_x = new_dim[0] / original_dim[0]
scale_y = new_dim[1] / original_dim[1]
K_scaled = K.copy()
K_scaled[0, 0] *= scale_x  # fx
K_scaled[1, 1] *= scale_y  # fy
K_scaled[0, 2] *= scale_x  # cx
K_scaled[1, 2] *= scale_y  # cy

# Picamera2の初期化
picam2 = Picamera2()

# 解像度を2592x1944に設定
config = picam2.create_still_configuration(main={"size": new_dim})
picam2.configure(config)

# カメラスタート
picam2.start()

print("キャプチャを開始します。Ctrl+Cで停止")

try:
    while True:
        # 画像を取得
        image = picam2.capture_array()
        image = cv2.rotate(image, cv2.ROTATE_180)

        # 歪み補正用のマップ作成
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K_scaled, D, np.eye(3), K_scaled, new_dim, cv2.CV_16SC2
        )
        
        # 画像の歪み補正
        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 画像の表示
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original", image.shape[1] // 2, image.shape[0] // 2)
        cv2.imshow("Original", image)
        
        cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Undistorted", undistorted_image.shape[1] // 2, undistorted_image.shape[0] // 2)
        cv2.imshow("Undistorted", undistorted_image)
        
        # 終了するにはEscキーを押す
        if cv2.waitKey(1) & 0xFF == 27:  # Escキー
            break

except KeyboardInterrupt:
    print("\nキャプチャが停止しました")
finally:
    picam2.stop()
    cv2.destroyAllWindows()