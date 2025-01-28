import cv2
import numpy as np
import json
from picamera2 import Picamera2

# キャリブレーションデータの読み込み
with open('/home/gadgelogger/calibration_data.json', 'r') as f:
    calibration_data = json.load(f)

DIM = tuple(calibration_data["DIM"])
K = np.array(calibration_data["K"])
D = np.array(calibration_data["D"])

# Picamera2の初期化
picam2 = Picamera2()
picam2.start()

print("キャプチャを開始します。Ctrl+Cで停止")

try:
    while True:
        # 画像を取得
        image = picam2.capture_array()
        image = cv2.rotate(image, cv2.ROTATE_180)

        # 歪み補正用のマップ作成
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, DIM, cv2.CV_16SC2
        )
        
        # 画像の歪み補正
        undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # 画像の表示
        cv2.imshow("Original", image)
        cv2.imshow("Undistorted", undistorted_image)
        
        # 終了するにはEscキーを押す
        if cv2.waitKey(1) & 0xFF == 27:  # Escキー
            break

except KeyboardInterrupt:
    print("\nキャプチャが停止しました")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
