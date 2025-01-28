import cv2
import numpy as np
from picamera2 import Picamera2
import time
import json

# Picamera2の初期化
picam2 = Picamera2()
# 解像度を指定する
picam2.configure(picam2.create_still_configuration(main={"size": (2592, 1944)}))
# カメラスタート
picam2.start()

# チェッカーボードのサイズ
CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# キャリブレーションフラグの設定
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# 3D座標の定義
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 3D座標と2D画像座標の格納リスト
objpoints = []
imgpoints = []

# 撮影する画像数
num_images = 100
successful_images = 0

print("キャリブレーションを開始します...")
print("チェッカーボードをカメラ前に配置してください")

while successful_images < num_images:
    time.sleep(2)
    image = picam2.capture_array()
    image = cv2.rotate(image, cv2.ROTATE_180)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 画像のサイズ取得
    DIM = gray.shape[::-1]
    
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (3, 3),
            (-1, -1),
            subpix_criteria
        )
        
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        
        cv2.drawChessboardCorners(image, CHECKERBOARD, corners_refined, ret)
        cv2.imshow('Calibration', image)
        cv2.waitKey(500)
        
        successful_images += 1
        print(f"進捗: {successful_images}/{num_images} 枚")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    print("キャリブレーションを開始します...")
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    
    try:
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        print("\nキャリブレーション完了!")
        print("RMS:", rms)
        
        # 結果の表示
        print("\n=== キャリブレーション結果 ===")
        print("\nDIM =", DIM)
        print("\nK =", K.tolist())
        print("\nD =", D.tolist())
        
        # キャリブレーションデータの保存
        calibration_data = {
            "DIM": DIM,
            "K": K.tolist(),
            "D": D.tolist()
        }
        
        with open('calibration_data.json', 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print("\nキャリブレーションデータを calibration_data.json に保存しました")
        
        picam2.stop()
            
    except Exception as e:
        print("エラーが発生しました:", str(e))
        picam2.stop()
        
else:
    print("十分な数の画像が収集できませんでした")
    picam2.stop()
