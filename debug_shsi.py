from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

# YOLOv8モデルをロード
model_path = "yolo11n.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.3,
    device="cpu"  # 'cuda:0' を使用する場合は、GPUを指定
)

# Webカメラの初期化
cap = cv2.VideoCapture(0)

try:
    while True:
        # フレームを取得
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        # フレームが4チャンネルの場合、3チャンネルに変換
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # フレームに対してSAHIを用いたスライス推論を適用
        result = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=256,
            slice_width=256,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # 検出結果をフレームに描画
        for object_prediction in result.object_prediction_list:
            x1, y1, x2, y2 = map(int, object_prediction.bbox)
            confidence = object_prediction.score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # フレームをウィンドウに表示
        cv2.imshow("Real-time Detection with SAHI", frame)

        # 'q'キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Webカメラを解放
    cap.release()
    # ウィンドウを閉じる
    cv2.destroyAllWindows()