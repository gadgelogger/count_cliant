from ultralytics import YOLO
import cv2

# YOLOv8モデルをロード
model = YOLO("yolo11n.pt")

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

        # フレームに対してYOLOモデルを適用
        results = model(frame, classes=[0], conf=0.3)  # 人物検出

        # 検出された人物の数を取得
        human_count = results[0].boxes.data.shape[0]
        print(f"検出された人物の数: {human_count}")

        # 検出結果をフレームに描画
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックスの座標
            confidence = float(box.conf[0])  # 信頼度
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # バウンディングボックスを描画
            cv2.putText(frame, f'Person: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # フレームをウィンドウに表示
        cv2.imshow("Real-time Detection", frame)

        # 'q'キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Webカメラを解放
    cap.release()
    # ウィンドウを閉じる
    cv2.destroyAllWindows()