import os
import os
import time
from collections import deque
import cv2
from ultralytics import YOLO

# Import your custom modules
from config import MODEL_PATH, SMOOTHING_FRAMES, SLOUCH_TOLERANCE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER
from tracker import calculate_posture_metric

def main():
    print("[INFO] Loading YOLOv8 Pose model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    metric_buffer = deque(maxlen=SMOOTHING_FRAMES)
    baseline_metric = None
    focus_start_time = time.time()
    slouch_start_time = None
    prev_time = time.time()
    last_alarm_time = 0

    print("\n--- Controls ---")
    print("Press 'c' while sitting upright to calibrate baseline.")
    print("Press 'q' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        results = model(frame, verbose=False, conf=0.5)

        current_ratio = None
        status_text = "Calibrate baseline ('c')"
        status_color = (200, 200, 200)

        for result in results:
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                kpts = result.keypoints.data[0].cpu().numpy()
                current_ratio = calculate_posture_metric(kpts)

                for idx in [L_EAR, R_EAR, L_SHOULDER, R_SHOULDER]:
                    x, y, conf = kpts[idx]
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 255), -1)
                break

        if current_ratio is not None:
            metric_buffer.append(current_ratio)
            smoothed_metric = sum(metric_buffer) / len(metric_buffer)

            if baseline_metric is not None:
                ratio_pct = smoothed_metric / baseline_metric

                if ratio_pct < SLOUCH_TOLERANCE:
                    status_text = f"SLOUCHING ({int(ratio_pct * 100)}%)"
                    status_color = (0, 0, 255)
                    if slouch_start_time is None:
                        slouch_start_time = time.time()
                else:
                    status_text = f"LOCKED IN POSTURE ({int(ratio_pct * 100)}%)"
                    status_color = (0, 255, 0)
                    slouch_start_time = None
            else:
                status_text = "Press 'c' to Calibrate"
                status_color = (0, 255, 255)
        else:
            status_text = "Subject Not Detected"
            status_color = (0, 0, 255)

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        elapsed_focus = int(current_time - focus_start_time)
        focus_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_focus))

        # HUD visuals
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Focus: {focus_str}", (w - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # if slouch lasts longer than 3.0 seconds 
        if slouch_start_time and (current_time - slouch_start_time > 3.0):
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
            cv2.putText(frame, "CORRECT POSTURE", (w // 2 - 180, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
        
        #trigger an alarm sound and a voice 
            if current_time - last_alarm_time > 7.0:
                os.system("(afplay /System/Library/Sounds/Glass.aiff; say 'Evelyn lock in!') &")
                last_alarm_time = current_time

        cv2.imshow("Posture & Focus Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if current_ratio is not None:
                baseline_metric = current_ratio
                metric_buffer.clear()
                print(f"[INFO] Calibrated baseline metric: {baseline_metric:.3f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()