import cv2
import numpy as np
from collections import deque
import sys


def open_camera():
    """Busca una cámara disponible y la devuelve si existe."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        for index in range(5):
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[OK] Cámara encontrada: índice {index}, backend {backend}")
                return cap

            cap.release()

    for index in range(5):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[OK] Cámara encontrada: índice {index}, backend por defecto")
            return cap

        cap.release()

    return None


class FireDetector:
    def __init__(self):
        self.prev_gray = None
        self.region_history = {}
        self.confirmed_frames = 0

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray), 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray.copy()

        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 5)
        motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), np.uint8), iterations=1)

        motion_ratio = cv2.countNonZero(motion_mask) / max(motion_mask.size, 1)
        return motion_mask, float(motion_ratio)

    def detect_fire_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 60, 60], dtype=np.uint8)
        upper1 = np.array([30, 255, 255], dtype=np.uint8)

        lower2 = np.array([170, 60, 60], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, hsv

    def analyze_region(self, frame, color_mask, motion_mask):
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            box = (x, y, x + w, y + h)
            roi_color = color_mask[y : y + h, x : x + w]
            roi_motion = motion_mask[y : y + h, x : x + w]

            if roi_color.size == 0:
                continue

            bbox_area = max(w * h, 1)
            color_ratio = cv2.countNonZero(roi_color) / bbox_area
            motion_ratio = cv2.countNonZero(roi_motion) / bbox_area

            if color_ratio < 0.04 or motion_ratio < 0.008:
                continue

            hsv = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2]
            fire_pixels = brightness[roi_color > 0]

            if fire_pixels.size == 0:
                continue

            history = self.region_history.setdefault(box, deque(maxlen=8))
            history.append(float(np.mean(fire_pixels)))

            if len(history) < 4:
                flicker_score = 0.0
            else:
                mean_hist = float(np.mean(history))
                std_hist = float(np.std(history))
                flicker_score = std_hist / max(mean_hist, 1.0)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            irregularity = 1.0 - min(solidity, 1.0)

            if flicker_score < 0.01 or irregularity < 0.05:
                continue

            color_score = min(color_ratio / 0.18, 1.0)
            motion_score = min(motion_ratio / 0.06, 1.0)
            flicker_score_norm = min(flicker_score / 0.10, 1.0)
            shape_score = min(irregularity / 0.30, 1.0)

            score = (
                0.38 * color_score
                + 0.30 * motion_score
                + 0.22 * flicker_score_norm
                + 0.10 * shape_score
            )

            if score > 0.42:
                candidates.append({
                    "box": box,
                    "score": score,
                    "motion": motion_ratio,
                    "flicker": flicker_score,
                })

        return candidates

    def process(self, frame):
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        color_mask, _ = self.detect_fire_color(blurred)
        motion_mask, motion_ratio = self.detect_motion(blurred)

        candidates = self.analyze_region(frame, color_mask, motion_mask)

        best_box = None
        best_score = 0.0

        for cand in candidates:
            if cand["score"] > best_score:
                best_score = cand["score"]
                best_box = cand["box"]

        if best_box is not None and best_score >= 0.45:
            self.confirmed_frames += 1
        else:
            self.confirmed_frames = max(0, self.confirmed_frames - 1)

        fire_detected = self.confirmed_frames >= 2

        display = frame.copy()
        if fire_detected and best_box is not None:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, "FUEGO DETECTADO", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"Score: {best_score:.2f}", (x1, min(y2 + 20, display.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, "MONITOREO NORMAL", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display, f"Movimiento: {motion_ratio:.3f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Confirmacion: {self.confirmed_frames}/2", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        color_view = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        color_view[color_mask > 0] = (0, 0, 255)

        motion_view = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        motion_view[motion_mask > 0] = (0, 255, 0)

        return display, color_view, motion_view, fire_detected


def generate_demo_frame(width=640, height=480, frame_index=0):
    """Genera una escena de prueba con una llama para validar el detector sin cámara."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    cx = width // 2
    cy = height // 2 + 50
    for i in range(1, 90):
        angle = np.linspace(0, 2 * np.pi, 100)
        r = i * 1.2 + (frame_index % 7)
        x = np.int32(cx + r * np.cos(angle) * 0.5)
        y = np.int32(cy - i * 1.5 + 20 * np.sin(frame_index * 0.3 + i * 0.18))
        for px, py in zip(x, y):
            if 0 <= px < width and 0 <= py < height:
                frame[py, px] = [0, max(50, 100 + i), 255]
    return frame


def main():
    camera = open_camera()
    detector = FireDetector()

    if camera is None:
        print("[INFO] No se encontró cámara. Ejecutando modo demostración.")
        for i in range(500):
            frame = generate_demo_frame(frame_index=i)
            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Demo - Fuego", output)
            cv2.imshow("Demo - Color", color_view)
            cv2.imshow("Demo - Movimiento", motion_view)
            if fire_detected:
                print("[ALERTA] Fuego detectado en modo demo.")
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    print("Sistema de detección de incendios iniciado. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[ERROR] La cámara no está entregando imágenes válidas.")
                break

            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Cámara", output)
            cv2.imshow("Detección de color", color_view)
            cv2.imshow("Movimiento", motion_view)

            if fire_detected:
                cv2.putText(output, "ALERTA: POSIBLE INCENDIO", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Cámara", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        print("Modo demo activado.")
        detector = FireDetector()
        for i in range(500):
            frame = generate_demo_frame(frame_index=i)
            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Demo - Fuego", output)
            cv2.imshow("Demo - Color", color_view)
            cv2.imshow("Demo - Movimiento", motion_view)
            if fire_detected:
                print("[ALERTA] Fuego detectado en modo demo.")
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        main()