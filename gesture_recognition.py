"""
Hand Gesture Recognition System
================================
Requirements:
    pip install mediapipe opencv-python numpy tensorflow scikit-learn

Usage:
    python gesture_recognition.py                        # Webcam mode (default)
    python gesture_recognition.py --source video.mp4    # Video file mode
    python gesture_recognition.py --train               # Collect training data & train ML model
    python gesture_recognition.py --model ml            # Use ML model instead of rule-based
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
import json
import os
import math
from dataclasses import dataclass, field
from typing import Optional

# ─── Optional ML imports (only needed with --train or --model ml) ──────────────
try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ─── Config ───────────────────────────────────────────────────────────────────
GESTURES = {
    "open_hand":    {"label": "Open Hand",    "emoji": "[ALL]"},
    "fist":         {"label": "Fist",          "emoji": "[FIST]"},
    "thumbs_up":    {"label": "Thumbs Up",     "emoji": "[THU+]"},
    "thumbs_down":  {"label": "Thumbs Down",   "emoji": "[THU-]"},
    "peace":        {"label": "Peace / V",     "emoji": "[V]"},
    "ok":           {"label": "OK",            "emoji": "[OK]"},
    "point":        {"label": "Pointing",      "emoji": "[>>]"},
    "rock":         {"label": "Rock",          "emoji": "[ROCK]"},
    "call_me":      {"label": "Call Me",       "emoji": "[CALL]"},
    "three":        {"label": "Three Fingers", "emoji": "[3]"},
    "four":         {"label": "Four Fingers",  "emoji": "[4]"},
    "pinch":        {"label": "Pinch",         "emoji": "[PINCH]"},
    "none":         {"label": "No Gesture",    "emoji": ""},
}

GESTURE_KEYS = list(GESTURES.keys())[:-1]  # exclude 'none'

COLORS = {
    "teal":   (117, 173, 29),
    "green":  (53, 153, 9),
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "gray":   (120, 120, 120),
    "yellow": (0, 200, 220),
    "red":    (50, 50, 220),
}

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class GestureResult:
    gesture_id: str
    label: str
    confidence: float
    hand_index: int
    handedness: str
    landmarks: list


@dataclass
class SessionStats:
    total_detections: int = 0
    gesture_counts: dict = field(default_factory=dict)
    confidence_sum: float = 0.0
    start_time: float = field(default_factory=time.time)
    fps_history: list = field(default_factory=list)


# ─── Landmark utilities ───────────────────────────────────────────────────────
class LandmarkUtils:
    @staticmethod
    def landmark_list(hand_landmarks):
        return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    @staticmethod
    def finger_extended(lm, tip, pip, mcp):
        return lm[tip][1] < lm[pip][1] and lm[tip][1] < lm[mcp][1]

    @staticmethod
    def thumb_extended(lm, handedness):
        tip, mcp = lm[4], lm[2]
        side = 1 if handedness == "Right" else -1
        return (tip[0] - mcp[0]) * side > 0.04 or abs(tip[0] - mcp[0]) > 0.05

    @staticmethod
    def thumb_down(lm):
        return lm[4][1] > lm[2][1] + 0.05

    @staticmethod
    def pinch_distance(lm):
        t, i = lm[4], lm[8]
        return math.hypot(t[0] - i[0], t[1] - i[1])

    @staticmethod
    def normalize_landmarks(lm):
        """Flatten & normalize landmarks relative to wrist for ML input."""
        base_x, base_y, base_z = lm[0]
        coords = []
        for (x, y, z) in lm:
            coords.extend([x - base_x, y - base_y, z - base_z])
        # Scale
        max_val = max(abs(v) for v in coords) or 1.0
        return [v / max_val for v in coords]


# ─── Rule-based classifier ────────────────────────────────────────────────────
class RuleBasedClassifier:
    """
    Classifies hand gestures using geometric rules over MediaPipe landmarks.
    No training required. Works out of the box.
    """

    def classify(self, lm: list, handedness: str) -> GestureResult:
        u = LandmarkUtils
        idx  = u.finger_extended(lm, 8,  6,  5)
        mid  = u.finger_extended(lm, 12, 10, 9)
        ring = u.finger_extended(lm, 16, 14, 13)
        pin  = u.finger_extended(lm, 20, 18, 17)
        thu  = u.thumb_extended(lm, handedness)
        t_dn = u.thumb_down(lm)
        pd   = u.pinch_distance(lm)

        ext_count = sum([thu, idx, mid, ring, pin])

        gesture_id, confidence = "none", 0.5

        if pd < 0.06 and idx and not mid and not ring and not pin:
            gesture_id, confidence = "ok", 0.89
        elif ext_count == 5:
            gesture_id, confidence = "open_hand", 0.95
        elif ext_count == 0:
            gesture_id, confidence = "fist", 0.92
        elif not thu and idx and not mid and not ring and not pin:
            gesture_id, confidence = "point", 0.94
        elif not thu and idx and mid and not ring and not pin:
            gesture_id, confidence = "peace", 0.93
        elif not thu and idx and not mid and not ring and pin:
            gesture_id, confidence = "rock", 0.91
        elif thu and not idx and not mid and not ring and pin:
            gesture_id, confidence = "call_me", 0.90
        elif not thu and idx and mid and ring and not pin:
            gesture_id, confidence = "three", 0.88
        elif not thu and idx and mid and ring and pin:
            gesture_id, confidence = "four", 0.90
        elif t_dn and not idx and not mid and not ring and not pin:
            gesture_id, confidence = "thumbs_down", 0.87
        elif thu and not idx and not mid and not ring and not pin and not t_dn:
            gesture_id, confidence = "thumbs_up", 0.88
        elif pd < 0.06:
            gesture_id, confidence = "pinch", 0.82

        return gesture_id, confidence


# ─── ML classifier ────────────────────────────────────────────────────────────
class MLClassifier:
    """
    TensorFlow MLP classifier trained on landmark feature vectors.
    Train with: python gesture_recognition.py --train
    """

    MODEL_PATH = "gesture_model.keras"
    DATA_PATH  = "gesture_data.json"

    def __init__(self):
        self.model = None
        self.class_names = GESTURE_KEYS
        if os.path.exists(self.MODEL_PATH):
            self.model = tf.keras.models.load_model(self.MODEL_PATH)
            print(f"[ML] Loaded model from {self.MODEL_PATH}")
        else:
            print("[ML] No saved model found. Run with --train to create one.")

    def classify(self, lm: list, handedness: str):
        if self.model is None:
            return "none", 0.0
        features = np.array(LandmarkUtils.normalize_landmarks(lm)).reshape(1, -1)
        probs = self.model.predict(features, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])

    @classmethod
    def train(cls, data_path=DATA_PATH):
        if not ML_AVAILABLE:
            print("TensorFlow not installed. pip install tensorflow")
            return
        if not os.path.exists(data_path):
            print(f"No training data at {data_path}. Collect data first.")
            return

        with open(data_path) as f:
            data = json.load(f)

        X, y = [], []
        for item in data:
            X.append(item["features"])
            y.append(GESTURE_KEYS.index(item["label"]))

        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, num_classes=len(GESTURE_KEYS))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(63,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(GESTURE_KEYS), activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        cb = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]

        print(f"Training on {len(X_train)} samples…")
        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=100, batch_size=32, callbacks=cb, verbose=1)

        model.save(cls.MODEL_PATH)
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation accuracy: {acc*100:.1f}%")
        print(f"Model saved to {cls.MODEL_PATH}")


# ─── Data collector (for ML training) ────────────────────────────────────────
class DataCollector:
    """Press 0-9 / a-b to label and save landmark samples."""

    def __init__(self, path=MLClassifier.DATA_PATH):
        self.path = path
        self.data = []
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} existing samples.")

    KEY_MAP = {ord(str(i)): GESTURE_KEYS[i] for i in range(min(10, len(GESTURE_KEYS)))}

    def save_sample(self, key, lm):
        label = self.KEY_MAP.get(key)
        if label is None:
            return False
        features = LandmarkUtils.normalize_landmarks(lm)
        self.data.append({"label": label, "features": features})
        with open(self.path, "w") as f:
            json.dump(self.data, f)
        print(f"Saved sample for '{label}' (total: {len(self.data)})")
        return True

    def counts(self):
        from collections import Counter
        return Counter(d["label"] for d in self.data)


# ─── Renderer ─────────────────────────────────────────────────────────────────
class Renderer:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    LANDMARK_STYLE = mp.solutions.drawing_utils.DrawingSpec(
        color=(93, 202, 165), thickness=2, circle_radius=3
    )
    CONNECTION_STYLE = mp.solutions.drawing_utils.DrawingSpec(
        color=(29, 158, 117), thickness=2
    )

    @classmethod
    def draw_landmarks(cls, frame, hand_landmarks):
        cls.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
            cls.LANDMARK_STYLE,
            cls.CONNECTION_STYLE,
        )

    @staticmethod
    def draw_gesture_box(frame, result: GestureResult, position: tuple):
        x, y = position
        label = result.label
        conf  = int(result.confidence * 100)
        text  = f"{label}  {conf}%"
        hand  = f"[{result.handedness}]"

        # Background box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x - 8, y - th - 12), (x + tw + 8, y + 6),
                      (30, 30, 30), -1)
        cv2.rectangle(frame, (x - 8, y - th - 12), (x + tw + 8, y + 6),
                      COLORS["teal"], 1)

        # Text
        cv2.putText(frame, text,  (x, y - 2),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS["white"], 2)
        cv2.putText(frame, hand,  (x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["teal"], 1)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = x, y + 24, 120, 5
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * result.confidence)
        bar_color = COLORS["teal"] if result.confidence > 0.75 else COLORS["yellow"]
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)

    @staticmethod
    def draw_sidebar(frame, stats: SessionStats, results: list, h: int, w: int):
        sidebar_w = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        x = w - sidebar_w + 10
        y = 20

        def put(text, dy=22, scale=0.5, color=COLORS["white"], thickness=1):
            nonlocal y
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += dy

        put("GESTURE SYSTEM", 28, 0.55, COLORS["teal"], 2)
        put(f"Hands: {len(results)}", 20, 0.45, COLORS["gray"])

        elapsed = time.time() - stats.start_time
        put(f"Session: {int(elapsed)}s", 20, 0.45, COLORS["gray"])
        put(f"Detections: {stats.total_detections}", 20, 0.45, COLORS["gray"])
        if stats.total_detections:
            avg_conf = stats.confidence_sum / stats.total_detections
            put(f"Avg conf: {int(avg_conf*100)}%", 20, 0.45, COLORS["gray"])

        y += 8
        put("── Active Gestures ──", 20, 0.4, COLORS["teal"])
        for r in results:
            put(f"  {r.label} ({int(r.confidence*100)}%)", 18, 0.4, COLORS["white"])

        y += 8
        put("── Top Gestures ──", 20, 0.4, COLORS["teal"])
        top = sorted(stats.gesture_counts.items(), key=lambda x: -x[1])[:5]
        for gid, cnt in top:
            put(f"  {GESTURES[gid]['label']}: {cnt}", 18, 0.4, COLORS["white"])

    @staticmethod
    def draw_fps(frame, fps: float):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS["teal"], 2)

    @staticmethod
    def draw_training_overlay(frame, collector: DataCollector, h: int, w: int):
        """Shows key-to-gesture mapping for data collection mode."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 200), (300, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        y = h - 190
        cv2.putText(frame, "TRAINING MODE — press key to label:", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["yellow"], 1)
        y += 18
        for k, label in DataCollector.KEY_MAP.items():
            cnt = collector.counts().get(label, 0)
            cv2.putText(frame, f"  [{chr(k)}] {label}: {cnt} samples",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["white"], 1)
            y += 15


# ─── Main pipeline ─────────────────────────────────────────────────────────────
class GestureRecognitionPipeline:
    def __init__(self, source=0, use_ml=False, training_mode=False):
        self.source = source
        self.training_mode = training_mode
        self.stats = SessionStats()

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85,
         )

        # Classifier
        if use_ml and ML_AVAILABLE:
            self.classifier = MLClassifier()
            print("[Pipeline] Using ML classifier")
        else:
            self.classifier = RuleBasedClassifier()
            print("[Pipeline] Using rule-based classifier")

        # Data collector
        self.collector = DataCollector() if training_mode else None

        self.renderer = Renderer()
        self._fps_time = time.time()
        self._fps_frame = 0
        self._fps = 0.0
        self._last_gesture_id = None

    def _update_fps(self):
        self._fps_frame += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self._fps = self._fps_frame / elapsed
            self._fps_frame = 0
            self._fps_time = now

    def process_frame(self, frame) -> tuple:
        """Process one frame. Returns (annotated_frame, list[GestureResult])."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_results = self.hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        results: list[GestureResult] = []

        if mp_results.multi_hand_landmarks:
            for i, hand_lm in enumerate(mp_results.multi_hand_landmarks):
                handedness = mp_results.multi_handedness[i].classification[0].label
                lm = LandmarkUtils.landmark_list(hand_lm)

                gesture_id, confidence = self.classifier.classify(lm, handedness)
                if confidence < 0.90:
                   gesture_id = "none"

                result = GestureResult(
                    gesture_id=gesture_id,
                    label=GESTURES.get(gesture_id, GESTURES["none"])["label"],
                    confidence=confidence,
                    hand_index=i,
                    handedness=handedness,
                    landmarks=lm,
                )
                results.append(result)

                # Draw skeleton
                self.renderer.draw_landmarks(frame, hand_lm)

                # Draw gesture label near wrist
                wrist_px = (int((1 - lm[0][0]) * w), int(lm[0][1] * h))
                label_pos = (max(10, wrist_px[0] - 60), max(50, wrist_px[1] - 50))
                self.renderer.draw_gesture_box(frame, result, label_pos)

                # Stats
                if gesture_id != "none":
                    self.stats.total_detections += 1
                    self.stats.confidence_sum += confidence
                    self.stats.gesture_counts[gesture_id] = \
                        self.stats.gesture_counts.get(gesture_id, 0) + 1

        # UI overlays
        self.renderer.draw_sidebar(frame, self.stats, results, h, w)
        self.renderer.draw_fps(frame, self._fps)

        if self.training_mode and self.collector:
            self.renderer.draw_training_overlay(frame, self.collector, h, w)

        self._update_fps()
        return frame, results

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open source '{self.source}'")
            return

        print("Running. Press Q to quit. Press S to save screenshot.")
        if self.training_mode:
            print("Training mode: press 0-9 to label gesture for current hand.")

        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Mirror webcam
            if self.source == 0:
                frame = cv2.flip(frame, 1)

            annotated, results = self.process_frame(frame)
            cv2.imshow("Hand Gesture Recognition", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"gesture_screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, annotated)
                print(f"Screenshot saved: {fname}")
            elif self.training_mode and self.collector and results:
                # Label first detected hand
                self.collector.save_sample(key, results[0].landmarks)

        cap.release()
        cv2.destroyAllWindows()
        print("\nSession complete.")
        print(f"Total detections : {self.stats.total_detections}")
        if self.stats.total_detections:
            avg = self.stats.confidence_sum / self.stats.total_detections
            print(f"Average confidence: {avg*100:.1f}%")
        print("Top gestures:")
        for gid, cnt in sorted(self.stats.gesture_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  {GESTURES[gid]['label']:20s} {cnt}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
    parser.add_argument("--source", default=0,
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--model", choices=["rules", "ml"], default="rules",
                        help="Classifier: 'rules' (default) or 'ml' (requires trained model)")
    parser.add_argument("--train", action="store_true",
                        help="Enter data-collection + training mode")
    parser.add_argument("--train-only", action="store_true",
                        help="Train ML model from saved data without opening camera")
    args = parser.parse_args()

    if args.train_only:
        if not ML_AVAILABLE:
            print("TensorFlow not installed. pip install tensorflow scikit-learn")
            return
        MLClassifier.train()
        return

    source = int(args.source) if str(args.source).isdigit() else args.source
    use_ml = args.model == "ml"
    training = args.train

    pipeline = GestureRecognitionPipeline(
        source=source,
        use_ml=use_ml,
        training_mode=training,
    )
    pipeline.run()


if __name__ == "__main__":
    main()