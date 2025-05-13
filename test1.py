import cv2
import time
import numpy as np
import pickle
import mediapipe as mp
from tensorflow.keras.models import load_model
import warnings

warnings.warn("This is a warning message.")
# Load the trained model
model = load_model('model_Tensorflow_Two_Hand_20250405_012045.keras')

# Constants
FIXED_LENGTH = 42
CAPTURE_DURATION = 42  # seconds
MODEL_INPUT_SHAPE = (FIXED_LENGTH, 1)
LABELS = ['115', '116', '130', '131', '132']

# Load normalization values from training (mean and std)
# If you saved these during training, load them. Otherwise, compute from training set.

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
)

def preprocess_landmarks(landmarks):
    """Pad or truncate landmarks to fixed length"""
    if landmarks is None:
        return np.zeros(FIXED_LENGTH, dtype=np.float32)
    arr = np.array(landmarks, dtype=np.float32)
    if len(arr) < FIXED_LENGTH:
        arr = np.pad(arr, (0, FIXED_LENGTH - len(arr)))
    return arr[:FIXED_LENGTH]

def normalize_landmarks(array):
    return (array - np.mean(array)) / (np.std(array) + 1e-8)

def extract_landmarks_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand1_landmarks = [0.0] * FIXED_LENGTH
    hand2_landmarks = [0.0] * FIXED_LENGTH

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, y_min = min(x_coords), min(y_coords)

            normalized = []
            for lm in hand_landmarks.landmark:
                normalized.append(lm.x - x_min)
                normalized.append(lm.y - y_min)

            if i == 0:
                hand1_landmarks = preprocess_landmarks(normalized)
            elif i == 1:
                hand2_landmarks = preprocess_landmarks(normalized)

    return hand1_landmarks, hand2_landmarks

def capture_images_and_predict():
    cap = cv2.VideoCapture(0)
    print("Capturing 42 frames...")

    collected_hand1 = []
    collected_hand2 = []

    start_time = time.time()
    while len(collected_hand1) < CAPTURE_DURATION:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        h1, h2 = extract_landmarks_from_image(frame)
        collected_hand1.append(h1)
        collected_hand2.append(h2)

        cv2.putText(frame, f"Captured: {len(collected_hand1)}/42", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capturing", frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):  # Wait 1 second
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert to np.array and average over time
    hand1_array = normalize_landmarks(np.mean(np.array(collected_hand1), axis=0)).reshape(FIXED_LENGTH, 1)
    hand2_array = normalize_landmarks(np.mean(np.array(collected_hand2), axis=0)).reshape(FIXED_LENGTH, 1)

    # Predict
    prediction = model.predict([np.expand_dims(hand1_array, axis=0),
                                np.expand_dims(hand2_array, axis=0)])

    predicted_label_index = np.argmax(prediction[0])
    predicted_label = LABELS[predicted_label_index]
    print(f"\nPredicted label: {predicted_label} (Confidence: {prediction[0][predicted_label_index]*100:.2f}%)")

if __name__ == "__main__":

    capture_images_and_predict()