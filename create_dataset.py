import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_collection.log'
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.6,
    max_num_hands=2
)

DATA_DIR = 'data'
LANDMARKS_PER_HAND = 42
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


def process_image(image_path):
    """Process a single image and extract hand landmarks."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.warning(f"Could not load image: {image_path}")
            return None, None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            logging.debug(f"No hands detected in: {image_path}")
            return None, None

        hand1_landmarks = []
        hand2_landmarks = []
        label = os.path.basename(os.path.dirname(image_path))

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, y_min = min(x_coords), min(y_coords)

            normalized_landmarks = []
            for lm in hand_landmarks.landmark:
                normalized_landmarks.append(lm.x - x_min)
                normalized_landmarks.append(lm.y - y_min)

            if hand_idx == 0:
                hand1_landmarks = normalized_landmarks
            elif hand_idx == 1:
                hand2_landmarks = normalized_landmarks

        if not hand1_landmarks:
            hand1_landmarks = [0] * LANDMARKS_PER_HAND
        if not hand2_landmarks:
            hand2_landmarks = [0] * LANDMARKS_PER_HAND

        data_entry = [hand1_landmarks, hand2_landmarks, label]
        return data_entry, label

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None, None


def collect_data():
    """Collect hand landmark data from images in DATA_DIR."""
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory not found: {DATA_DIR}")
        return None

    data = []
    total_images = 0

    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(dir_path):
            total_images += len([f for f in os.listdir(dir_path)
                                 if os.path.splitext(f.lower())[1] in ALLOWED_EXTENSIONS])

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for dir_ in os.listdir(DATA_DIR):
            dir_path = os.path.join(DATA_DIR, dir_)
            if not os.path.isdir(dir_path):
                continue

            for img_name in os.listdir(dir_path):
                if os.path.splitext(img_name.lower())[1] not in ALLOWED_EXTENSIONS:
                    continue

                img_path = os.path.join(dir_path, img_name)
                data_entry, label = process_image(img_path)

                if data_entry is not None:
                    data.append(data_entry)
                    logging.debug(f"Processed {img_name} - Label: {label}")

                pbar.update(1)

    return data


def main():
    """Main function to execute data collection and saving."""
    try:
        data = collect_data()

        if not data:
            logging.error("No data collected. Exiting.")
            return

        output_file = 'data_two_hands_structured.pickle'
        with open(output_file, 'wb') as f:
            pickle.dump({'data': data}, f)

        logging.info(f"Successfully saved {len(data)} samples to {output_file}")
        print(f"Done! Saved {len(data)} samples to {output_file}")

        if data:
            print("\nSample data entry:")
            print(f"Hand 1 landmarks (first 5): {data[0][0][:5]}...")
            print(f"Hand 2 landmarks (first 5): {data[0][1][:5]}...")
            print(f"Label: {data[0][2]}")

        unique_labels = set(entry[2] for entry in data)
        print("\nDataset Statistics:")
        for label in unique_labels:
            count = sum(1 for entry in data if entry[2] == label)
            print(f"Label '{label}': {count} samples")

    except Exception as e:
        logging.error(f"Main execution error: {e}")
        print(f"Error occurred. Check data_collection.log for details.")


if __name__ == "__main__":
    main()

hands.close()