import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Concatenate, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.regularizers import l2
import logging
from datetime import datetime

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="training.log",
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
try:
    with open("data_two_hands_structured.pickle", "rb") as f:
        data_dict = pickle.load(f)
    data = data_dict["data"]
    logging.info(f"Loaded {len(data)} data samples")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    raise

# Extract hand landmarks and labels
hand1_data = [entry[0] for entry in data]
hand2_data = [entry[1] for entry in data]
labels = [entry[2] for entry in data]

# Encode labels
unique_labels = sorted(set(labels))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
labels_array = np.array([label_to_idx[label] for label in labels], dtype=np.int32)
logging.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")

# Fixed landmark length
fixed_length = 42

# Function to preprocess landmarks
def preprocess_landmarks(landmarks, fixed_length):
    """Preprocess landmarks to a fixed length with padding/truncation."""
    try:
        landmarks = np.array(landmarks, dtype=np.float32)
        current_length = len(landmarks)
        if current_length == fixed_length:
            return landmarks
        elif current_length < fixed_length:
            return np.pad(
                landmarks, (0, fixed_length - current_length), mode="constant", constant_values=0
            )
        else:
            return landmarks[:fixed_length]
    except Exception as e:
        logging.error(f"Error in preprocessing landmarks: {e}")
        return None

# Process hand data
processed_hand1, processed_hand2, valid_indices = [], [], []

for i, (h1, h2) in enumerate(zip(hand1_data, hand2_data)):
    h1_processed = preprocess_landmarks(h1, fixed_length)
    h2_processed = preprocess_landmarks(h2, fixed_length)
    if h1_processed is not None and h2_processed is not None:
        processed_hand1.append(h1_processed)
        processed_hand2.append(h2_processed)
        valid_indices.append(i)

# Update labels
labels_array = labels_array[valid_indices]
logging.info(f"Processed {len(processed_hand1)} valid samples")

# Convert to NumPy arrays
hand1_array = np.array(processed_hand1, dtype=np.float32)
hand2_array = np.array(processed_hand2, dtype=np.float32)

# Normalize data
hand1_array = (hand1_array - np.mean(hand1_array)) / (np.std(hand1_array) + 1e-8)
hand2_array = (hand2_array - np.mean(hand2_array)) / (np.std(hand2_array) + 1e-8)

# Reshape for model input
hand1_array = np.expand_dims(hand1_array, axis=-1)
hand2_array = np.expand_dims(hand2_array, axis=-1)

print(f"Hand 1 data shape: {hand1_array.shape}")
print(f"Hand 2 data shape: {hand2_array.shape}")
print(f"Labels shape: {labels_array.shape}")

# Split dataset
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    hand1_array, hand2_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array
)

# Convert labels to categorical
num_classes = len(unique_labels)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define model inputs
input_hand1 = Input(shape=(fixed_length, 1), name="hand1_input")
input_hand2 = Input(shape=(fixed_length, 1), name="hand2_input")

# Define first branch
h1 = Flatten()(input_hand1)
h1 = Dense(256, activation="relu", kernel_regularizer=l2(0.005))(h1)
h1 = BatchNormalization()(h1)
h1 = Dropout(0.3)(h1)

# Define second branch
h2 = Flatten()(input_hand2)
h2 = Dense(256, activation="relu", kernel_regularizer=l2(0.005))(h2)
h2 = BatchNormalization()(h2)
h2 = Dropout(0.3)(h2)

# Merge branches
combined = Concatenate()([h1, h2])
x = Dense(512, activation="relu", kernel_regularizer=l2(0.005))(combined)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu", kernel_regularizer=l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
output = Dense(num_classes, activation="softmax")(x)

# Create model
model = Model(inputs=[input_hand1, input_hand2], outputs=output)
model.summary()

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy", "Precision", "Recall"],
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.2,
    patience=10,
    min_lr=0.000001,
    mode="max",
    verbose=1,
)

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)

# Train model
history = model.fit(
    [x1_train, x2_train],
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=([x1_test, x2_test], y_test),
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1,
)

# Evaluate model
test_metrics = model.evaluate([x1_test, x2_test], y_test, verbose=2)
test_loss, test_accuracy, test_precision, test_recall = test_metrics
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Precision: {test_precision * 100:.2f}%")
print(f"Test Recall: {test_recall * 100:.2f}%")

# Generate classification report
y_pred = model.predict([x1_test, x2_test])
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=unique_labels, zero_division=1))

# Save final model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model.save(f"model_Tensorflow_2Hand_{timestamp}.keras")
logging.info(f"Final model saved as model_Tensorflow_Two_Hand_{timestamp}.keras")
