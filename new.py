import threading
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame
import os
import threading
from PIL import Image, ImageFont, ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display
from streamlit_webrtc import webrtc_streamer
from matplotlib import pyplot as plt

# Initialize session state
if 'stop_button' not in st.session_state:
    st.session_state['stop_button'] = False
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = True

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame


ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

fig_place = st.empty()
fig, ax = plt.subplots(1, 1)

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax.cla()
    ax.hist(gray.ravel(), 256, [0, 256])
    fig_place.pyplot(fig)



# Initialize pygame for audio playback
# os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init()
last_played_gesture = None

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sign_language_model (1).h5")

model = load_model()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Arabic gesture classes
sign_language_classes = [
    'الزائدة الدودية', 'العمود الفقري', 'الصدر', 'جهاز التنفس', 'الهيكل العظمي',
    'القصبة الهوائية', 'الوخز بالإبر', 'ضغط الدم', 'كبسولة', 'زكام', 'الجهاز الهضمي',
    'يشرب', 'قطارة', 'أدوية', 'صحي', 'يسمع', 'القلب', 'المناعة', 'يستنشق', 'تلقيح',
    'الكبد', 'دواء', 'ميكروب', 'منغولي', 'عضلة', 'البنكرياس', 'صيدلية', 'البلعوم',
    'إعاقة جسدية', 'فحص جسدي', 'تلقيح النباتات', 'نبض', 'فحص البصر', 'صمت',
    'جمجمة', 'نوم', 'سماعة الطبيب', 'فيروس', 'ضعف بصري', 'استيقاظ', 'جرح'
]

# Process landmarks
def process_landmarks(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def pad_landmarks():
    return [0.0] * 63

# Classify gesture
def classify_gesture(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        combined = process_landmarks(result.multi_hand_landmarks[0])
        if len(result.multi_hand_landmarks) > 1:
            combined.extend(process_landmarks(result.multi_hand_landmarks[1]))
        else:
            combined.extend(pad_landmarks())

        landmarks_array = np.array(combined).reshape(1, -1)
        prediction = model.predict(landmarks_array, verbose=0)
        class_id = np.argmax(prediction[0])
        confidence = prediction[0][class_id]

        return sign_language_classes[class_id], result.multi_hand_landmarks, confidence

    return None, None, None

# Draw Arabic text on frame
def draw_text_with_arabic(frame, text, position, font_path="arial.ttf", font_size=48, color=(0, 255, 0)):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, font_size)

    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    pos = (position[0] - text_width // 2, position[1] - text_height // 2)

    draw.text(pos, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

# Process uploaded image
def process_uploaded_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# Play audio
def play_audio_for_gesture(gesture):
    global last_played_gesture
    audio_path = f"videos/{gesture.strip()}.mp3"

    if gesture and gesture != last_played_gesture and os.path.exists(audio_path):
        last_played_gesture = gesture

        def play():
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
            except Exception as e:
                print("❌ Audio error:", e)

        threading.Thread(target=play, daemon=True).start()
    else:
        print("⚠️ No audio file:", audio_path)

# Main App
def main():
    st.title("نظام التعرف على لغة الإشارة للصم والبكم")
    source = st.radio("اختر مصدر الإدخال:", ["كاميرا الويب", "تحميل صورة"])

    if source == "تحميل صورة":
        uploaded_file = st.file_uploader("اختر صورة من جهازك", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image_bytes = uploaded_file.read()
            frame = process_uploaded_image(image_bytes)
            gesture, hand_landmarks, confidence = classify_gesture(frame)

            if hand_landmarks:
                for landmarks in hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if gesture:
                frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50))
                st.write(f"الإشارة المكتشفة: {gesture}")
                if confidence:
                    st.write(f"نسبة الثقة: {confidence:.2%}")
                play_audio_for_gesture(gesture)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="الصورة المعالجة", use_column_width=True)

    else:  # Webcam
        st.write("اضغط على زر الإيقاف لإنهاء العرض")
        # video_placeholder = st.empty()
        # prediction_placeholder = st.empty()
        # confidence_placeholder = st.empty()
        # stop_button = st.button("إيقاف")
        ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

        # fig_place = st.empty()
        # fig, ax = plt.subplots(1, 1)

        # while ctx.state.playing:
        #     with lock:
        #         img = img_container["img"]
        #     if img is None:
        #         continue
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # ax.cla()
            # ax.hist(gray.ravel(), 256, [0, 256])
            # fig_place.pyplot(fig)

        # cap = cv2.VideoCapture(0)

        # try:
        #     while cap.isOpened() and not stop_button:
        #         ret, frame = cap.read()
        #         if not ret:
        #             st.error("فشل في تشغيل الكاميرا")
        #             break

        #         gesture, hand_landmarks, confidence = classify_gesture(frame)

        #         if hand_landmarks:
        #             for landmarks in hand_landmarks:
        #                 mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        #         if gesture:
        #             frame = draw_text_with_arabic(frame, f"الإشارة: {gesture}", (frame.shape[1] // 2, 50))
        #             prediction_placeholder.text(f"الإشارة المكتشفة: {gesture}")
        #             if confidence:
        #                 confidence_placeholder.write(f"نسبة الثقة: {confidence:.2%}")
        #             play_audio_for_gesture(gesture)

        #         video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # finally:
        #     cap.release()
        #     st.session_state['camera_running'] = False

if __name__ == "__main__":
    main()
