import cv2
import mediapipe as mp
import argparse


from utils.feature_extraction import *
from utils.strings import *
from utils.model import ASLClassificationModel
from config import MODEL_NAME, MODEL_CONFIDENCE

import streamlit as st

import pyttsx3

# Khởi tạo TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)   # tốc độ đọc
tts_engine.setProperty('volume', 1.0) # âm lượng 0.0 - 1.0

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create message handler
    expression_handler = ExpressionHandler()

    # Streamlit app
    # ===============================
    # 📘 Streamlit giao diện mới
    # ===============================
    st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.2rem;
                text-align: center;
                color: #333;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .sub-title {
                font-size: 1.1rem;
                text-align: center;
                color: #777;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main-title'>🤟 Sign Language Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Ứng dụng nhận diện ký hiệu tay theo thời gian thực</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎥 Camera Input")
        video_placeholder = st.empty()

    with col2:
        st.markdown("### 🧾 Kết quả dự đoán")
        prediction_placeholder = st.empty()

    # Load model
    print("Initialising model ...")
    model = ASLClassificationModel.load_model(r"D:\Sign-Language-Classification-main\models\simple_9_expression_model.pkl")


    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=MODEL_CONFIDENCE,
                                      min_tracking_confidence=MODEL_CONFIDENCE)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=MODEL_CONFIDENCE,
                           min_tracking_confidence=MODEL_CONFIDENCE)

    # Initialize drawing utility
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Starting the application
    print("Starting application")

    # Set up the holistic model
    while cap.isOpened():
        # Check if getting frame is successful
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        face_results = face_mesh.process(image)

        # Process the image and find hands
        hand_results = hands.process(image)

        # Extract feature from face and hand results
        feature = extract_features(mp_hands, face_results, hand_results)
        expression = model.predict(feature)
        expression_handler.receive(expression)

        expression_handler.receive(expression)

        # Lấy message để đọc
        message_to_speak = expression_handler.get_message()

        # Phát âm thanh nếu message khác với lần đọc trước
        if message_to_speak != getattr(expression_handler, "last_spoken", None):
            tts_engine.say(message_to_speak)
            tts_engine.runAndWait()
            expression_handler.last_spoken = message_to_speak


        # Draw the face mesh annotations on the image
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Draw the hand annotations on the image
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Display the image and prediction
        video_placeholder.image(image, channels="RGB", use_column_width=True)
        prediction_placeholder.markdown(f'''<h2 class="big-font">{expression_handler.get_message()}</h2>''', unsafe_allow_html=True)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
