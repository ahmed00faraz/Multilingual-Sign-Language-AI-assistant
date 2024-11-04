import pickle
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = './model.p'
NUM_LANDMARKS = 21

with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.3)  # static_image_mode=False for real-time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

labels_dict = {0: 'Delay', 1: 'Gate', 2: 'I',}  # Replace with your labels


def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
    return landmarks


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
    right_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                left_hand_landmarks = extract_landmarks(hand_landmarks)
                mp_drawing.draw_landmarks(  # Draw left hand landmarks
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            elif handedness.classification[0].label == "Right":
                right_hand_landmarks = extract_landmarks(hand_landmarks)
                mp_drawing.draw_landmarks(  # Draw right hand landmarks
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


    features = np.concatenate([left_hand_landmarks, right_hand_landmarks])
    prediction = model.predict([features])[0]
    predicted_character = labels_dict[prediction]

    cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
