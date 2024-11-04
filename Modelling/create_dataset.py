import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

DATA_DIR = './test2/data'
NUM_CLASSES = 3
NUM_LANDMARKS = 21

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def extract_landmarks(hand_landmarks):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append(landmark.x)
        landmarks.append(landmark.y)
    return landmarks


data = []
labels = []

for j in range(NUM_CLASSES):  # Iterate through classes
    class_dir = os.path.join(DATA_DIR, str(j))
    for img_path in os.listdir(class_dir):  # Iterate through images in each class
        if img_path.endswith('.jpg'): # Only process JPG images
            left_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
            right_hand_landmarks = [0.0] * (NUM_LANDMARKS * 2)
            
            img = cv2.imread(os.path.join(class_dir, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == "Left":
                        left_hand_landmarks = extract_landmarks(hand_landmarks)
                    elif handedness.classification[0].label == "Right":  # Add "elif" here
                        right_hand_landmarks = extract_landmarks(hand_landmarks)

            features = np.concatenate([left_hand_landmarks, right_hand_landmarks])
            data.append(features)
            labels.append(j)

data = np.array(data)
labels = np.array(labels)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created successfully!")
hands.close()
