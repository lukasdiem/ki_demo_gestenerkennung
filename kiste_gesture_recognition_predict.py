import cv2
import numpy as np
import mediapipe as mp
import mediapipe_helper as mp_helper

from PIL import ImageFont, ImageDraw, Image
from path import Path
from keras.models import load_model
from pilmoji import Pilmoji

mp_hands = mp.solutions.hands


model_dir = Path("models")
model_name = "hand_gesture_model_v2"
model_path = model_dir / model_name

model = load_model(model_path.with_suffix(".h5"))

class_names = []
with open(model_path.with_suffix(".csv"), 'r', encoding='utf-8') as file:
    line = file.readline()
    for class_name in line.split(","):
        class_names.append(class_name.strip())


# font = ImageFont.truetype(<font-file>, <font-size>)
font = ImageFont.truetype("fonts/Roboto-Light.ttf", 24)
#font = ImageFont.truetype("arial", 16)

# Open the camera

#  0 ... back camera
#  1 ... front camera
cap = cv2.VideoCapture(1)

detect_hands = True
predict_gesture = True

with mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3
) as hands:

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        frame_vis = frame.copy()

        if not ret:
            print("Error reading the frame")
            break

        if detect_hands:
            results = hands.process(frame)
            frame_vis = mp_helper.draw_results(results, frame_vis)

        if detect_hands and predict_gesture and mp_helper.is_hand_present(results):
            # Get features and predict the gesture
            features = mp_helper.landmark_to_feature(results)
            score = model.predict(np.array([features]), batch_size=1, verbose=0)
            
            pred_label = class_names[np.argmax(score[0])]
            pred_score = np.max(score[0])

            pil_image = Image.fromarray(frame_vis.astype('uint8'), 'RGB')
            pil_draw = ImageDraw.Draw(pil_image)
            
            with Pilmoji(pil_image) as pilmoji:
                pilmoji.text((10, 10), f"Class: {pred_label}\nScore: {pred_score:0.4f}", (255, 255, 255), font=font)
            #pil_draw.text((10, 10), pred_label.encode('utf-8').decode('utf-8'), (255, 255, 255), font=font)
            frame_vis = np.array(pil_image)
            
        # Check if the "r" key is pressed
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            # Toggle prediction
            predict_gesture = not predict_gesture
        elif key == ord("d"):
            # Toggle hand detection
            detect_hands = not detect_hands
        elif key == ord("q"):
            # Check if the "q" key is pressed to exit the loop
            break

        # Display the frame
        cv2.imshow("Camera", frame_vis)


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
