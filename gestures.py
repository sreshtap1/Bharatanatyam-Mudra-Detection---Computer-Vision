import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from config import *

# Load the trained model
model = tf.keras.models.load_model(path_to_model)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands in the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of the hand
            x_min, y_min, x_max, y_max = 1.0, 1.0, 0.0, 0.0
            for landmark in hand_landmarks.landmark:
                x_min = min(x_min, landmark.x)
                x_max = max(x_max, landmark.x)
                y_min = min(y_min, landmark.y)
                y_max = max(y_max, landmark.y)

            # Convert normalized coordinates to pixel coordinates
            height, width, _ = frame.shape
            x_min = int(x_min * width) - 30
            x_max = int(x_max * width) + 20
            y_min = int(y_min * height) - 30
            y_max = int(y_max * height) + 20  

            # Crop and preprocess the hand region
            hand_image = frame[y_min:y_max, x_min:x_max]
            try: 
                resized_hand = cv2.resize(hand_image, input_shape)
                preprocessed_hand = np.expand_dims(resized_hand, axis=0) / 255.0
                
                # Make predictions only if hand is detected
                predictions = model.predict(preprocessed_hand)
                predicted_mudra = mudra_names[np.argmax(predictions)]
                
                # Display the predicted mudra on the frame
                cv2.putText(frame, predicted_mudra, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                pass
    # Display the frame with predictions
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
