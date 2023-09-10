import cv2
import mediapipe as mp
import os
import shutil

# gesture_name = "Hamsapakshika"  # Replace with the mudra name
# gesture_name = "Mukula"  # Replace with the mudra name
gesture_name = "Fist"  # Replace with the mudra name
num_samples = 100  # Number of images per mudra

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

frame_captured = 0

if os.path.exists(f"training/{gesture_name}"):    
    shutil.rmtree(f"training/{gesture_name}", ignore_errors=True)

os.makedirs(f"training/{gesture_name}")

while True:
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
            x_min, y_min, x_max, y_max = 1, 1, 0.0, 0.0
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
            
            # Display the cropped hand region
            try:
                # Crop and save the hand region
                hand_image = frame[y_min:y_max, x_min:x_max]
                image_filename = f"training/{gesture_name}/image_{frame_captured}.jpg"            

                cv2.imwrite(image_filename, hand_image)
                frame_captured += 1
                cv2.imshow("Cropped", hand_image)
            except Exception as e:
                print(e)

    cv2.imshow("Cropped Hand Image", frame)
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or frame_captured >= num_samples:
        break

cap.release()
cv2.destroyAllWindows()
