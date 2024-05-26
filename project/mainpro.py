import cv2
import mediapipe as mp
import pyautogui

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Capture video from the webcam
cap = cv2.VideoCapture(1)

# Variables to store previous hand position for swipe detection
prev_x = None
prev_y = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 6)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            handedness = results.multi_handedness[results.multi_hand_landmarks.index(landmarks)].classification[0].label

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mid = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mid = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            if handedness == "Left":  # Left hand controls the mouse
                mcp_x = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                mcp_y = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y

                cursor_x = int(mcp_x * screen_width)
                cursor_y = int(mcp_y * screen_height)

                pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

                if index_tip.y >= index_mid.y:
                    pyautogui.click()

            elif handedness == "Right":  # Right hand controls the keyboard
                # Check if both index and middle fingers are raised
                if index_tip.y < index_mid.y and middle_tip.y < middle_mid.y:
                    x, y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
                    if prev_x is not None and prev_y is not None:
                        dx = x - prev_x
                        dy = y - prev_y

                        if abs(dx) > abs(dy):
                            if dx > 50:  # Swipe right
                                pyautogui.press('right')
                            elif dx < -50:  # Swipe left
                                pyautogui.press('left')
                        else:  # Vertical swipe
                            if dy > 50:  # Swipe down
                                pyautogui.press('down')
                            elif dy < -50:  # Swipe up
                                pyautogui.press('up')

                    prev_x = x
                    prev_y = y

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
