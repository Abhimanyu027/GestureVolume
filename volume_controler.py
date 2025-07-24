import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Pycaw audio setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_min, vol_max = volume.GetVolumeRange()[:2]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lm_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Volume control logic
    if lm_list:
        x1, y1 = lm_list[4][1:]   # Thumb tip
        x2, y2 = lm_list[8][1:]   # Index finger tip
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        # Draw circles and line
        cv2.circle(img, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(img, (x2, y2), 8, (255, 0, 0), -1)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)

        # Calculate distance
        length = hypot(x2 - x1, y2 - y1)

        # Convert length to volume range
        vol = np.interp(length, [20, 200], [vol_min, vol_max])
        volume.SetMasterVolumeLevel(vol, None)

        # Optional: Display volume bar
        vol_bar = np.interp(length, [20, 200], [400, 150])
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), -1)

    # Show the image
    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
