import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # if id == 0:
                cv2.circle(img, (cx, cy), 10, (255, 0, 255),
                           cv2.FILLED, cv2.LINE_4)
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3, cv2.LINE_4)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
