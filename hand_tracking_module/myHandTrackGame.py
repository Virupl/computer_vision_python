import cv2
import mediapipe as mp
import time
import handTrackingModule as htm

pTime = 0
cTime = 0
url = 'http://192.168.92.40:4747/video'
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(url)
detector = htm.handDetector()

while True:
    success, img = cap.read()

    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3, cv2.LINE_4)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
