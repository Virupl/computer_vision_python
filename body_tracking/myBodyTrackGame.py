import cv2
import time
from bodyTrackingModule import poseDetector

##################################
url = 'http://192.168.1.22:4747/video'
video = 'C:\\Users\\PC\\Documents\\Computer vision\\body_tracking\\images\\video.mp4'
pTime = 0
cTime = 0
##################################

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(video)
# cap.set(3, 640)
# cap.set(4, 480)

detector = poseDetector()

while True:
    succes, img = cap.read()

    if not succes:
        break

    img = cv2.resize(img, (640, 480))

    img = detector.findPose(img)
    lmList = detector.findPosition(img, False)
    if len(lmList) != 0:
        # print(lmList[0])
        x1, y1 = lmList[14][1], lmList[14][2]
        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED, cv2.LINE_4)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"Fps: {int(fps)}", (20, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 20, 255), 2, cv2.LINE_4)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
