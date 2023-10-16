import cv2
import mediapipe as mp
import pyautogui
import time

##################################
url = 'http://192.168.1.22:4747/video'
video = 'body_tracking/images/video.mp4'
pTime = 0
cTime = 0
##################################

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(video)
cap.set(3, 640)
cap.set(4, 480)

mp_pose = mp.solutions.pose
poses = mp_pose.Pose(static_image_mode=False,
                     model_complexity=1,
                     smooth_landmarks=True,
                     enable_segmentation=False,
                     smooth_segmentation=True,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


while True:
    succes, img = cap.read()

    if not succes:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgRGB = img

    results = poses.process(imgRGB)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape

            cx, cy = int(lm.x*w), int(lm.y*h)
            # print(id, cx, cy)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0),
                       cv2.FILLED, cv2.LINE_4)

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
