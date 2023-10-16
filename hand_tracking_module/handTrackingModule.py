import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHand=2, model_complexity=1, detectionCon=0.5, trackCon=0.5) -> None:
        self.mode = mode
        self.maxHand = maxHand
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.maxHand, self.model_complexity, self.detectionCon, self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True) -> None:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0),
                               cv2.FILLED, cv2.LINE_4)

        return lmList


# def main():
#     pTime = 0
#     cTime = 0
#     url = 'http://192.168.92.40:4747/video'

#     # cap = cv2.VideoCapture(0)

#     cap = cv2.VideoCapture(url)

#     detector = handDetector()

#     while True:
#         success, img = cap.read()
#         if not success:
#             break

#         img = detector.findHands(img)
#         lmList = detector.findPosition(img)
#         if len(lmList) != 0:
#             print(lmList[4])

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (10, 50),
#                     cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3, cv2.LINE_4)

#         cv2.imshow("Image", img)

#         if cv2.waitKey(1) & 0xff == ord('q'):
#             break


# if __name__ == "__main__":
#     main()
