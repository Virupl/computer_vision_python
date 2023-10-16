import mediapipe as mp
import cv2


class poseDetector():
    def __init__(self, mode=False, modelCom=1, smLand=True, enSeg=False, smSeg=True, dConf=0.5, tConf=0.5) -> None:
        self.static_image_mode = mode
        self.model_complexity = modelCom
        self.smooth_landmarks = smLand
        self.enable_segmentation = enSeg
        self.smooth_segmentation = smSeg
        self.min_detection_confidence = dConf
        self.min_tracking_confidence = tConf

        self.mp_pose = mp.solutions.pose
        self.poses = self.mp_pose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks,
                                       self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.poses.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255),
                               cv2.FILLED, cv2.LINE_4)

        return lmList
