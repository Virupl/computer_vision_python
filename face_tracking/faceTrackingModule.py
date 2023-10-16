import cv2
import mediapipe as mp


class faceDetector():
    def __init__(self, mode=False, maxNumFace=1, refland=False, dConf=0.5, tConf=0.5) -> None:
        self.static_image_mode = mode
        self.max_num_faces = maxNumFace
        self.refine_landmarks = refland
        self.min_detection_confidence = dConf
        self.min_tracking_confidence = tConf

        self.mpFace = mp.solutions.face_mesh
        self.faces = self.mpFace.FaceMesh(self.static_image_mode, self.max_num_faces,
                                          self.refine_landmarks, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faces.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, self.mpFace.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape

                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 1, (0, 0, 255),
                                   cv2.FILLED, cv2.LINE_4)

        return lmList
