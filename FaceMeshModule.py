import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    """
    미디어 파이프 라이브러리를 사용하여 468의 랜드 마크를 찾기위한 페이스 메쉬 검출기.
    픽셀 형식으로 랜드 마크 포인트를 획득하는 데 도움이됩니다
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        """
        :param staticMode: 정적 모드에서는 검출는 각 이미지에서 이루어집니다 : 느린
        :param maxFaces: 감지할 최대 얼굴 수
        :param minDetectionCon: 최소 탐지 신뢰도 임계값
        :param minTrackCon: 최소 추적 신뢰도 임계값
        """
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        BGR 이미지에서 얼굴의 랜드 마크를 검색합니다.
        :param img: 얼굴 랜드마크를 찾을 이미지입니다.
        :param draw: 이미지에 출력을 그리는 플래그입니다.
        :return: 그림이 있거나 없는 이미지
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img,False)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()