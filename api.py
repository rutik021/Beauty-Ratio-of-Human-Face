import cv2
import GoldenFace.goldenMath as goldenMath
import GoldenFace.functions as functions
import GoldenFace.landmark as landmark

class GoldenFace:

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.image_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel("D:/cv2/landmark.yaml")  # Ensure the path to your model file is correct
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.faces = self.face_detector.detectMultiScale(self.image_gray, 1.3, 5)

        for faceBorders in self.faces:
            (x, y, w, h) = faceBorders
            self.faceBorders = faceBorders
            _, self.landmarks = self.landmark_detector.fit(self.image_gray, [faceBorders])
            self.facePoints = landmark.detect_landmark(self.landmarks[0])
            break

    def draw_face_cover(self, color):
        (x, y, w, h) = self.faceBorders
        self.img = cv2.rectangle(self.img, (x, y), (x + w, y + h), color, 2)

    def calculate_TGSM(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        return goldenMath.calculate_TGSM(self.faceBorders, self.facePoints)

    def calculate_VFM(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        return goldenMath.calculate_VFM(self.faceBorders, self.facePoints)

    def calculate_TZM(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        return goldenMath.calculate_TZM(self.faceBorders, self.facePoints)

    def calculate_TSM(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        return goldenMath.calculate_TSM(self.faceBorders, self.facePoints)

    def calculate_LC(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        return goldenMath.calculate_LC(self.faceBorders, self.facePoints)

    def geometric_ratio(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        TZM = goldenMath.calculate_TZM(self.faceBorders, self.facePoints)
        TGSM = goldenMath.calculate_TGSM(self.faceBorders, self.facePoints)
        VFM = goldenMath.calculate_VFM(self.faceBorders, self.facePoints)
        TSM = goldenMath.calculate_TSM(self.faceBorders, self.facePoints)
        LC = goldenMath.calculate_LC(self.faceBorders, self.facePoints)

        avg = (TZM + TGSM + VFM + TSM + LC) / 5
        return 100 - avg

    def face_to_vec(self):
        goldenMath.unit_size = goldenMath.calculate_unit(self.facePoints)
        vector = goldenMath.face_to_vec(self.faceBorders, self.facePoints)
        return vector

    def face_similarity(self, vector2):
        return goldenMath.vector_face_similarity(self.face_to_vec(), vector2)

    def similarity_ratio(self):
        facevec = self.face_to_vec()
        goldenFace = functions.load_face_vec("goldenFace.json")
        similarity = goldenMath.vector_face_similarity(facevec, goldenFace)
        return similarity

    def get_landmarks(self):
        return self.landmarks

    def get_facial_points(self):
        return self.facePoints

    def draw_facial_points(self, color):
        self.img = goldenMath.draw_facial_points(self.img, self.facePoints, color)

    def draw_landmarks(self, color):
        self.img = goldenMath.draw_landmarks(self.img, self.landmarks, color)

    def get_face_border(self):
        return self.faceBorders

    def write_image(self, name):
        cv2.imwrite(name, self.img)

    def save_face_vec(self, path):
        functions.save_face_vec(self.face_to_vec(), path)
