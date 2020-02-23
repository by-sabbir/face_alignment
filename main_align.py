import os
import dlib
import cv2 as cv
import numpy as np
from time import time

base_dir = os.path.dirname(__file__)
shape_predictor_file = os.path.join(base_dir, "files", "shape_predictor_5_face_landmarks.dat")
aligned_photos = os.path.join(base_dir, "aligned_photos")
if not os.path.isdir(aligned_photos):
    os.mkdir(aligned_photos)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_file)

COLOR = {
    "green": (128, 255, 128),
    "red": (128, 128, 255),
    "blue": (255, 128, 128),
    "white": (200, 255, 200)
}


def draw_rect(image, rect):
    x, y = rect.tl_corner().x, rect.tl_corner().y
    w, h = rect.width(), rect.height()
    cv.rectangle(image, (x, y), (x + w, y + h), COLOR["green"], 2)


def get_angle(image, shape, type=5):
    if type == 5:
        shapes = np.array(shape.parts())
        left_eye = ((shapes[0].x + shapes[1].x) // 2, (shapes[0].y + shapes[1].y) // 2)
        right_eye = ((shapes[2].x + shapes[3].x) // 2, (shapes[2].y + shapes[3].y) // 2)
        nose = (shapes[4].x, shapes[4].y)
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        cv.circle(image, left_eye, 2, COLOR["white"], -1)
        cv.circle(image, right_eye, 2, COLOR["white"], -1)
        cv.circle(image, nose, 2, COLOR["white"], -1)
        cv.circle(image, eye_center, 5, COLOR["white"], -1)
        
        cv.line(image, left_eye, right_eye, COLOR["blue"], 2)
        cv.line(image, left_eye, nose, COLOR["blue"], 2)
        cv.line(image, right_eye, nose, COLOR["blue"], 2)

        # computing angle
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        cv.putText(image, str(int(abs(angle))) + " deg", (eye_center[0], eye_center[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, COLOR["white"], 1)
        
    return eye_center, angle

def align_img(image, center, angle):
    height, width = image.shape[0], image.shape[1]
    M = cv.getRotationMatrix2D(center, angle, 1)
    return cv.warpAffine(image, M, (width, height))


def main():
    source = 0
    cap = cv.VideoCapture(source)
    cv.namedWindow("window", cv.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        unaligned = np.copy(frame)
        if not ret:
            continue
        # detecting 
        faces = face_detector(frame, 0)
        if len(faces) < 1:
            continue
        
        for face in faces:
            points = shape_predictor(frame, face)
            center, angle = get_angle(frame, points, type=5)
            warped_img = align_img(unaligned, center, angle)
            faces_w = face_detector(warped_img, 0)
            for face_w in faces_w:
                points = shape_predictor(warped_img, face_w)
                center, angle = get_angle(warped_img, points, type=5)

        try:
            cv.imshow("window", warped_img)
        except cv.error:
            raise "no face"

        if cv.waitKey(10) & 0xff == 27:
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
