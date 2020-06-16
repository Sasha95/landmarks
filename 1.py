import cv2
import dlib

img = cv2.imread('2020-06-17_01-46.png', cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = detector(img)

for face in faces:

    cv2.putText(img, "{} face(s) found".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    landmarks = predictor(img, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

cv2.imshow("Frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
