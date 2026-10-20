import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Mouth landmarks
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_LIP = 13
BOTTOM_LIP = 14


def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

            lm = face_landmarks.landmark

            left = lm[LEFT_MOUTH]
            right = lm[RIGHT_MOUTH]
            top = lm[TOP_LIP]
            bottom = lm[BOTTOM_LIP]

            mouth_width = distance(left, right)
            mouth_height = distance(top, bottom)
            if mouth_height >0.01:
                smile_ratio = mouth_width / mouth_height
            else:
                smile_ratio=0

            if smile_ratio > 4.0:
                text = "Smiling ..."
                color = (0, 0, 0)
            else:
                text = "Not Smiling"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.putText(
                frame,
                f"Ratio: {smile_ratio:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()