import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    text = "No Face"

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

            lm = face_landmarks.landmark

            # Face landmarks
            nose = lm[1]
            left_face = lm[234]
            right_face = lm[454]
            forehead = lm[10]
            chin = lm[152]

            # Left / Right
            center_x = (left_face.x + right_face.x) / 2

            if nose.x < center_x - 0.02:
                text = "Looking Left"
            elif nose.x > center_x + 0.02:
                text = "Looking Right"
            else:
                text = "Looking Forward"

            # Up / Down
            center_y = (forehead.y + chin.y) / 2

            if nose.y < center_y - 0.03:
                text = "Looking Up"
            elif nose.y > center_y + 0.03:
                text = "Looking Down"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )

    cv2.imshow("Head Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()