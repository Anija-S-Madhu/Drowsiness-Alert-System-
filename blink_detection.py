import cv2
import mediapipe as mp
import math

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# Left eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Distance function
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Webcam
cap = cv2.VideoCapture(0)

blink_count = 0
blink_detected = False

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    print(results.multi_face_landmarks)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            landmarks = face_landmarks.landmark

            # Eye points
            left = landmarks[LEFT_EYE[0]]
            top1 = landmarks[LEFT_EYE[1]]
            top2 = landmarks[LEFT_EYE[2]]
            right = landmarks[LEFT_EYE[3]]
            bottom1 = landmarks[LEFT_EYE[4]]
            bottom2 = landmarks[LEFT_EYE[5]]

            # Calculate distances
            horizontal = distance(left, right)
            vertical = (distance(top1, bottom2) + distance(top2, bottom1)) / 2

            # Eye Aspect Ratio (EAR)
            ear = vertical / horizontal

            # Blink detection
            if ear < 0.22:
                if not blink_detected:
                    blink_count += 1
                    blink_detected = True
            else:
                blink_detected = False

            # Draw face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

            # Display EAR
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Display Blink Count
            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()