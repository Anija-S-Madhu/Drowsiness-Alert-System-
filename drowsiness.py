import cv2
import mediapipe as mp
import math
import time
from playsound import playsound
import threading

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]

def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

cap = cv2.VideoCapture(0)

blink_count = 0
blink_detected = False

eye_closed_start = None
alarm_playing = False

def play_alarm():
    global alarm_playing
    playsound("alarm.mp3")
    alarm_playing = False

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

            landmarks = face_landmarks.landmark

            left = landmarks[LEFT_EYE[0]]
            top1 = landmarks[LEFT_EYE[1]]
            top2 = landmarks[LEFT_EYE[2]]
            right = landmarks[LEFT_EYE[3]]
            bottom1 = landmarks[LEFT_EYE[4]]
            bottom2 = landmarks[LEFT_EYE[5]]

            horizontal = distance(left, right)
            vertical = (distance(top1, bottom2) +
                        distance(top2, bottom1)) / 2

            ear = vertical / horizontal

            # Blink Counter
            if ear < 0.22:
                if not blink_detected:
                    blink_count += 1
                    blink_detected = True
            else:
                blink_detected = False

            # --------------------------
            # Drowsiness Detection
            # --------------------------
            if ear < 0.18:

                if eye_closed_start is None:
                    eye_closed_start = time.time()

                closed_time = time.time() - eye_closed_start

                if closed_time >= 2:

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3
                    )

                    if not alarm_playing:
                        alarm_playing = True
                        threading.Thread(
                            target=play_alarm,
                            daemon=True
                        ).start()

            else:
                eye_closed_start = None

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Blinks: {blink_count}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()