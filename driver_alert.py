import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import sys

# ---------- TTS (speak once in a safe thread) ----------
def speak_once(text):
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except:
            pass

    threading.Thread(target=_run, daemon=True).start()

WARNING_TEXT = "Attention! You are driving. If you want to sleep, kindly stop your driving."

# ---------- MediaPipe Setup ----------
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except AttributeError:
    print("ERROR: Please rename your file! It cannot be 'mediapipe.py'")
    sys.exit()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22
SLEEP_FRAMES = 15
counter = 0
already_warned = False   # ✅ ye ensure karega har cycle me 1 hi dafa bole

def eye_aspect_ratio(eye_points):
    v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    h  = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (v1 + v2) / (2.0 * h)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # agar face detect na ho
    if not results.multi_face_landmarks:
        counter = 0
        already_warned = False
        cv2.putText(frame, "Face not detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Voice Alert Assistant", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    mesh_points = results.multi_face_landmarks[0].landmark

    l_eye = [(int(mesh_points[i].x * w), int(mesh_points[i].y * h)) for i in LEFT_EYE]
    r_eye = [(int(mesh_points[i].x * w), int(mesh_points[i].y * h)) for i in RIGHT_EYE]

    ear = (eye_aspect_ratio(l_eye) + eye_aspect_ratio(r_eye)) / 2.0

    if ear < EAR_THRESHOLD:
        counter += 1

        # ✅ sirf us moment par speak jab threshold cross ho
        if counter >= SLEEP_FRAMES and not already_warned:
            already_warned = True
            speak_once(WARNING_TEXT)

        if counter >= SLEEP_FRAMES:
            cv2.putText(frame, "!!! STOP DRIVING !!!", (70, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

    else:
        # eyes open => reset
        counter = 0
        already_warned = False

    color = (0, 255, 0) if ear >= EAR_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, f"EAR: {ear:.2f} | Frames: {counter}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Voice Alert Assistant", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
