
import cv2
import numpy as np
import time
import threading
from collections import deque
from deepface import DeepFace
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pygame

# --- Spotify Setup ---
CLIENT_ID = ''
CLIENT_SECRET = ''
REDIRECT_URI = 'http://127.0.0.1:8888/callback'
SCOPE = 'user-read-playback-state user-modify-playback-state user-read-currently-playing'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
))

emotion_playlist_map = {
    'happy': '2tSY2OBJjXInxUd65udRWm',
    'sad': '2GoMV9Kkd1xU0FMSKw3zpX',
    'neutral': '4dlVxx6xFW0NCCw0OSzPVT',
    'angry': '1a72eCY0tXqNNmTuarsWjF',
}

def get_active_device():
    devices = sp.devices()['devices']
    return devices[0]['id'] if devices else None

def play_playlist(emotion):
    if emotion in emotion_playlist_map:
        playlist_id = emotion_playlist_map[emotion]
        device_id = get_active_device()
        if device_id:
            sp.start_playback(device_id=device_id, context_uri=f"spotify:playlist:{playlist_id}")

def pause_music():
    device_id = get_active_device()
    if device_id:
        sp.pause_playback(device_id=device_id)

def get_current_track():
    playback = sp.current_playback()
    if playback and playback['is_playing']:
        track = playback['item']
        return f"{track['name']} - {', '.join(artist['name'] for artist in track['artists'])}"
    return "No song playing"

# --- Pygame Alert Sound ---
pygame.mixer.init()
alert_thread = None
stop_alarm = threading.Event()

def play_alert_sound():
    def loop():
        while not stop_alarm.is_set():
            pygame.mixer.music.load("alarm.wav")
            pygame.mixer.music.set_volume(1.0)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not stop_alarm.is_set():
                time.sleep(0.1)
    global alert_thread
    if not alert_thread or not alert_thread.is_alive():
        stop_alarm.clear()
        alert_thread = threading.Thread(target=loop, daemon=True)
        alert_thread.start()

def stop_alert_sound():
    stop_alarm.set()
    pygame.mixer.music.stop()

# --- Mediapipe EAR Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_ear(landmarks, eye):
    pts = np.array([landmarks[i] for i in eye])
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return vertical / (2.0 * horizontal)

# --- Emotion Detection ---
def scan_emotion(cap):
    valid_emotions = {'happy', 'sad', 'neutral', 'angry'}  
    start_time = time.time()
    detected_emotion = ""
    while time.time() - start_time < 8:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=True)
                detected_emotion = analysis[0]['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Emotion Detection Error: {e}")
        elapsed = int(6 - (time.time() - start_time))
        cv2.putText(frame, f"Scanning... {elapsed}s", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Emotion Scan", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return ""
    return detected_emotion

# --- OpenCV Menu ---
cap = cv2.VideoCapture(0)
menu_displayed = True
choice_made = False

while not choice_made:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, "Press 1: Scan for Emotion & Play Music", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Press 2: Quit", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.imshow("Emotion & Drowsiness Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        choice_made = True
    elif key == ord('2'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

frame_window = deque(maxlen=15)
is_alerting = False
current_emotion = scan_emotion(cap)
play_playlist(current_emotion)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face.landmark])
        ear = (compute_ear(landmarks, LEFT_EYE) + compute_ear(landmarks, RIGHT_EYE)) / 2.0
        frame_window.append(ear < 0.2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if len(frame_window) == frame_window.maxlen and all(frame_window):
            cv2.putText(frame, "DROWSINESS DETECTED!", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if not is_alerting:
                play_alert_sound()
                is_alerting = True
        for idx in LEFT_EYE + RIGHT_EYE:
            x, y = landmarks[idx].astype(int)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    track_display = get_current_track()
    cv2.putText(frame, f"Emotion: {current_emotion}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (204, 51, 0), 2)
    cv2.putText(frame, f"Now Playing: {track_display}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 102, 0), 2)
    cv2.putText(frame, "Press 's' to stop alarm | 'e' to rescan | 'q' to quit", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 102), 2)

    cv2.imshow("Emotion & Drowsiness Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        stop_alert_sound()
        is_alerting = False
    elif key == ord('e'):
        current_emotion = scan_emotion(cap)
        play_playlist(current_emotion)

cap.release()
cv2.destroyAllWindows()
stop_alert_sound()
