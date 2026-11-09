
import os               
import threading        
import time              
import urllib.request   
import winsound         

import cv2              
import mediapipe as mp  
import numpy as np      
from mediapipe.tasks.python import BaseOptions, vision



EAR_THRESHOLD = 0.21   # if EAR goes below this, the eye counts as "closed"
DROWSY_SECONDS = 2     # eyes closed longer than this many seconds = drowsy
ALARM_FREQUENCY = 1000 # pitch of the alarm beep in Hz
ALARM_DURATION = 400   # length of each beep in milliseconds
CAMERA_INDEX = 0       # 0 = default webcam; try 1 if you have more than one

# The face landmark model file. MediaPipe needs it to find the face points.
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Landmark numbers (indexes) for each eye in MediaPipe's 468-point face mesh.
# [p1, p2, p3, p4, p5, p6]
#   p1 = outer corner, p4 = inner corner (the horizontal pair)
#   p2/p6 and p3/p5   = top/bottom pairs (the vertical pairs)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def download_model_if_needed():
    """Download the face landmark model once, if it isn't already here."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model (one time only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def alarm_loop(alarm_on):
    
    while True:
        alarm_on.wait()  # sleep here until the alarm is switched on
        winsound.Beep(ALARM_FREQUENCY, ALARM_DURATION)


def eye_aspect_ratio(eye_points):
    """
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Open eye  -> EAR is larger (about 0.25 - 0.35)
    Closed eye -> EAR drops close to 0
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = np.linalg.norm(p2 - p6)    # distance between top and bottom
    vertical_2 = np.linalg.norm(p3 - p5)    # second top/bottom distance
    horizontal = np.linalg.norm(p1 - p4)    # distance corner to corner
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_eye_points(face_landmarks, eye_indexes, frame_width, frame_height):
    
    points = []
    for index in eye_indexes:
        landmark = face_landmarks[index]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        points.append([x, y])
    return np.array(points)


def main():
    download_model_if_needed()

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,  # we only need one face
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # webcam.
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Could not open the webcam. Try a different CAMERA_INDEX.")
        return

    start_time = time.time()

    # When the eyes first closed (None = eyes are currently open).
    closed_since = None


    alarm_on = threading.Event()
    threading.Thread(target=alarm_loop, args=(alarm_on,), daemon=True).start()

    while True:
        
        success, frame = camera.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        # Flipping the frame 
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # face was found, work with its landmarks.
        if result.face_landmarks:
            face = result.face_landmarks[0]  # the first (only) face

            left_eye = get_eye_points(face, LEFT_EYE, frame_width, frame_height)
            right_eye = get_eye_points(face, RIGHT_EYE, frame_width, frame_height)

            # Calculate EAR for each eye, then average the two.
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # green dot on every eye point.
            for point in np.concatenate([left_eye, right_eye]):
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            #EAR value. Red text if below threshold, else green.
            color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Tracking how long the eyes have been continuously closed.
            if ear < EAR_THRESHOLD:
                if closed_since is None:
                    closed_since = time.time()  # eyes just closed: start timer
                closed_time = time.time() - closed_since
            else:
                closed_since = None             # eyes open: reset the timer
                closed_time = 0.0

            # Show the closed-eye time so you can check that it works.
            cv2.putText(frame, f"Closed: {closed_time:.1f}s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            
            if closed_time > DROWSY_SECONDS:
                alarm_on.set()
                cv2.putText(frame, "DROWSY!", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                alarm_on.clear()
        else:
            # No face
            # timer and stop the alarm.
            closed_since = None
            alarm_on.clear()
            cv2.putText(frame, "No face found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        cv2.imshow("Drowsiness Detector (press q to quit)", frame)

        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    
    alarm_on.clear()
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
