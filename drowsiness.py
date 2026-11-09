"""
Step 1 of the Eye Blink & Drowsiness Detector.

What this script does:
  - Opens the webcam
  - Finds the face and the eye landmarks with MediaPipe (Face Mesh points)
  - Draws the eye points on screen
  - Calculates and shows the live Eye Aspect Ratio (EAR)

Press 'q' to quit.
"""

import os               # to check if the model file already exists
import time             # to give MediaPipe a timestamp for each frame
import urllib.request   # to download the model file the first time

import cv2              # OpenCV: webcam, drawing, showing the window
import mediapipe as mp  # MediaPipe: finds the face landmarks
import numpy as np      # NumPy: easy math on (x, y) points
from mediapipe.tasks.python import BaseOptions, vision

# ---------------------------------------------------------------------------
# SETTINGS (change these to tune the detector)
# ---------------------------------------------------------------------------
EAR_THRESHOLD = 0.21   # if EAR goes below this, the eye counts as "closed"
DROWSY_SECONDS = 2     # eyes closed longer than this many seconds = drowsy
CAMERA_INDEX = 0       # 0 = default webcam; try 1 if you have more than one

# The face landmark model file. MediaPipe needs it to find the face points.
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Landmark numbers (indexes) for each eye in MediaPipe's 468-point face mesh.
# Order matters for the EAR formula: [p1, p2, p3, p4, p5, p6]
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


def eye_aspect_ratio(eye_points):
    """
    Calculate the Eye Aspect Ratio (EAR) for one eye.

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
    """
    Turn MediaPipe's landmarks into pixel (x, y) points for one eye.
    MediaPipe gives x and y as fractions (0.0 to 1.0) of the image size,
    so we multiply by the width/height to get pixel positions.
    """
    points = []
    for index in eye_indexes:
        landmark = face_landmarks[index]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        points.append([x, y])
    return np.array(points)


def main():
    download_model_if_needed()

    # Set up MediaPipe's face landmarker in VIDEO mode (for webcam frames).
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,  # we only need one face
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Open the webcam.
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Could not open the webcam. Try a different CAMERA_INDEX.")
        return

    start_time = time.time()

    # When the eyes first closed (None = eyes are currently open).
    closed_since = None

    while True:
        # 1. Read one frame from the webcam.
        success, frame = camera.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        # Flip the frame so it acts like a mirror (feels more natural).
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        # 2. MediaPipe wants RGB colors, but OpenCV gives BGR. Convert it.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 3. Find the face landmarks. VIDEO mode needs a growing timestamp (ms).
        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 4. If a face was found, work with its landmarks.
        if result.face_landmarks:
            face = result.face_landmarks[0]  # the first (only) face

            left_eye = get_eye_points(face, LEFT_EYE, frame_width, frame_height)
            right_eye = get_eye_points(face, RIGHT_EYE, frame_width, frame_height)

            # Calculate EAR for each eye, then average the two.
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # 5. Draw a small green dot on every eye point.
            for point in np.concatenate([left_eye, right_eye]):
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # 6. Show the EAR value. Red text if below threshold, else green.
            color = (0, 0, 255) if ear < EAR_THRESHOLD else (0, 255, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 7. Track how long the eyes have been continuously closed.
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

            # Closed longer than DROWSY_SECONDS = drowsy (blinks are too short).
            if closed_time > DROWSY_SECONDS:
                cv2.putText(frame, "DROWSY!", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        else:
            # No face: we can't tell if the eyes are closed, so reset the timer.
            closed_since = None
            cv2.putText(frame, "No face found", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 8. Show the frame in a window.
        cv2.imshow("Drowsiness Detector (press q to quit)", frame)

        # 9. Wait 1 ms for a key press; quit if it's 'q'.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up: release the webcam, close the window and the landmarker.
    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()


# Only run main() when this file is run directly (not when imported).
if __name__ == "__main__":
    main()
