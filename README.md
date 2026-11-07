# Eye Blink & Drowsiness Detector

A beginner computer vision project that uses your webcam to watch your eyes.
It uses **OpenCV** for the webcam and drawing, **MediaPipe** face landmarks to
find the eyes, and the **Eye Aspect Ratio (EAR)** formula to tell if the eyes
are open or closed. No model training needed.

**Current step (1):** shows the eye landmarks and the live EAR value.
Blink counting and the drowsiness alarm come in later steps.

## How EAR works

Each eye has 6 points. EAR compares the eye's height to its width:

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```

Open eyes give about 0.25–0.35. Closed eyes drop close to 0.

## Setup (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python drowsiness.py
```

The first run downloads the face model (`face_landmarker.task`, about 4 MB).
Press **q** to quit.

## Settings

Change these at the top of `drowsiness.py`:

- `EAR_THRESHOLD` – below this, the eye counts as closed (default 0.21)
- `DROWSY_SECONDS` – how long closed eyes count as drowsy (default 1.5)
- `CAMERA_INDEX` – which webcam to use (default 0)
