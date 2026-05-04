
# Run this once to download the MediaPipe hand landmarker model:

import urllib.request, pathlib, sys

URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEST = pathlib.Path(__file__).parent / "hand_landmarker.task"

if DEST.exists():
    print(f"Model already present: {DEST}")
    sys.exit(0)

print(f"Downloading {URL} ...")
urllib.request.urlretrieve(URL, DEST)
print(f"Saved to {DEST}  ({DEST.stat().st_size // 1024} KB)")