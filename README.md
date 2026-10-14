# Selfie Capture

A small Python app that uses your webcam to detect a face and smile, then saves
a selfie automatically.

## Requirements

- Python 3.9 or newer
- A webcam

## Setup

From the project folder, create and activate a virtual environment, then install
the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with
`source .venv/bin/activate`.

## Run

```powershell
python main.py
```

The app opens camera 0 by default. If that is not the webcam you want, pass a
different device index:

```powershell
python main.py --camera 1
```

Smile at the camera to save a PNG image. While you keep smiling, the app saves
at most one image per second. Images go into the `selfies` folder beside the
project by default; choose another directory with `--output`. Press **q** in
the camera window to quit.

The face and smile cascade XML files must stay beside `main.py`. Captured images
are stored locally and are not uploaded.

## Tests

Run the camera-independent tests with:

```powershell
python -m unittest discover -v
```
