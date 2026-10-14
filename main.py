import argparse
import datetime
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
FACE_CASCADE_PATH = PROJECT_DIR / "haarcascade_frontalface_default.xml"
SMILE_CASCADE_PATH = PROJECT_DIR / "haarcascade_smile.xml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "selfies"
CAPTURE_INTERVAL_SECONDS = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a selfie when the camera detects a smile."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0; try 1 if needed).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for captured selfies (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def run(camera_index: int, output_dir: Path) -> None:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Install project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
    smile_cascade = cv2.CascadeClassifier(str(SMILE_CASCADE_PATH))
    if face_cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {FACE_CASCADE_PATH}")
    if smile_cascade.empty():
        raise RuntimeError(f"Could not load smile cascade: {SMILE_CASCADE_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check the camera connection "
            "or try another index with --camera."
        )

    next_capture_at = 0.0
    try:
        print("Smile to capture a selfie. Press q to quit.")
        while True:
            success, original_frame = camera.read()
            if not success or original_frame is None:
                raise RuntimeError("Could not read a frame from the camera.")

            frame = original_frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            smiling = False

            for x, y, width, height in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 255),
                    2,
                )
                face_gray = gray[y : y + height, x : x + width]
                face_color = frame[y : y + height, x : x + width]
                smiles = smile_cascade.detectMultiScale(face_gray, 1.3, 25)

                for smile_x, smile_y, smile_width, smile_height in smiles:
                    cv2.rectangle(
                        face_color,
                        (smile_x, smile_y),
                        (smile_x + smile_width, smile_y + smile_height),
                        (0, 0, 255),
                        2,
                    )
                    smiling = True

            now = time.monotonic()
            if smiling and now >= next_capture_at:
                timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S-%f"
                )
                image_path = output_dir / f"selfie-{timestamp}.png"
                if not cv2.imwrite(str(image_path), original_frame):
                    raise RuntimeError(f"Could not save selfie to {image_path}.")
                print(f"Saved {image_path}")
                next_capture_at = now + CAPTURE_INTERVAL_SECONDS

            cv2.imshow("Selfie Capture", frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args.camera, args.output)
    except RuntimeError as error:
        parser.exit(1, f"{parser.prog}: error: {error}\n")


if __name__ == "__main__":
    main()
