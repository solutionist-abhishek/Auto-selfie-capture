from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

from main import (
    DEFAULT_OUTPUT_DIR,
    FACE_CASCADE_PATH,
    build_parser,
    run,
)


class FakeFrame:
    def copy(self) -> "FakeFrame":
        return FakeFrame()

    def __getitem__(self, _key: object) -> "FakeFrame":
        return FakeFrame()


class FakeCascade:
    def __init__(self, is_face_cascade: bool) -> None:
        self.is_face_cascade = is_face_cascade

    def empty(self) -> bool:
        return False

    def detectMultiScale(
        self, _image: FakeFrame, _scale_factor: float, _min_neighbors: int
    ) -> list[tuple[int, int, int, int]]:
        if self.is_face_cascade:
            return [(0, 0, 10, 10)]
        return [(1, 1, 2, 2)]


class FakeCamera:
    def __init__(self, read_success: bool = True) -> None:
        self.read_success = read_success
        self.frame = FakeFrame()
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.read_success:
            return True, self.frame
        return False, None

    def release(self) -> None:
        self.released = True


def make_fake_cv2(
    camera: FakeCamera,
) -> tuple[types.ModuleType, list[tuple[str, FakeFrame]]]:
    module = types.ModuleType("cv2")
    saved_images: list[tuple[str, FakeFrame]] = []
    module.COLOR_BGR2GRAY = 0
    module.CascadeClassifier = lambda path: FakeCascade(
        path == str(FACE_CASCADE_PATH)
    )
    module.VideoCapture = lambda _camera_index: camera
    module.cvtColor = lambda _frame, _color: FakeFrame()
    module.rectangle = lambda *_args: None
    module.imwrite = lambda path, frame: saved_images.append((path, frame)) or True
    module.imshow = lambda *_args: None
    module.waitKey = lambda _delay: ord("q")
    module.destroyAllWindows = lambda: None
    return module, saved_images


class ArgumentParserTests(unittest.TestCase):
    def test_defaults_to_first_camera_and_project_selfies_folder(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.camera, 0)
        self.assertEqual(args.output, DEFAULT_OUTPUT_DIR)

    def test_camera_and_output_can_be_configured(self) -> None:
        args = build_parser().parse_args(
            ["--camera", "2", "--output", "captured"]
        )

        self.assertEqual(args.camera, 2)
        self.assertEqual(args.output, Path("captured"))

    def test_smile_saves_original_frame_and_releases_camera(self) -> None:
        camera = FakeCamera()
        fake_cv2, saved_images = make_fake_cv2(camera)

        with TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"cv2": fake_cv2}
        ):
            output_dir = Path(directory) / "selfies"
            run(0, output_dir)
            self.assertTrue(output_dir.is_dir())

        self.assertTrue(camera.released)
        self.assertEqual(len(saved_images), 1)
        image_path, frame = saved_images[0]
        self.assertTrue(Path(image_path).name.startswith("selfie-"))
        self.assertEqual(Path(image_path).suffix, ".png")
        self.assertIs(frame, camera.frame)

    def test_failed_camera_read_reports_error_and_releases_camera(self) -> None:
        camera = FakeCamera(read_success=False)
        fake_cv2, _ = make_fake_cv2(camera)

        with TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"cv2": fake_cv2}
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not read a frame"):
                run(0, Path(directory))

        self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
