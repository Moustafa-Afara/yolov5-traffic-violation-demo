"""
Traffic speed-violation demo built on YOLOv5.

Detects vehicles with YOLOv5 (Ultralytics, pinned to commit 724d5b2, June 2022), gives each
vehicle a stable ID with a small IoU tracker, asks `get_speed()` for that vehicle's speed,
and labels it Legal (green) or Illegal (red) against a speed window.

IMPORTANT: the default speed source is SIMULATED. No speed is measured from the video.
`SimulatedSpeed` stands in for a roadside speed sensor so the labelling pipeline can be
demonstrated end to end. Replace it with a real source (sensor feed, or a camera-based
estimator) by passing any callable with the same signature to `run()`.

Usage:
    python speed_violation_demo.py --source traffic.mp4
    python speed_violation_demo.py --source 0            # webcam
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
YOLOV5 = ROOT / "yolov5"
if not (YOLOV5 / "models").exists():
    sys.exit("yolov5/ is empty. Run:  git submodule update --init")
sys.path.insert(0, str(YOLOV5))

# PyTorch >= 2.6 loads checkpoints with weights_only=True by default; the 2022 YOLOv5 code
# expects the old behaviour. Only load weights you trust (the official release file here).
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{"weights_only": False, **k})

from models.common import DetectMultiBackend  # noqa: E402
from utils.dataloaders import LoadImages, LoadStreams  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_coords  # noqa: E402
from utils.plots import Annotator  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

from tracker import IoUTracker  # noqa: E402

WEIGHTS_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids
GREEN, RED = (0, 200, 0), (0, 0, 255)  # BGR


class SimulatedSpeed:
    """Stand-in for a roadside speed sensor: one plausible speed per vehicle, with small jitter.

    Each new track ID draws a base speed uniformly from [low, high] km/h; every call returns
    that base plus up to +/- `jitter` km/h. Seeded, so a run is reproducible.
    """

    def __init__(self, low=40.0, high=150.0, jitter=3.0, seed=0):
        self.low, self.high, self.jitter = low, high, jitter
        self.rng = random.Random(seed)
        self.base = {}

    def __call__(self, track_id, box, frame_index, fps):
        if track_id not in self.base:
            self.base[track_id] = self.rng.uniform(self.low, self.high)
        return self.base[track_id] + self.rng.uniform(-self.jitter, self.jitter)


def ensure_weights(path: Path) -> Path:
    if not path.exists():
        print(f"Downloading {WEIGHTS_URL} -> {path}")
        torch.hub.download_url_to_file(WEIGHTS_URL, str(path))
    return path


@torch.no_grad()
def run(source, weights, get_speed, min_speed=60.0, max_speed=120.0, imgsz=640, conf_thres=0.35,
        iou_thres=0.45, device="", output="output.mp4", max_frames=None, view=False):
    device = select_device(device)
    model = DetectMultiBackend(str(ensure_weights(Path(weights))), device=device)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    webcam = str(source).isnumeric()
    dataset = (LoadStreams(str(source), img_size=imgsz, stride=stride, auto=model.pt) if webcam
               else LoadImages(str(source), img_size=imgsz, stride=stride, auto=model.pt))
    fps = 30.0
    if not webcam and dataset.cap is not None:
        fps = dataset.cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracker = IoUTracker()
    writer = None
    counts = {"legal": set(), "illegal": set()}

    for frame_index, (path, im, im0s, vid_cap, _) in enumerate(dataset):
        if max_frames is not None and frame_index >= max_frames:
            break
        im = torch.from_numpy(im).to(device).float() / 255.0
        if im.ndim == 3:
            im = im[None]
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=list(VEHICLE_CLASSES))[0]
        frame = (im0s[0] if webcam else im0s).copy()

        boxes, classes = np.zeros((0, 4)), np.zeros(0)
        if len(pred):
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], frame.shape).round()
            boxes, classes = pred[:, :4].cpu().numpy(), pred[:, 5].cpu().numpy()

        annotator = Annotator(frame, line_width=2)
        for track in tracker.update(boxes, classes, frame_index):
            speed = get_speed(track.track_id, track.box, frame_index, fps)
            legal = min_speed <= speed <= max_speed
            counts["legal" if legal else "illegal"].add(track.track_id)
            label = f"#{track.track_id} {'Legal' if legal else 'Illegal'} {speed:.0f} km/h"
            annotator.box_label(track.box, label, color=GREEN if legal else RED)
        frame = annotator.result()
        cv2.putText(frame, f"Speed input: SIMULATED (not measured) | legal window {min_speed:.0f}-{max_speed:.0f} km/h",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if output:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if view:
            cv2.imshow("speed violation demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
    print(f"Vehicles labelled: {len(counts['legal'] | counts['illegal'])} "
          f"(ever legal: {len(counts['legal'])}, ever illegal: {len(counts['illegal'])}). Output: {output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video file, image folder, or webcam index (0)")
    p.add_argument("--weights", default=str(ROOT / "yolov5s.pt"), help="downloaded automatically if missing")
    p.add_argument("--min-speed", type=float, default=60.0, help="below this is Illegal (km/h)")
    p.add_argument("--max-speed", type=float, default=120.0, help="above this is Illegal (km/h)")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="'' auto, 'cpu', or '0' for the first GPU")
    p.add_argument("--output", default="output.mp4", help="annotated video path ('' to disable)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--view", action="store_true", help="show frames in a window (q to quit)")
    p.add_argument("--seed", type=int, default=0, help="seed of the simulated speeds")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.source, a.weights, SimulatedSpeed(seed=a.seed), a.min_speed, a.max_speed, a.imgsz, a.conf,
        device=a.device, output=a.output, max_frames=a.max_frames, view=a.view)
