"""
A small IoU tracker: keeps the same ID on the same object from frame to frame.

How it works
------------
Each frame, every detection is matched to the existing track whose last box overlaps it
most (intersection-over-union, IoU). Matching is greedy, highest IoU first, and a match
is only accepted above `iou_threshold`. Unmatched detections start new tracks; tracks
that go unmatched for more than `max_age` frames are dropped. A track is reported only
after it has been seen `min_hits` times, which suppresses one-frame false detections.

This is the simplest tracker that gives each vehicle a stable ID. It has no motion model
(unlike SORT's Kalman filter), so it can swap IDs when two vehicles cross closely.
"""
from dataclasses import dataclass, field

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in `a` (N,4) and every box in `b` (M,4), boxes as x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    cls: int
    hits: int = 1
    missed: int = 0
    history: list = field(default_factory=list)  # (frame_index, box) pairs


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: np.ndarray, classes: np.ndarray, frame_index: int) -> list[Track]:
        """Feed this frame's detections; returns the confirmed tracks visible in this frame."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        classes = np.asarray(classes, dtype=int).reshape(-1)
        track_boxes = np.array([t.box for t in self.tracks]).reshape(-1, 4)
        ious = iou_matrix(track_boxes, boxes)

        matched_tracks, matched_dets = set(), set()
        for flat in np.argsort(-ious, axis=None):
            ti, di = np.unravel_index(flat, ious.shape)
            if ious[ti, di] < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            t = self.tracks[ti]
            t.box, t.cls, t.hits, t.missed = boxes[di], int(classes[di]), t.hits + 1, 0
            t.history.append((frame_index, boxes[di].copy()))
            matched_tracks.add(ti)
            matched_dets.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        for di in range(len(boxes)):
            if di not in matched_dets:
                t = Track(self._next_id, boxes[di], int(classes[di]))
                t.history.append((frame_index, boxes[di].copy()))
                self.tracks.append(t)
                self._next_id += 1

        return [t for t in self.tracks if t.missed == 0 and t.hits >= self.min_hits]
