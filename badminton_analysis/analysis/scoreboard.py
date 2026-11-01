"""Read true point winners from a broadcast scoreboard, for checking automatic scoring.

No text recognition: it watches the two score-digit boxes (one per player row)
and records which one changes. A change in one box only is a point to that row.
Both boxes changing at once (new game, scoreboard graphic hidden or swapped)
is recorded separately and not counted as a point.

Find the boxes once per broadcast with --preview, then read the match:

    python -m badminton_analysis.analysis.scoreboard video.mp4 --cells 410,42,30,20 410,64,30,20 --preview 5
    python -m badminton_analysis.analysis.scoreboard video.mp4 --cells 410,42,30,20 410,64,30,20 --output truth.json
"""

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ScoreChange:
    time_sec: float
    rows: list
    kind: str  # "point" when one row changed, "both" otherwise


def detect_score_changes(samples, threshold=18.0, stable_samples=3):
    """Find stable changes in the score boxes.

    samples: iterable of (time_sec, [row0_crop, row1_crop]) grayscale arrays.
    threshold: mean absolute pixel difference that counts as a changed box.
    stable_samples: samples the new state must hold before it counts, which
        skips digit-flip animations and brief overlays.
    """
    changes = []
    reference = None
    pending = None
    for time_sec, crops in samples:
        crops = [np.asarray(crop, dtype=np.float32) for crop in crops]
        if reference is None:
            reference = crops
            continue

        changed = [row for row, crop in enumerate(crops) if np.mean(np.abs(crop - reference[row])) > threshold]
        if not changed:
            pending = None
            continue

        same_as_pending = pending is not None and pending["rows"] == changed and all(
            np.mean(np.abs(crop - pending["crops"][row])) <= threshold for row, crop in enumerate(crops)
        )
        if same_as_pending:
            pending["count"] += 1
        else:
            pending = {"rows": changed, "crops": crops, "time": time_sec, "count": 1}

        if pending["count"] >= stable_samples:
            kind = "point" if len(changed) == 1 else "both"
            changes.append(ScoreChange(time_sec=round(pending["time"], 3), rows=changed, kind=kind))
            reference = pending["crops"]
            pending = None
    return changes


def _crop(gray, cell):
    x, y, w, h = cell
    return gray[y:y + h, x:x + w]


def iter_video_cells(video_path, cells, sample_fps=5.0):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    frame_index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame_index / fps, [_crop(gray, cell) for cell in cells]
        frame_index += 1
    cap.release()


def read_scoreboard(video_path, cells, sample_fps=5.0, threshold=18.0, stable_samples=3):
    return detect_score_changes(iter_video_cells(video_path, cells, sample_fps), threshold, stable_samples)


def save_preview(video_path, cells, time_sec, output_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read a frame at {time_sec}s from {video_path}")
    for row, (x, y, w, h) in enumerate(cells):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(frame, str(row + 1), (x + w + 4, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(output_path, frame)


def parse_cell(text):
    x, y, w, h = (int(value) for value in text.split(","))
    return x, y, w, h


def main():
    parser = argparse.ArgumentParser(description="Read point winners from a broadcast scoreboard")
    parser.add_argument("video_path")
    parser.add_argument("--cells", nargs=2, required=True, type=parse_cell, metavar="X,Y,W,H",
                        help="Score-digit boxes for row 1 and row 2, in video pixels")
    parser.add_argument("--preview", type=float, default=None,
                        help="Save scoreboard_preview.png at this time (seconds) with the boxes drawn, then exit")
    parser.add_argument("--output", default="scoreboard_truth.json")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=18.0)
    args = parser.parse_args()

    if args.preview is not None:
        save_preview(args.video_path, args.cells, args.preview, "scoreboard_preview.png")
        print("Saved scoreboard_preview.png")
        return

    changes = read_scoreboard(args.video_path, args.cells, args.sample_fps, args.threshold)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"cells": args.cells, "changes": [asdict(change) for change in changes]}, file, indent=2)
        file.write("\n")
    points = [change for change in changes if change.kind == "point"]
    print(f"Scoreboard: {len(points)} points, {len(changes) - len(points)} other changes -> {args.output}")


if __name__ == "__main__":
    main()
