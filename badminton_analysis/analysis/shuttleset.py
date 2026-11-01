"""Check line calls and automatic scoring against ShuttleSet labels.

ShuttleSet (https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet,
MIT licence) labels every rally of 44 broadcast singles matches with its
winner, why it ended and where each shot landed. The videos themselves are
YouTube uploads listed in match.csv and are not part of the dataset; the
labels only line up with that exact upload.

    # 1. Download the labels (small CSV files)
    python -m badminton_analysis.analysis.shuttleset fetch --dest data/shuttleset
    # 2. Before analysing the video, write the labelled court corners for it
    python -m badminton_analysis.analysis.shuttleset court data/shuttleset <match> outputs/<video_name> --video <video.mp4>
    # 3. Analyse the video with main.py using the same output directory, then:
    python -m badminton_analysis.analysis.shuttleset evaluate data/shuttleset <match> outputs/<video_name>
"""

import argparse
import csv
import json
import os
import urllib.request
from dataclasses import asdict, dataclass

import numpy as np

from ..court.mapper import CourtMapper, compute_expanded_roi
from .line_calls import NET_Y, SIDE_LABELS, LineCallConfig, call_landing, compute_line_calls, load_run
from .scoring import OTHER_SIDE, rally_winner_side


REPO_API = "https://api.github.com/repos/wywyWang/CoachAI-Projects/contents/ShuttleSet/set"
REPO_RAW = "https://raw.githubusercontent.com/wywyWang/CoachAI-Projects/main/ShuttleSet/set"
LABEL_SIZE = (1280, 720)  # frame size the ShuttleSet pixel labels are given in

LOSE_REASONS = {
    "出界": "out",              # landed out
    "掛網": "net",              # hit the net
    "未過網": "net",            # did not cross the net
    "對手落地致勝": "in",        # opponent's shot landed in
    "落點判斷失誤": "in",        # let it drop, misjudging it as out
}


@dataclass
class TruthRally:
    set: int
    rally: int
    start_sec: float
    end_sec: float          # time of the last hit
    winner: str             # "A" (match winner) or "B"
    winner_side: str        # "upper" (far) or "lower" (near)
    a_side: str             # side player A is on during this rally
    reason: str             # "in", "out", "net" or "other"
    lose_reason_raw: str
    landing_image: list = None
    landing_court: list = None
    score_after: list = None  # [A, B] after the rally


def fetch_labels(dest, match=None):
    """Download match.csv, homography.csv and the set CSVs (for one match, or all)."""
    os.makedirs(dest, exist_ok=True)
    for name in ("match.csv", "homography.csv"):
        urllib.request.urlretrieve(f"{REPO_RAW}/{name}", os.path.join(dest, name))
    with urllib.request.urlopen(REPO_API) as response:
        folders = [item["name"] for item in json.load(response) if item["type"] == "dir"]
    if match:
        folders = [folder for folder in folders if folder == match]
        if not folders:
            raise ValueError(f"No ShuttleSet match named {match}")
    for folder in folders:
        os.makedirs(os.path.join(dest, folder), exist_ok=True)
        url_folder = urllib.request.quote(folder)
        with urllib.request.urlopen(f"{REPO_API}/{url_folder}") as response:
            files = [item["name"] for item in json.load(response) if item["name"].endswith(".csv")]
        for name in files:
            urllib.request.urlretrieve(f"{REPO_RAW}/{url_folder}/{name}", os.path.join(dest, folder, name))
    return folders


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_match_info(labels_dir, match):
    info = next((row for row in _read_csv(os.path.join(labels_dir, "match.csv")) if row["video"] == match), None)
    homography = next((row for row in _read_csv(os.path.join(labels_dir, "homography.csv")) if row["video"] == match), None)
    if info is None or homography is None:
        raise ValueError(f"{match} is not in {labels_dir}/match.csv and homography.csv")
    corners = [
        (float(homography["upleft_x"]), float(homography["upleft_y"])),
        (float(homography["upright_x"]), float(homography["upright_y"])),
        (float(homography["downright_x"]), float(homography["downright_y"])),
        (float(homography["downleft_x"]), float(homography["downleft_y"])),
    ]
    return info, corners


def scale_points(points, video_size):
    sx = video_size[0] / LABEL_SIZE[0]
    sy = video_size[1] / LABEL_SIZE[1]
    return [(x * sx, y * sy) for x, y in points]


def _time_to_sec(text):
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _shot_time(row, fps):
    if row.get("frame_num"):
        return float(row["frame_num"]) / fps
    return float(_time_to_sec(row["time"]))


def load_truth(labels_dir, match, fps=30.0, video_size=LABEL_SIZE):
    """Return (truth rallies, info, corners scaled to the video size)."""
    info, label_corners = load_match_info(labels_dir, match)
    label_mapper = CourtMapper(label_corners)
    net_image_y = float(label_mapper.court_to_image(np.array([3.05, NET_Y]))[1])

    rallies = []
    set_number = 1
    while os.path.exists(os.path.join(labels_dir, match, f"set{set_number}.csv")):
        shots = _read_csv(os.path.join(labels_dir, match, f"set{set_number}.csv"))
        by_rally = {}
        for shot in shots:
            by_rally.setdefault(int(shot["rally"]), []).append(shot)
        for rally_number in sorted(by_rally):
            rally_shots = by_rally[rally_number]
            last = rally_shots[-1]
            if not last.get("getpoint_player"):
                continue
            hitter_side = "lower" if float(last["player_location_y"]) > net_image_y else "upper"
            a_side = hitter_side if last["player"] == "A" else OTHER_SIDE[hitter_side]
            truth = TruthRally(
                set=set_number,
                rally=rally_number,
                start_sec=round(_shot_time(rally_shots[0], fps), 3),
                end_sec=round(_shot_time(last, fps), 3),
                winner=last["getpoint_player"],
                winner_side=a_side if last["getpoint_player"] == "A" else OTHER_SIDE[a_side],
                a_side=a_side,
                reason=LOSE_REASONS.get(last["lose_reason"], "other"),
                lose_reason_raw=last["lose_reason"],
                score_after=[int(last["roundscore_A"]), int(last["roundscore_B"])],
            )
            if last.get("landing_x") and last.get("landing_y"):
                point = (float(last["landing_x"]), float(last["landing_y"]))
                court = label_mapper.image_to_court([point[0], point[1]])
                truth.landing_image = [round(v, 1) for v in scale_points([point], video_size)[0]]
                truth.landing_court = [float(court[0]), float(court[1])]
            rallies.append(truth)
        set_number += 1
    return rallies, info, scale_points(label_corners, video_size)


def write_court_annotation(labels_dir, match, output_dir, video_size):
    """Write court_annotations.txt from the labelled corners so main.py skips court detection."""
    _, label_corners = load_match_info(labels_dir, match)
    corners = [(int(round(x)), int(round(y))) for x, y in scale_points(label_corners, video_size)]
    roi_corners = compute_expanded_roi(corners, (video_size[1], video_size[0]))
    mid_height = CourtMapper(corners).mid_height
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "court_annotations.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"corners={corners}\n")
        file.write(f"roi_corners={roi_corners}\n")
        file.write(f"mid_height={mid_height}\n")
    return path, corners


def match_rallies(calls, truths, before=1.0, after=3.0, expected_delay=0.8):
    """Pair each true rally with the predicted rally ending closest to its last hit."""
    used = set()
    pairs = []
    for truth in truths:
        candidates = [
            index for index, call in enumerate(calls)
            if index not in used and truth.end_sec - before <= call.end_sec <= truth.end_sec + after
        ]
        if not candidates:
            pairs.append((truth, None))
            continue
        index = min(candidates, key=lambda i: abs(calls[i].end_sec - (truth.end_sec + expected_delay)))
        used.add(index)
        pairs.append((truth, calls[index]))
    extras = [call for index, call in enumerate(calls) if index not in used]
    return pairs, extras


def evaluate_shuttleset(calls, truths, match_type="singles"):
    pairs, extras = match_rallies(calls, truths)
    rows = []
    for truth, call in pairs:
        row = {"set": truth.set, "rally": truth.rally, "end_sec": truth.end_sec,
               "true_reason": truth.reason, "true_winner_side": truth.winner_side,
               "predicted_rally": None, "predicted_call": None, "predicted_winner_side": None,
               "confidence": None, "landing_error_m": None, "winner_result": "missed", "call_result": None}
        if truth.reason in ("in", "out") and truth.landing_court is not None:
            row["true_call"], row["true_margin_m"] = call_landing(truth.landing_court, match_type)
        if call is not None:
            row["predicted_rally"] = call.rally
            row["predicted_call"] = call.call
            row["confidence"] = call.confidence
            winner_side, row["reason"] = rally_winner_side(call)
            row["predicted_winner_side"] = winner_side
            row["winner_result"] = (
                "undecided" if winner_side is None else "correct" if winner_side == truth.winner_side else "wrong"
            )
            if truth.reason in ("in", "out"):
                row["call_result"] = "correct" if call.call == truth.reason else "wrong"
            if truth.landing_court is not None and call.landing_court is not None and truth.reason != "net":
                row["landing_error_m"] = round(float(np.linalg.norm(
                    np.array(call.landing_court) - np.array(truth.landing_court))), 2)
        rows.append(row)

    def share(items, key, value="correct"):
        decided = [item for item in items if item[key] is not None and item[key] != "missed"]
        return (sum(item[key] == value for item in decided), len(decided))

    call_rows = [row for row in rows if row["call_result"] is not None]
    errors = [row["landing_error_m"] for row in rows if row["landing_error_m"] is not None]
    by_reason = {}
    for reason in ("in", "out", "net", "other"):
        subset = [row for row in rows if row["true_reason"] == reason]
        if subset:
            by_reason[reason] = {
                "rallies": len(subset),
                "winner_correct": sum(row["winner_result"] == "correct" for row in subset),
            }

    sets = {}
    for truth, _ in pairs:
        sets[truth.set] = truth.score_after
    predicted_sets = {number: [0, 0] for number in sets}
    for (truth, call), row in zip(pairs, rows):
        if row["predicted_winner_side"] is not None:
            winner = 0 if row["predicted_winner_side"] == truth.a_side else 1
            predicted_sets[truth.set][winner] += 1

    return {
        "true_rallies": len(truths),
        "rallies_found": sum(call is not None for _, call in pairs),
        "extra_rallies": len(extras),
        "winner_correct": sum(row["winner_result"] == "correct" for row in rows),
        "winner_wrong": sum(row["winner_result"] == "wrong" for row in rows),
        "winner_missed": sum(row["winner_result"] == "missed" for row in rows),
        "winner_accuracy": round(sum(row["winner_result"] == "correct" for row in rows) / len(rows), 3) if rows else None,
        "line_calls_correct": share(call_rows, "call_result")[0],
        "line_calls_checked": len(call_rows),
        "landing_error_median_m": round(float(np.median(errors)), 2) if errors else None,
        "by_true_reason": by_reason,
        "true_set_scores": {str(k): v for k, v in sets.items()},
        "predicted_set_scores": {str(k): v for k, v in predicted_sets.items()},
        "extras": [{"rally": call.rally, "end_sec": call.end_sec, "call": call.call} for call in extras],
        "rallies": rows,
    }


def run_evaluate(labels_dir, match, output_dir, match_type="singles"):
    metadata, records = load_run(output_dir)
    video_size = (int(metadata["video"]["width"]), int(metadata["video"]["height"]))
    truths, info, corners = load_truth(labels_dir, match, float(metadata["video"]["fps"]), video_size)
    calls = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    report = evaluate_shuttleset(calls, truths, match_type)
    report["match"] = match
    report["players"] = {"A": info["winner"], "B": info["loser"]}

    path = os.path.join(output_dir, "shuttleset_accuracy.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"\nShuttleSet check: {match}")
    print(f"  Rallies found: {report['rallies_found']}/{report['true_rallies']}  "
          f"(extra predicted rallies: {report['extra_rallies']})")
    print(f"  Rally winner correct: {report['winner_correct']}/{report['true_rallies']}"
          f" ({(report['winner_accuracy'] or 0) * 100:.0f}%), wrong {report['winner_wrong']}, "
          f"missed {report['winner_missed']}")
    print(f"  Line calls correct: {report['line_calls_correct']}/{report['line_calls_checked']}  "
          f"median landing error: {report['landing_error_median_m']} m")
    for reason, stats in report["by_true_reason"].items():
        print(f"    rallies ending '{reason}': winner correct {stats['winner_correct']}/{stats['rallies']}")
    for number, true_score in report["true_set_scores"].items():
        predicted = report["predicted_set_scores"][number]
        print(f"  Game {number}: predicted {predicted[0]}-{predicted[1]}, true {true_score[0]}-{true_score[1]} "
              f"({info['winner']} first)")
    print(f"Report: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="ShuttleSet labels: fetch, set up the court, check accuracy")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download ShuttleSet labels from GitHub")
    fetch.add_argument("--dest", default="data/shuttleset")
    fetch.add_argument("--match", default=None, help="Only this match folder (default: all)")

    court = sub.add_parser("court", help="Write court_annotations.txt from the labelled court corners")
    court.add_argument("labels_dir")
    court.add_argument("match")
    court.add_argument("output_dir")
    court.add_argument("--video", required=True, help="The match video, to read its frame size")

    evaluate = sub.add_parser("evaluate", help="Compare an analysis output directory with the labels")
    evaluate.add_argument("labels_dir")
    evaluate.add_argument("match")
    evaluate.add_argument("output_dir")
    evaluate.add_argument("--match-type", choices=["singles", "doubles"], default="singles")

    args = parser.parse_args()
    if args.command == "fetch":
        folders = fetch_labels(args.dest, args.match)
        print(f"Downloaded labels for {len(folders)} matches to {args.dest}")
    elif args.command == "court":
        import cv2
        cap = cv2.VideoCapture(args.video)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if not size[0]:
            raise RuntimeError(f"Unable to read video size: {args.video}")
        path, corners = write_court_annotation(args.labels_dir, args.match, args.output_dir, size)
        print(f"Court corners {corners} written to {path}")
    else:
        run_evaluate(args.labels_dir, args.match, args.output_dir, args.match_type)


if __name__ == "__main__":
    main()
