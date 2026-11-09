"""Line calls from two or more phones filming the same court.

Each phone's calls come from its own view (line_calls.py), and each says how far off its landing point may be
(landing_error_m): a pixel near the far baseline spans more of the court than one near the phone, so a landing
far away is placed less precisely. For each rally the phones' landing points are averaged, each weighted by one
over its error squared, and the rally is called from the average. Phones behind both baselines then each place
the landings near them, and a phone at the side places a landing's distance from a baseline better than one
behind the court.

Rallies are matched across phones by landing time, once the phones' clocks are lined up by the shift that lines
up most of their rallies. Each phone's court is counted from the corner marked first, so another phone's court
may be the same court counted from another corner: turned around or mirrored. The turn that puts the phones'
landings closest together is used.

Combine the calls of analysed videos (outputs/<video>) or capture recordings (recordings/<date-time>):

    python -m badminton_analysis.analysis.multi_camera outputs/<phone 1> outputs/<phone 2>

The combined calls use the first one's frame numbers and court, and are written next to its own calls.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ..court.reference import BADMINTON_COURT_LENGTH, BADMINTON_COURT_WIDTH
from .line_calls import NET_Y, LineCallConfig, RallyCall, call_landing, compute_line_calls, load_run, summarize

COMBINED_CALLS = "combined_line_calls.json"
# Counting x and/or y from the other side: the same court, marked from another corner.
TURNS = ((False, False), (True, False), (False, True), (True, True))
TURN_NAMES = {(False, False): "same way", (True, False): "mirrored across", (False, True): "mirrored lengthways",
              (True, True): "turned around"}
DISAGREE = 3.0        # phones whose landing points are this many sigmas apart disagree
MATCH_WINDOW = 0.5    # seconds between two phones' landing times of the same rally, at most
GUESS_WINDOW = 3.0    # seconds the recordings' start times may be off from the phones' real offset
LEVELS = ["low", "medium", "high"]


@dataclass
class CombinedCall(RallyCall):
    cameras: list = field(default_factory=list)   # each phone's own call of the rally


def turn(point, flips):
    """A court point counted from the other side across (flips[0]) and/or along the court (flips[1])."""
    x, y = point
    return [BADMINTON_COURT_WIDTH - x if flips[0] else x, BADMINTON_COURT_LENGTH - y if flips[1] else y]


def landing_time(call, fps):
    """Seconds into the recording at which the rally's shuttle landed, or was last seen in flight."""
    return (call.landing_frame if call.landing_frame is not None else call.end_frame) / fps


def pair_up(times_a, times_b, offset, window=MATCH_WINDOW):
    """Index pairs (i, j) of the same moments once offset is added to times_b: at most window apart, each time in
    one pair at most, closest first."""
    close = sorted((abs(a - b - offset), i, j) for i, a in enumerate(times_a) for j, b in enumerate(times_b)
                   if abs(a - b - offset) <= window)
    used_a, used_b, pairs = set(), set(), []
    for _, i, j in close:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def find_offset(times_a, times_b, window=MATCH_WINDOW, guess=None, search=GUESS_WINDOW):
    """Seconds to add to times_b to line them up with times_a: the shift that lines up the most rallies, taken as
    the median of their differences. With a guess, as from the recordings' start times, only shifts within
    search seconds of it are tried. Returns (offset, pairs as from pair_up)."""
    differences = sorted(a - b for a in times_a for b in times_b if guess is None or abs(a - b - guess) <= search)
    if not differences:
        return (guess or 0.0), []
    count, first, start = 0, 0, 0
    for end, value in enumerate(differences):
        while value - differences[start] > window:
            start += 1
        if end - start + 1 > count:
            count, first = end - start + 1, start
    offset = float(np.median(differences[first:first + count]))
    return offset, pair_up(times_a, times_b, offset, window)


def find_turn(points_a, points_b):
    """The turn of court b that puts its landing points closest to court a's, from points of the same landings.
    Distances are capped so that one rally matched wrongly cannot decide it."""
    def cost(flips):
        return sum(min(3.0, float(np.hypot(*np.subtract(a, turn(b, flips))))) for a, b in zip(points_a, points_b))
    return min(TURNS, key=cost)


def _placed(member):
    return member["point"] is not None and bool(member["call"].landing_error_m)


def _group_time(group):
    return float(np.mean([member["time"] for member in group]))


def _group_point(group):
    """The landing point of the rally so far: the one placed most precisely, or any."""
    placed = [member for member in group if _placed(member)]
    if placed:
        return min(placed, key=lambda member: max(member["call"].landing_error_m))["point"]
    return next((member["point"] for member in group if member["point"] is not None), None)


def combine(cameras, config=None, window=MATCH_WINDOW, search=GUESS_WINDOW):
    """One call per rally from several phones' calls.

    cameras: dicts with "name", "calls" (RallyCall, in the phone's own frames and court) and "fps", and optionally
    "started", the recording's start in seconds on a clock all phones share (capture's). The first is the
    reference: the combined calls use its frame numbers and court. Returns (calls, alignment): alignment gives
    each phone's offset (seconds added to its times to get the reference's), turn and rallies matched with the
    phones before it.
    """
    config = config or LineCallConfig()
    reference = cameras[0]
    groups, alignment = [], []
    for index, camera in enumerate(cameras):
        calls, fps = camera["calls"], float(camera["fps"])
        times = [landing_time(call, fps) for call in calls]
        offset, pairs, flips = 0.0, [], TURNS[0]
        if index:
            guess = None
            if camera.get("started") is not None and reference.get("started") is not None:
                guess = camera["started"] - reference["started"]
            offset, pairs = find_offset([_group_time(group) for group in groups], times, window, guess, search)
            known = [(_group_point(groups[i]), calls[j].landing_court) for i, j in pairs
                     if calls[j].landing_court and _group_point(groups[i]) is not None]
            if known:
                flips = find_turn(*zip(*known))
        matched = {j: i for i, j in pairs}
        for j, call in enumerate(calls):
            member = {"camera": camera["name"], "reference": index == 0, "call": call, "fps": fps, "offset": offset,
                      "time": times[j] + offset,
                      "point": turn(call.landing_court, flips) if call.landing_court else None}
            if j in matched:
                groups[matched[j]].append(member)
            else:
                groups.append([member])
        alignment.append({"camera": camera["name"], "offset_s": round(offset, 3), "turn": TURN_NAMES[flips],
                          "matched": len(pairs)})
    groups.sort(key=_group_time)
    calls = [combine_rally(number, group, float(reference["fps"]), config) for number, group in enumerate(groups, 1)]
    return calls, alignment


def _at_most(confidence, cap):
    return min(confidence, cap, key=LEVELS.index)


def combine_rally(number, members, fps, config):
    """The call of one rally from each phone's call of it, in the reference phone's frames and court."""
    placed = [member for member in members if _placed(member)]
    base = next((member for member in members if member["reference"]), None)
    if base is None:
        base = min(placed, key=lambda member: max(member["call"].landing_error_m)) if placed else members[0]
    own = base["call"]

    def frame(value):   # a frame of the base phone's recording, as a frame of the reference's
        return None if value is None else int(round((value / base["fps"] + base["offset"]) * fps))

    values = {item.name: getattr(own, item.name) for item in fields(RallyCall)}
    values.update(rally=number, flags=[], landing_frame=frame(own.landing_frame),
                  landing_detect_frame=frame(own.landing_detect_frame), start_frame=frame(own.start_frame),
                  end_frame=frame(own.end_frame), start_sec=round(own.start_sec + base["offset"], 3),
                  end_sec=round(own.end_sec + base["offset"], 3), landing_court=base["point"])
    if not base["reference"]:
        values["landing_image"] = None   # a point in another phone's picture
    rally = CombinedCall(**values, cameras=[{
        "camera": member["camera"], "rally": member["call"].rally, "call": member["call"].call,
        "margin_m": member["call"].margin_m, "landing_court": member["point"],
        "landing_error_m": member["call"].landing_error_m, "landing_method": member["call"].landing_method,
        "confidence": member["call"].confidence,
        "landing_sec": round(landing_time(member["call"], member["fps"]), 3)} for member in members])
    for member in members:
        rally.flags += [f"{member['camera']}: {flag}" for flag in member["call"].flags]

    if len(placed) < 2:   # nothing to combine: the call of the one phone that placed the landing, or the base's
        only = placed[0] if placed else base
        for name in ("call", "margin_m", "landing_error_m", "landing_method", "confidence"):
            setattr(rally, name, getattr(only["call"], name))
        rally.landing_court = only["point"]
        if len(members) > 1:
            rally.flags.append(f"only {only['camera']} placed the landing" if placed else "no phone placed the landing")
    else:
        errors = np.maximum(np.array([member["call"].landing_error_m for member in placed], float), 1e-3)
        weights = 1.0 / errors ** 2
        points = np.array([member["point"] for member in placed], float)
        point = (weights * points).sum(axis=0) / weights.sum(axis=0)
        rally.landing_court = [float(point[0]), float(point[1])]
        rally.landing_error_m = [round(float(value), 3) for value in 1.0 / np.sqrt(weights.sum(axis=0))]
        rally.landing_method = "combined"
        rally.call, rally.margin_m = call_landing(point, config.match_type)
        rally.confidence = "medium" if abs(rally.margin_m) < config.close_call_margin else "high"
        if max(float(np.sqrt(np.sum((p - point) ** 2 * w))) for p, w in zip(points, weights)) > DISAGREE:
            apart = max(float(np.hypot(*(p - q))) for p in points for q in points)
            rally.flags.append(f"phones disagree: their landings are {apart:.2f} m apart")
            rally.confidence = "low"
        if len({member["call"].call for member in placed}) > 1:
            rally.flags.append("phones split: " + ", ".join(
                f"{member['camera']} {member['call'].call.upper()}" for member in placed))
            rally.confidence = _at_most(rally.confidence, "medium")
        held = [member for member in placed if member["call"].landing_method == "held"]
        if held:   # a hand beside the shuttle in one picture may be nowhere near it in another
            rally.confidence = "low" if len(held) == len(placed) else _at_most(rally.confidence, "medium")
        if rally.margin_m < -config.max_out:
            rally.flags.append(f"landing {-rally.margin_m:.1f} m outside the court: "
                               "the shuttle was probably still in the air")
            rally.confidence = "low"
        elif abs(rally.margin_m) < config.close_call_margin:
            rally.flags.append(f"close call: {abs(rally.margin_m) * 100:.0f} cm from the line")
        net_distance = abs(point[1] - NET_Y)
        if net_distance < config.net_margin:
            rally.flags.append(f"landed {net_distance * 100:.0f} cm from the net: which side it fell on is uncertain")
    if rally.landing_court is not None:
        rally.landing_side = "upper" if rally.landing_court[1] < NET_Y else "lower"
    return rally


def to_call(data):
    """A RallyCall from its JSON, ignoring fields it does not have."""
    names = {item.name for item in fields(RallyCall)}
    return RallyCall(**{key: value for key, value in data.items() if key in names})


def load_camera(folder, match_type="singles"):
    """A phone's calls from an analysis output directory (line_calls.json, or computed from its detections) or a
    capture recording (live_line_calls.json), with its frame rate and, for a recording, its start time."""
    camera = {"name": os.path.basename(os.path.normpath(folder)), "folder": folder, "started": None}
    capture_path, metadata_path = os.path.join(folder, "capture.json"), os.path.join(folder, "metadata.json")
    if os.path.exists(capture_path):
        with open(capture_path, "r", encoding="utf-8") as file:
            info = json.load(file)
        camera.update(fps=info["fps"], started=info.get("started"))
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            camera["fps"] = json.load(file)["video"]["fps"]
    report_path, live_path = os.path.join(folder, "line_calls.json"), os.path.join(folder, "live_line_calls.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as file:
            report = json.load(file)
        camera["calls"] = [to_call(data) for data in report["rallies"]]
        camera["match_type"] = report.get("config", {}).get("match_type")
    elif os.path.exists(live_path):
        with open(live_path, "r", encoding="utf-8") as file:
            camera["calls"] = [to_call(data) for data in json.load(file)]
    else:
        metadata, records = load_run(folder)
        camera["calls"] = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    if "fps" not in camera:
        raise ValueError(f"{folder}: no metadata.json or capture.json to take the frame rate from")
    return camera


def write_combined(path, calls, alignment, cameras, config):
    report = {
        "config": asdict(config),
        "summary": summarize(calls),
        "cameras": [{**entry, "folder": camera.get("folder")} for entry, camera in zip(alignment, cameras)],
        "rallies": [asdict(rally) for rally in calls],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report


def print_combined(calls, alignment, fps):
    reference = alignment[0]["camera"]
    for entry in alignment[1:]:
        print(f"{entry['camera']}: {entry['offset_s']:+.2f} s against {reference}, court {entry['turn']}, "
              f"{entry['matched']} rallies matched")
    print(f"\nCombined line calls: {len(calls)} rallies")
    for rally in calls:
        margin = f"{rally.margin_m * 100:+.0f} cm" if rally.margin_m is not None else "-"
        error = f" ±{max(rally.landing_error_m) * 100:.0f}" if rally.landing_error_m else ""
        landed = rally.landing_frame / fps if rally.landing_frame is not None else rally.end_sec
        print(f"  #{rally.rally:<3} {landed:7.2f}s  {rally.call.upper():<4} {margin:>8}{error:<5} [{rally.confidence}]")
        for seen in rally.cameras:
            own = f"{seen['margin_m'] * 100:+.0f} cm" if seen["margin_m"] is not None else "-"
            spread = f" ±{max(seen['landing_error_m']) * 100:.0f}" if seen["landing_error_m"] else ""
            print(f"        {seen['camera']}: {seen['call'].upper()} {own}{spread} ({seen['landing_method']})")
        for flag in rally.flags:
            print(f"        - {flag}")


def main():
    parser = argparse.ArgumentParser(description="Combine the line calls of phones filming the same court")
    parser.add_argument("folders", nargs="+", help="Analysis output directories or capture recordings, one per phone")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default=None,
                        help="Default: as the first folder's line calls were made, else singles")
    args = parser.parse_args()
    if len(args.folders) < 2:
        parser.error("give at least two folders, one per phone")
    cameras = [load_camera(folder, args.match_type or "singles") for folder in args.folders]
    names = [camera["name"] for camera in cameras]
    for number, camera in enumerate(cameras, 1):
        if names.count(camera["name"]) > 1:
            camera["name"] = f"phone {number}"
    config = LineCallConfig(match_type=args.match_type or cameras[0].get("match_type") or "singles")
    calls, alignment = combine(cameras, config)
    path = os.path.join(args.folders[0], COMBINED_CALLS)
    write_combined(path, calls, alignment, cameras, config)
    print_combined(calls, alignment, float(cameras[0]["fps"]))
    print(f"Combined line calls: {path}")


if __name__ == "__main__":
    main()
