"""Shuttlecock line calls from an existing analysis run.

Reads ``metadata.json`` and ``detections.jsonl`` written by the main pipeline
(and ``tracknet.csv``, when TrackNet has been run on the video: its shuttle
positions are used instead of the detector's), splits the video into rallies
from shuttle motion (fixed-camera friendly: no camera cuts needed), finds where
the last shot of each rally landed and calls it in or out against the singles
or doubles boundary.

Run it again on an existing output directory without re-processing the video:

    python -m badminton_analysis.analysis.line_calls outputs/<video_name> --match-type singles
"""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from ..court.mapper import CourtMapper
from ..court.reference import (
    BADMINTON_COURT_LENGTH,
    BADMINTON_COURT_WIDTH,
    BADMINTON_SINGLES_MARGIN,
)


NET_Y = BADMINTON_COURT_LENGTH / 2.0
SIDE_LABELS = {"upper": "far", "lower": "near"}
TRACKNET_CSV = "tracknet.csv"   # shuttle positions from detection/tracknet.py, next to detections.jsonl


@dataclass
class LineCallConfig:
    match_type: str = "singles"
    # Speeds and distances in screen heights, so thresholds hold across resolutions.
    move_speed: float = 0.25        # faster than this (per second) = shuttle in flight
    still_radius: float = 0.015     # settled shuttle stays within this radius
    max_link_gap: float = 0.25      # seconds; larger detection gaps are not used for speed
    merge_gap: float = 1.0          # seconds; flight pieces closer than this form one rally
    top_exit_margin: float = 0.05   # flight last seen this close to the top edge = high shot above the frame
    offscreen_gap: float = 3.0      # seconds a high shot may stay above the frame without ending the rally
    min_rally_duration: float = 0.5
    min_rally_points: int = 6
    settle_window: float = 1.5      # seconds after the last flight sample to look for the landing
    settle_points: int = 3
    # The shuttle at rest is followed until this many seconds after the last flight sample, not until someone
    # picks it up, so the call can be made while the match goes on. On four ShuttleSet matches this moved one
    # landing, by 2 cm.
    rest_window: float = 2.0
    # Error of a landing point, shown as a ring in the review. position_error (screen heights) covers
    # detection, timing and bounce together: at 0.02, 70% of ShuttleSet's labelled landings fall within
    # one sigma. Retune it on recordings from your own camera.
    position_error: float = 0.02
    bounce_error: float = 0.3       # meters a shuttle seen only after it bounced may have moved
    # A shuttle that comes down fast often slides or bounces on for a meter or more, too fast to count as
    # resting. Once it is on the floor its place on the court map is real, so it moves a few meters per
    # second there at most, while a shuttle still dropping through the air sweeps across the map.
    slide_window: float = 0.7       # seconds before the shuttle comes to rest to look for the touch-down
    slide_max_speed: float = 12.0   # m/s on the court map; faster is still in the air
    slide_drop: float = 3.0         # the flight must cross the court map this many times faster than the slide
    max_out: float = 3.0            # meters outside the court; a "landing" further out was still in the air
    hand_radius: float = 0.06       # settled shuttle this close to a hand = held, not landed
    close_call_margin: float = 0.15  # meters from a line that counts as a close call
    net_margin: float = 0.3          # meters from the net where the landing side is uncertain
    static_cell: float = 0.01        # grid cell size for finding fixed false detections
    static_share: float = 0.02       # a cell holding more than this share of all detections is suspect
    static_off_court: float = 2.0    # meters outside the court a suspect cell must map to before it is dropped
    # Between points the shuttle is tapped or thrown back to the server: short, low, and the
    # players hardly move. A real rally covers more of the screen height and both players run.
    min_rally_height: float = 0.35   # shuttle's vertical travel in screen heights
    min_player_move: float = 3.0     # meters walked by the player who moved least


@dataclass
class RallyCall:
    rally: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    landing_frame: int = None
    landing_detect_frame: int = None
    landing_image: list = None
    landing_court: list = None
    landing_side: str = None
    call: str = "unknown"
    margin_m: float = None
    landing_error_m: list = None    # 1-sigma error of the landing point along court x and y, meters
    landing_method: str = None
    confidence: str = "low"
    flags: list = field(default_factory=list)


def court_bounds(match_type):
    """Return (x_min, x_max, y_min, y_max) in meters for rally play."""
    if match_type == "doubles":
        return 0.0, BADMINTON_COURT_WIDTH, 0.0, BADMINTON_COURT_LENGTH
    if match_type == "singles":
        return (
            BADMINTON_SINGLES_MARGIN,
            BADMINTON_COURT_WIDTH - BADMINTON_SINGLES_MARGIN,
            0.0,
            BADMINTON_COURT_LENGTH,
        )
    raise ValueError(f"Unknown match type: {match_type}")


def call_landing(court_point, match_type):
    """Call a landing point in or out.

    Returns (call, margin_m). The margin is the signed distance to the nearest
    boundary line: positive inside the court, negative outside. A shuttle on
    the line is in, so zero counts as in.
    """
    x_min, x_max, y_min, y_max = court_bounds(match_type)
    x, y = float(court_point[0]), float(court_point[1])
    margin = min(x - x_min, x_max - x, y - y_min, y_max - y)
    return ("in" if margin >= 0 else "out"), round(margin, 3)


def load_tracknet_csv(path):
    """{frame: [x, y]} from tracknet.csv. Frames where TrackNet did not see the shuttle are left out."""
    with open(path, "r", encoding="utf-8", newline="") as file:
        return {int(row["frame"]): [float(row["x"]), float(row["y"])] for row in csv.DictReader(file)}


def load_run(output_dir, use_tracknet=True):
    with open(os.path.join(output_dir, "metadata.json"), "r", encoding="utf-8") as file:
        metadata = json.load(file)
    # Prefer the file next to metadata.json: the stored path goes stale if the folder is moved.
    detections_path = os.path.join(output_dir, "detections.jsonl")
    if not os.path.exists(detections_path):
        detections_path = metadata.get("outputs", {}).get("detections") or detections_path
    records = []
    with open(detections_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # TrackNet's shuttle replaces the detector's in every frame. It follows the shuttle down to the floor
    # more often: on four ShuttleSet matches it got 121 of 146 line calls right against 106 of 142, and
    # filling its gaps with the detector's positions made the calls worse.
    tracknet_path = os.path.join(output_dir, TRACKNET_CSV)
    if use_tracknet and os.path.exists(tracknet_path):
        points = load_tracknet_csv(tracknet_path)
        for record in records:
            record["shuttlecock"] = {"image": points.get(record["frame"])}
    return metadata, records


def _shuttle_samples(records, fps):
    samples = []
    for record in records:
        point = (record.get("shuttlecock") or {}).get("image")
        if not point:
            continue
        time_sec = record.get("time_sec")
        if time_sec is None:
            time_sec = record["frame"] / fps
        samples.append({
            "frame": int(record["frame"]),
            "detect_frame": int(record.get("detect_frame", record["frame"])),
            "t": float(time_sec),
            "p": np.array(point[:2], dtype=float),
        })
    return samples


def _hand_points(record):
    points = []
    for player in (record.get("players") or {}).values():
        if not player:
            continue
        for hand in (player.get("hands") or {}).values():
            if hand:
                points.append(np.array(hand[:2], dtype=float))
    return points


def drop_static_spots(samples, frame_height, mapper, config):
    """Remove detections stuck on one off-court spot, such as a light or sign the detector mistakes for the shuttle.

    A shuttle resting on the floor between rallies also piles up detections in one place, but it lies on or
    near the court, so only spots that map well outside the court are dropped.
    """
    if not samples:
        return samples, []
    cell = max(1.0, config.static_cell * frame_height)
    counts = {}
    for sample in samples:
        key = _cell(sample, cell)
        counts[key] = counts.get(key, 0) + 1
    static = static_cells(counts, len(samples), cell, mapper, config)
    if not static:
        return samples, []
    kept = [s for s in samples if _cell(s, cell) not in static]
    spots = [[round((key[0] + 0.5) * cell, 1), round((key[1] + 0.5) * cell, 1)] for key in sorted(static)]
    return kept, spots


def _cell(sample, cell):
    return int(sample["p"][0] // cell), int(sample["p"][1] // cell)


def static_cells(counts, total, cell, mapper, config):
    """Cells holding more than static_share of `total` detections that lie well outside the court."""
    static = set()
    for key, count in counts.items():
        if count <= config.static_share * total:
            continue
        centre = [(key[0] + 0.5) * cell, (key[1] + 0.5) * cell]
        _, margin = call_landing(mapper.image_to_court(centre), config.match_type)
        if margin < -config.static_off_court:
            static.add(key)
    return static


def rally_gap(sample, frame_height, config):
    """How long the flight may pause after this sample and still be the same rally. A clear or lob often
    leaves the top of the frame; that is not the end of the rally."""
    return config.offscreen_gap if sample["p"][1] < config.top_exit_margin * frame_height else config.merge_gap


def flight_groups(samples, frame_height, config):
    """Indices of the samples where the shuttle flies, grouped into stretches without a pause longer than
    rally_gap."""
    move_px = config.move_speed * frame_height
    moving = []
    for i in range(1, len(samples)):
        dt = samples[i]["t"] - samples[i - 1]["t"]
        if dt <= 0 or dt > config.max_link_gap:
            continue
        speed = np.linalg.norm(samples[i]["p"] - samples[i - 1]["p"]) / dt
        if speed > move_px:
            moving.append(i)

    groups = []
    for i in moving:
        if groups and samples[i]["t"] - samples[groups[-1][-1]]["t"] <= rally_gap(samples[groups[-1][-1]],
                                                                                    frame_height, config):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def find_rallies(samples, frame_height, config):
    """Group shuttle samples into rallies. Returns (first_index, last_index) of flight samples."""
    rallies = []
    for group in flight_groups(samples, frame_height, config):
        first = max(0, group[0] - 1)
        last = group[-1]
        duration = samples[last]["t"] - samples[first]["t"]
        if duration >= config.min_rally_duration and len(group) >= config.min_rally_points:
            rallies.append((first, last))
    return rallies


def _player_path(records, start_frame, end_frame, side):
    points = [
        record["players"][side]["court"]
        for record in records
        if start_frame <= record["frame"] <= end_frame
        and (record.get("players") or {}).get(side) and record["players"][side].get("court")
    ]
    if len(points) < 2:
        return None
    return float(np.sum(np.linalg.norm(np.diff(np.array(points, dtype=float), axis=0), axis=1)))


def is_play(samples, first, last, records, frame_height, config):
    """Tell a rally from the shuttle being knocked about between points.

    Only a candidate that stays low and leaves both players nearly still is dropped:
    a two-shot rally (serve, then a winner) moves the players little too, but the
    serve sends the shuttle high.
    """
    heights = [samples[i]["p"][1] for i in range(first, last + 1)]
    if (max(heights) - min(heights)) / frame_height >= config.min_rally_height:
        return True
    paths = [_player_path(records, samples[first]["frame"], samples[last]["frame"], side)
             for side in ("upper", "lower")]
    paths = [path for path in paths if path is not None]
    # With a player missing there is nothing to go on, so keep the rally.
    return len(paths) < 2 or min(paths) >= config.min_player_move


def _find_settle(samples, start, stop, end_t, frame_height, config):
    """Find the first run of samples where the shuttle sits still after flight."""
    radius = config.still_radius * frame_height
    for j in range(start, stop):
        if samples[j]["t"] - end_t > config.settle_window:
            break
        run = [j]
        for k in range(j + 1, stop):
            if samples[k]["t"] - end_t > config.rest_window:
                break
            if samples[k]["t"] - samples[run[-1]]["t"] > config.max_link_gap:
                break
            if np.linalg.norm(samples[k]["p"] - samples[j]["p"]) > radius:
                break
            run.append(k)
        if len(run) >= config.settle_points:
            return run
    return None


def _linked(samples, a, b, config):
    return samples[b]["t"] - samples[a]["t"] <= config.max_link_gap


def _velocity(samples, a, b):
    return (samples[b]["p"] - samples[a]["p"]) / (samples[b]["t"] - samples[a]["t"])


def _fit_velocity(samples, indices):
    """Least-squares image velocity (px/s) through the samples, with their mean time and position."""
    t = np.array([samples[i]["t"] for i in indices])
    p = np.array([samples[i]["p"] for i in indices])
    t_mean, p_mean = t.mean(), p.mean(axis=0)
    velocity = ((t - t_mean)[:, None] * (p - p_mean)).sum(axis=0) / max(float(((t - t_mean) ** 2).sum()), 1e-9)
    return velocity, t_mean, p_mean


def _run(samples, start, stop, config, count=3):
    """Indices of up to count linked samples from start on."""
    indices = [start]
    while len(indices) < count and indices[-1] + 1 < stop and _linked(samples, indices[-1], indices[-1] + 1, config):
        indices.append(indices[-1] + 1)
    return indices


def _find_slide(samples, first, stop, settle, fps, frame_height, mapper, config):
    """Where a shuttle that slid or bounced on after landing touched down, as _find_touchdown
    returns it, or None if it did not slide.

    Looks back from where it came to rest for the sample k where it hit the floor: its image
    motion carries on the same way after k, and on the court map it slows from flight speed to
    sliding speed. Decelerating in the air is smooth and keeps it fast on the map, and a hit sends
    it back. Of several such samples, take the one where it slowed down most in the image.
    """
    move_px = config.move_speed * frame_height
    rest, rest_end = settle[0], settle[-1]
    court = {}

    def map_speed(a, b):
        for i in (a, b):
            if i not in court:
                court[i] = np.array(mapper.image_to_court([float(samples[i]["p"][0]), float(samples[i]["p"][1])])[:2])
        return float(np.linalg.norm(court[b] - court[a])) / (samples[b]["t"] - samples[a]["t"])

    best = None
    for k in range(rest_end - 1, first + 1, -1):
        if samples[rest]["t"] - samples[k]["t"] > config.slide_window:
            break
        if not all(_linked(samples, i, i + 1, config) for i in range(k - 2, rest_end)):
            continue
        incoming = _fit_velocity(samples, [k - 2, k - 1, k])
        speed_in = float(np.linalg.norm(incoming[0]))
        if not move_px <= speed_in <= 5 * frame_height:   # faster is a jump to a false detection
            continue
        runs = [_run(samples, j, stop, config) for j in range(k + 1, rest_end + 1)]
        outgoing = [_fit_velocity(samples, run) for run in runs if len(run) >= 2]
        if not outgoing or any(float(incoming[0] @ v) < 0 for v, _, _ in outgoing if np.linalg.norm(v) > 0.5 * move_px):
            continue
        slide_speed = max((map_speed(j, j + 2) for j in range(k + 1, rest_end - 1)), default=0.0)
        if slide_speed > config.slide_max_speed or map_speed(k - 2, k) < config.slide_drop * slide_speed:
            continue
        slowdown = speed_in - max(float(np.linalg.norm(v)) for v, _, _ in outgoing)
        if best is None or slowdown > best[0]:
            best = (slowdown, k, incoming, outgoing[0])
    if best is None:
        return None
    _, k, (v_in, t_in, p_in), (v_out, t_out, p_out) = best
    # The touch-down is where the lines of motion before and after k come closest, between k and the next sample.
    change = v_in - v_out
    apart = p_in - v_in * t_in - p_out + v_out * t_out
    t = float(np.clip(-(apart @ change) / max(float(change @ change), 1e-9), samples[k]["t"], samples[k + 1]["t"]))
    point = (p_in + v_in * (t - t_in) + p_out + v_out * (t - t_out)) / 2
    offset = int(round((t - samples[k]["t"]) * fps))
    return point, samples[k]["frame"] + offset, samples[k]["detect_frame"] + offset, v_in


def _find_touchdown(samples, first, last, stop, settle, fps, frame_height, mapper, config):
    """Where and when the shuttle first touched the floor: (image point, frame, detect frame,
    incoming image velocity in px/s, or None when the point is the first still sample).

    A shuttle often bounces or skids on for 0.3 m or more after touching down, so where it
    comes to rest is too deep. If it slid on fast enough to look like flight, find where the
    slide began. Otherwise the touch-down lies between the last fast sample and the next one:
    find the moment when the incoming flight, continued at its last speed, meets the slow
    motion that follows. Without samples on both sides, or if the shuttle was lost on its way
    down, use the first still sample.
    """
    slide = _find_slide(samples, first, stop, settle, fps, frame_height, mapper, config)
    if slide:
        return slide
    after = last + 1
    if not (first < last and after < stop and _linked(samples, last - 1, last, config)
            and all(_linked(samples, i, i + 1, config) for i in range(last, max(after, settle[0])))):
        sample = samples[settle[0]]
        return sample["p"], sample["frame"], sample["detect_frame"], None
    v_in = _velocity(samples, last - 1, last)
    v_out = np.zeros(2)
    if after + 1 < stop and _linked(samples, after, after + 1, config):
        v_out = _velocity(samples, after, after + 1)
    gap = samples[after]["t"] - samples[last]["t"]
    # Next sample = last + v_in * s + v_out * (gap - s); solve for s, the time to touch-down.
    change = v_in - v_out
    s = (samples[after]["p"] - samples[last]["p"] - v_out * gap) @ change / max(float(change @ change), 1e-9)
    s = float(np.clip(s, 0.0, gap))
    offset = int(round(s * fps))
    return samples[last]["p"] + v_in * s, samples[last]["frame"] + offset, samples[last]["detect_frame"] + offset, v_in


def _court_jacobian(matrix, point):
    """Meters on the court per pixel at an image point, from the image-to-court homography."""
    w = matrix @ np.array([point[0], point[1], 1.0])
    x, y = w[0] / w[2], w[1] / w[2]
    return np.array([[matrix[0, 0] - x * matrix[2, 0], matrix[0, 1] - x * matrix[2, 1]],
                     [matrix[1, 0] - y * matrix[2, 0], matrix[1, 1] - y * matrix[2, 1]]]) / w[2]


def landing_error(matrix, point, velocity, fps, frame_height, config):
    """1-sigma error of a landing point along court x and y, in meters.

    Adds up the position error and, for a touch-down placed between two frames, how far the
    shuttle travels in half a frame. Without a touch-down estimate the point is where the shuttle
    was first seen still, so it may have bounced on: add bounce_error.
    """
    jacobian = _court_jacobian(matrix, point)
    variance = (config.position_error * frame_height) ** 2 * np.sum(jacobian ** 2, axis=1)
    if velocity is None:
        variance += config.bounce_error ** 2
    else:
        variance += (jacobian @ velocity / (2 * fps)) ** 2
    return [round(float(value), 3) for value in np.sqrt(variance)]


def compute_line_calls(metadata, records, config=None):
    config = config or LineCallConfig()
    fps = float(metadata["video"]["fps"])
    frame_height = float(metadata["video"]["height"])
    mapper = CourtMapper([tuple(corner) for corner in metadata["court"]["corners"]])
    records_by_frame = {int(record["frame"]): record for record in records}

    samples, _ = drop_static_spots(_shuttle_samples(records, fps), frame_height, mapper, config)
    rallies = [
        (first, last) for first, last in find_rallies(samples, frame_height, config)
        if is_play(samples, first, last, records, frame_height, config)
    ]

    calls = []
    for index, (first, last) in enumerate(rallies):
        next_start = rallies[index + 1][0] if index + 1 < len(rallies) else len(samples)
        calls.append(call_rally(index + 1, samples, first, last, next_start, records_by_frame, fps, frame_height,
                                mapper, config))
    return calls


def call_rally(number, samples, first, last, stop, records_by_frame, fps, frame_height, mapper, config):
    """The call for the rally whose flight runs from samples[first] to samples[last]. The next rally's
    samples start at stop."""
    rally = RallyCall(
        rally=number,
        start_frame=samples[first]["frame"],
        end_frame=samples[last]["frame"],
        start_sec=round(samples[first]["t"], 3),
        end_sec=round(samples[last]["t"], 3),
    )

    settle = _find_settle(samples, last, stop, samples[last]["t"], frame_height, config)
    if settle:
        point, landing_frame, landing_detect_frame, velocity = _find_touchdown(
            samples, first, last, stop, settle, fps, frame_height, mapper, config)
        rally.landing_method = "settled"
        hand_px = config.hand_radius * frame_height
        for k in settle:
            hands = _hand_points(records_by_frame.get(samples[k]["frame"], {}))
            if any(np.linalg.norm(samples[k]["p"] - hand) < hand_px for hand in hands):
                rally.landing_method = "held"
                rally.flags.append("shuttle still near a player's hand: caught or picked up, not a floor landing")
                break
    else:
        point = samples[last]["p"]
        landing_frame, landing_detect_frame = samples[last]["frame"], samples[last]["detect_frame"]
        rally.landing_method = "last_seen"
        rally.flags.append("shuttle lost before it settled: landing taken from last flight position")

    court_point = mapper.image_to_court([float(point[0]), float(point[1])])
    rally.landing_frame = landing_frame
    rally.landing_detect_frame = landing_detect_frame
    rally.landing_image = [round(float(point[0]), 1), round(float(point[1]), 1)]
    rally.landing_court = [float(court_point[0]), float(court_point[1])]
    rally.landing_side = "upper" if court_point[1] < NET_Y else "lower"
    rally.call, rally.margin_m = call_landing(court_point, config.match_type)
    # No shot lands that far out. The point is the shuttle high in the air, which the court
    # mapping puts on the floor far behind it.
    in_air = rally.margin_m < -config.max_out
    if in_air:
        rally.flags.append(f"landing {-rally.margin_m:.1f} m outside the court: "
                           "the shuttle was probably still in the air")
    elif rally.landing_method != "last_seen":  # a lost shuttle's position is a guess, not a measurement
        rally.landing_error_m = landing_error(mapper.matrix, point, velocity, fps, frame_height, config)

    if abs(rally.margin_m) < config.close_call_margin:
        rally.flags.append(f"close call: {abs(rally.margin_m) * 100:.0f} cm from the line")
    net_distance = abs(rally.landing_court[1] - NET_Y)
    if net_distance < config.net_margin:
        rally.flags.append(f"landed {net_distance * 100:.0f} cm from the net: which side it fell on is uncertain")
    if rally.landing_method == "settled" and not in_air:
        rally.confidence = "medium" if abs(rally.margin_m) < config.close_call_margin else "high"
    return rally


class LiveLineCaller:
    """compute_line_calls for records that arrive while the match is played: each rally is called as soon
    as nothing that comes later can change its call.

    A call depends on the shuttle from the start of the rally until it has rested for rest_window, plus
    the two samples after that which _find_slide may fit, so it is final rest_window + 2 * max_link_gap
    after the last flight sample. By then the rally must also be over, which takes merge_gap, or
    offscreen_gap after a shot that left the top of the frame. Flight that started since and goes on may
    still turn out to be the next rally, which would end the search for where this one came to rest, so
    the call must then come out the same either way. Records before the end of the last called rally are
    let go.

    Detections stuck on one off-court spot are dropped as in compute_line_calls, judged from the counts of
    all detections so far, taken as at least min_total: otherwise, early in a match, a shuttle resting
    outside the court would look stuck before its own rally had been called.
    """

    def __init__(self, metadata, config=None, check_every=0.2, min_total=10000):
        self.config = config or LineCallConfig()
        self.fps = float(metadata["video"]["fps"])
        self.frame_height = float(metadata["video"]["height"])
        self.mapper = CourtMapper([tuple(corner) for corner in metadata["court"]["corners"]])
        self.cell = max(1.0, self.config.static_cell * self.frame_height)
        self.counts, self.heavy, self.total, self.min_total = {}, set(), 0, min_total
        self.check_every = check_every
        self.next_check = -np.inf
        self.records = []
        self.called = 0

    def update(self, records, now):
        """Take the records finished since the last update. now is the time (seconds, as in time_sec) up to
        which every record has been given. Returns the calls that are now final, in order."""
        for record in records:
            self.records.append(record)
            point = (record.get("shuttlecock") or {}).get("image")
            if point:
                key = _cell({"p": point}, self.cell)
                self.counts[key] = self.counts.get(key, 0) + 1
                self.total += 1
                if self.counts[key] > self.config.static_share * self.min_total:
                    self.heavy.add(key)
        if now < self.next_check:
            return []
        self.next_check = now + self.check_every
        return self._final_calls(now)

    def finish(self):
        """The calls left at the end of the match."""
        return self._final_calls(np.inf)

    def _final_calls(self, now):
        config, height = self.config, self.frame_height
        static = static_cells({key: self.counts[key] for key in self.heavy}, max(self.total, self.min_total),
                              self.cell, self.mapper, config)
        samples = [s for s in _shuttle_samples(self.records, self.fps) if _cell(s, self.cell) not in static]
        groups = flight_groups(samples, height, config)
        rallies = [(first, last) for first, last in find_rallies(samples, height, config)
                   if is_play(samples, first, last, self.records, height, config)]

        def ended(group):
            return now - samples[group[-1]]["t"] > rally_gap(samples[group[-1]], height, config)

        records_by_frame = {int(record["frame"]): record for record in self.records}
        rest_end = config.rest_window + 2 * config.max_link_gap
        calls = []
        for index, (first, last) in enumerate(rallies):
            end_t = samples[last]["t"]
            if now - end_t <= max(rest_end, rally_gap(samples[last], height, config)):
                break
            stop = rallies[index + 1][0] if index + 1 < len(rallies) else len(samples)
            call = call_rally(self.called + 1, samples, first, last, stop, records_by_frame, self.fps, height,
                              self.mapper, config)
            flying = groups[-1] if groups and groups[-1][0] > last and not ended(groups[-1]) else None
            if flying and stop in (flying[0] - 1, len(samples)):
                other = len(samples) if stop == flying[0] - 1 else flying[0] - 1
                if asdict(call) != asdict(call_rally(self.called + 1, samples, first, last, other, records_by_frame,
                                                     self.fps, height, self.mapper, config)):
                    break
            self.called += 1
            calls.append(call)

        if len(calls) < len(rallies):
            keep = samples[rallies[len(calls)][0]]["frame"]
        elif groups and not ended(groups[-1]):
            keep = samples[max(0, groups[-1][0] - 1)]["frame"]
        else:   # a rally that starts later begins at most max_link_gap before its first flight sample
            keep = (now - 1.0) * self.fps if np.isfinite(now) else np.inf
        self.records = [record for record in self.records if record["frame"] >= keep]
        return calls


def summarize(calls):
    counts = {"rallies": len(calls), "in": 0, "out": 0, "needs_review": 0}
    for rally in calls:
        if rally.call in ("in", "out"):
            counts[rally.call] += 1
        if rally.confidence != "high":
            counts["needs_review"] += 1
    return counts


def write_report(path, calls, config):
    report = {
        "config": asdict(config),
        "summary": summarize(calls),
        "rallies": [asdict(rally) for rally in calls],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report


def draw_landing_map(path, calls, match_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..court.reference import BADMINTON_BACK_SERVICE_OFFSET, BADMINTON_SERVICE_LINE_FROM_NET

    width, length = BADMINTON_COURT_WIDTH, BADMINTON_COURT_LENGTH
    margin = BADMINTON_SINGLES_MARGIN
    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_facecolor("#2f6b3a")
    line = {"color": "white", "linewidth": 1.4}
    ax.plot([0, width, width, 0, 0], [0, 0, length, length, 0], **line)
    for x in (margin, width - margin):
        ax.plot([x, x], [0, length], **line)
    for y in (BADMINTON_BACK_SERVICE_OFFSET, length - BADMINTON_BACK_SERVICE_OFFSET,
              NET_Y - BADMINTON_SERVICE_LINE_FROM_NET, NET_Y + BADMINTON_SERVICE_LINE_FROM_NET):
        ax.plot([0, width], [y, y], **line)
    ax.plot([width / 2, width / 2], [0, NET_Y - BADMINTON_SERVICE_LINE_FROM_NET], **line)
    ax.plot([width / 2, width / 2], [NET_Y + BADMINTON_SERVICE_LINE_FROM_NET, length], **line)
    ax.plot([-0.3, width + 0.3], [NET_Y, NET_Y], color="#dddddd", linewidth=3)

    for rally in calls:
        if rally.landing_court is None:
            continue
        x, y = rally.landing_court
        color = "#2ecc71" if rally.call == "in" else "#e74c3c"
        filled = rally.confidence == "high"
        ax.scatter([x], [y], s=90, facecolors=color if filled else "none", edgecolors=color, linewidths=2, zorder=3)
        ax.annotate(str(rally.rally), (x, y), xytext=(6, 4), textcoords="offset points", color="white", fontsize=8)

    ax.set_xlim(-1.5, width + 1.5)
    ax.set_ylim(length + 1.5, -1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Landing points ({match_type})\ngreen = in, red = out, hollow = needs review", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_line_call_video(source_video, output_path, calls, fps, use_detect_frames, hold_seconds=2.0):
    """Draw each landing point and its call onto a copy of the video."""
    import cv2

    from ..media.video_audio import encode_vscode_compatible_mp4

    hold_frames = int(fps * hold_seconds)
    markers = {}
    for rally in calls:
        start = rally.landing_detect_frame if use_detect_frames else rally.landing_frame
        if start is None:
            continue
        for frame_index in range(start, start + hold_frames):
            markers[frame_index] = rally

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = output_path + ".tmp.mp4"
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        rally = markers.get(frame_index)
        if rally is not None:
            color = (80, 200, 60) if rally.call == "in" else (60, 60, 230)
            x, y = int(rally.landing_image[0]), int(rally.landing_image[1])
            cv2.circle(frame, (x, y), 18, color, 3, cv2.LINE_AA)
            label = f"Rally {rally.rally}: {rally.call.upper()} ({rally.margin_m * 100:+.0f} cm)"
            if rally.confidence != "high":
                label += " ?"
            cv2.rectangle(frame, (20, height - 70), (20 + 18 * len(label), height - 20), (0, 0, 0), -1)
            cv2.putText(frame, label, (32, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    try:
        encode_vscode_compatible_mp4(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_line_calls(output_dir, match_type="singles", render_video=False, config=None):
    """Compute line calls for an output directory and write the report files."""
    config = config or LineCallConfig(match_type=match_type)
    metadata, records = load_run(output_dir)
    calls = compute_line_calls(metadata, records, config)

    report_path = os.path.join(output_dir, "line_calls.json")
    report = write_report(report_path, calls, config)
    map_path = os.path.join(output_dir, "line_calls_map.png")
    draw_landing_map(map_path, calls, config.match_type)

    video_path = None
    if render_video:
        annotated = metadata.get("outputs", {}).get("video")
        use_detect_frames = bool(annotated and os.path.exists(annotated))
        source = annotated if use_detect_frames else metadata["video"]["path"]
        video_path = os.path.join(output_dir, f"line_calls_{metadata['video']['name']}.mp4")
        render_line_call_video(source, video_path, calls, float(metadata["video"]["fps"]), use_detect_frames)

    print_summary(calls)
    print(f"Line call report: {report_path}")
    print(f"Landing map: {map_path}")
    if video_path:
        print(f"Line call video: {video_path}")
    return report


def print_summary(calls):
    print(f"\nLine calls: {len(calls)} rallies")
    for rally in calls:
        side = SIDE_LABELS.get(rally.landing_side, "?")
        margin = f"{rally.margin_m * 100:+.0f} cm" if rally.margin_m is not None else "-"
        print(f"  #{rally.rally:<3} {rally.start_sec:7.2f}s-{rally.end_sec:7.2f}s  "
              f"{rally.call.upper():<4} {margin:>8}  {side:<4} side  "
              f"[{rally.confidence}, {rally.landing_method}]")
        for flag in rally.flags:
            print(f"        - {flag}")


def main():
    parser = argparse.ArgumentParser(description="Call shuttlecock landings in or out from an analysis output directory")
    parser.add_argument("output_dir", help="Analysis output directory, e.g. outputs/<video_name>")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--video", action="store_true", help="Also render a video with landing points and calls")
    parser.add_argument("--close-call-margin", type=float, default=LineCallConfig.close_call_margin,
                        help="Meters from a line that flags a call for review")
    parser.add_argument("--move-speed", type=float, default=LineCallConfig.move_speed,
                        help="Shuttle speed (screen heights per second) treated as flight")
    args = parser.parse_args()

    config = LineCallConfig(
        match_type=args.match_type,
        close_call_margin=args.close_call_margin,
        move_speed=args.move_speed,
    )
    run_line_calls(args.output_dir, render_video=args.video, config=config)


if __name__ == "__main__":
    main()
