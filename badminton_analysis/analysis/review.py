"""Review automatic rally calls one by one and build the score from the confirmed winners.

For each rally the end of the clip is replayed with the automatic call drawn on
it. One key accepts the call or overrides it:

    Enter / Space  accept the automatic winner
    n              near side won the rally
    f              far side won the rally
    x              not a rally (warm-up, replay, shuttle knocked back)
    r              replay the clip
    b              go back one rally
    q / Esc        save and quit (run again to continue where you left off)

Decisions are saved to review.json after every key, and the score to
review_report.json.

    python -m badminton_analysis.analysis.review outputs/<video_name> --names "Lee,Tan" --player1-side lower
"""

import argparse
import json
import os

from .line_calls import SIDE_LABELS, LineCallConfig, compute_line_calls, load_run
from .scoring import MatchScore, parse_pair, rally_winner_side


NOT_A_RALLY = "none"


class ReviewSession:
    """Reviewer decisions over a list of line calls, and the score they produce."""

    def __init__(self, calls, player0_side="lower", start_points=(0, 0), start_games=(0, 0),
                 names=("Player 1", "Player 2"), decisions=None):
        self.calls = list(calls)
        self.player0_side = player0_side
        self.start_points = tuple(start_points)
        self.start_games = tuple(start_games)
        self.names = tuple(names)
        known = {call.rally for call in self.calls}
        # rally number -> "upper", "lower" or NOT_A_RALLY
        self.decisions = {int(k): v for k, v in (decisions or {}).items() if int(k) in known}

    def predicted_side(self, index):
        return rally_winner_side(self.calls[index])[0]

    def decide(self, index, winner_side):
        if winner_side not in ("upper", "lower", NOT_A_RALLY):
            raise ValueError(f"unknown decision: {winner_side}")
        self.decisions[self.calls[index].rally] = winner_side

    def accept(self, index):
        """Accept the automatic call. Returns False when there is no call to accept."""
        side = self.predicted_side(index)
        if side is None:
            return False
        self.decide(index, side)
        return True

    def winner_side(self, index):
        """The reviewed winner if there is one, else the automatic call."""
        rally = self.calls[index].rally
        return self.decisions.get(rally, self.predicted_side(index))

    def first_unreviewed(self):
        for index, call in enumerate(self.calls):
            if call.rally not in self.decisions:
                return index
        return len(self.calls)

    def replay(self, upto=None):
        """Score rallies [0, upto) and return (rows, MatchScore)."""
        match = MatchScore(player0_side=self.player0_side, start_points=self.start_points,
                           start_games=self.start_games)
        rows = []
        for index, call in enumerate(self.calls[:upto]):
            side = self.winner_side(index)
            row = {"rally": call.rally, "end_sec": call.end_sec,
                   "reviewed": call.rally in self.decisions,
                   "changed": call.rally in self.decisions and side != self.predicted_side(index),
                   "winner_side": None if side == NOT_A_RALLY else side,
                   "winner": None, "score_before": list(match.points), "score_after": None,
                   "game": match.game_number, "sides": self.side_names(match)}
            if side in ("upper", "lower") and not match.match_over:
                row["winner"] = match.player_on(side)
                event = match.award(row["winner"])
                row["score_after"], row["game"] = event["score_after"], event["game"]
            rows.append(row)
        return rows, match

    def side_names(self, match):
        """Who is on which side at the current point of the match."""
        return {side: self.names[match.player_on(side)] for side in ("upper", "lower")}

    def to_json(self):
        return {
            "player0_start_side": self.player0_side,
            "start_score": {"points": list(self.start_points), "games": list(self.start_games)},
            "names": list(self.names),
            "decisions": {str(k): v for k, v in sorted(self.decisions.items())},
        }

    def report(self):
        rows, match = self.replay()
        reviewed = sum(row["reviewed"] for row in rows)
        return {
            "players": {"0": self.names[0], "1": self.names[1]},
            "final_score": match.score_text(),
            "games": list(match.games),
            "rallies_reviewed": reviewed,
            "rallies_total": len(rows),
            "calls_changed": sum(row["changed"] for row in rows),
            "rallies": rows,
        }


def load_session(output_dir, calls, player0_side, start_points, start_games, names):
    path = os.path.join(output_dir, "review.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            saved = json.load(file)
        # The setup the review was started with wins, so resuming keeps the same score.
        return ReviewSession(
            calls,
            player0_side=saved.get("player0_start_side", player0_side),
            start_points=tuple(saved.get("start_score", {}).get("points", start_points)),
            start_games=tuple(saved.get("start_score", {}).get("games", start_games)),
            names=tuple(saved.get("names", names)),
            decisions=saved.get("decisions"),
        )
    return ReviewSession(calls, player0_side, start_points, start_games, names)


def save_session(output_dir, session):
    for name, data in (("review.json", session.to_json()), ("review_report.json", session.report())):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")


def describe_call(call, side_names):
    if call.call not in ("in", "out") or call.landing_side is None:
        return "No landing found"
    side = SIDE_LABELS[call.landing_side]
    margin = f" by {abs(call.margin_m):.2f} m" if call.margin_m is not None else ""
    return f"{call.call.upper()} on the {side} side{margin} ({side_names[call.landing_side]}'s half)"


# --- Player ---------------------------------------------------------------------------------

KEY_ENTER = (13, 10)
KEY_ESC = 27
WINDOW = "Rally review"
COLORS = {"upper": (255, 170, 60), "lower": (60, 200, 255)}


def _draw_panel(frame, lines, help_line):
    import cv2

    h, w = frame.shape[:2]
    scale = h / 720.0
    line_h = int(30 * scale)
    top = len(lines) * line_h + int(12 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - line_h - int(8 * scale)), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (int(12 * scale), (i + 1) * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * scale, color, max(1, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, help_line, (int(12 * scale), h - int(12 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 * scale, (220, 220, 220), max(1, int(1.5 * scale)), cv2.LINE_AA)


def _clip_frames(call, fps, lead_seconds=3.0, tail_seconds=1.5):
    end_frame = max(call.end_frame, call.landing_frame or 0)
    first = max(call.start_frame, end_frame - int(lead_seconds * fps))
    return first, end_frame + int(tail_seconds * fps)


def play_rally(cap, call, fps, lines, help_line, max_width=1280):
    """Play one rally's ending, then hold the last frame. Returns the key pressed."""
    import cv2

    first, last = _clip_frames(call, fps)
    # Records count frames from 1; the capture counts from 0.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
    delay = max(1, int(1000 / fps))
    shown = None
    for frame_number in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if call.landing_image and call.landing_frame and frame_number >= call.landing_frame:
            x, y = (int(v) for v in call.landing_image)
            color = (0, 0, 255) if call.call == "out" else (0, 220, 0)
            cv2.circle(frame, (x, y), int(frame.shape[0] * 0.02), color, 3, cv2.LINE_AA)
        _draw_panel(frame, lines, help_line)
        if frame.shape[1] > max_width:
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * max_width / frame.shape[1])))
        shown = frame
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key != 255:
            return key
    while True:
        if shown is not None:
            cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(0) & 0xFF
        if key != 255:
            return key


def run_review(output_dir, match_type="singles", player0_side="lower", start_points=(0, 0),
               start_games=(0, 0), names=("Player 1", "Player 2"), video_path=None, skip_high=False):
    import cv2

    metadata, records = load_run(output_dir)
    calls = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    if not calls:
        print("No rallies found.")
        return None
    session = load_session(output_dir, calls, player0_side, start_points, start_games, names)
    names = session.names
    if skip_high:
        for index, call in enumerate(calls):
            if call.rally not in session.decisions and call.confidence == "high":
                session.accept(index)

    video_path = video_path or metadata["video"]["path"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path} (pass --video)")
    fps = float(metadata["video"]["fps"])
    help_line = "Enter: accept   n: near won   f: far won   x: not a rally   r: replay   b: back   q: quit"

    index = session.first_unreviewed()
    if skip_high:
        while index < len(calls) and calls[index].rally in session.decisions:
            index += 1
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while index < len(calls):
        call = calls[index]
        rows, match = session.replay(index)
        sides = session.side_names(match)
        predicted = session.predicted_side(index)
        decided = session.decisions.get(call.rally)
        lines = [
            (f"Rally {index + 1}/{len(calls)}  ({call.end_sec:.0f}s)   Game {match.game_number}   "
             f"{names[0]} {match.points[0]} - {match.points[1]} {names[1]}", (255, 255, 255)),
            (f"Far: {sides['upper']}    Near: {sides['lower']}", (200, 200, 200)),
            (f"Call: {describe_call(call, sides)}", (255, 255, 255)),
            (f"Auto: {SIDE_LABELS[predicted] + ' side (' + sides[predicted] + ') wins' if predicted else 'undecided'}"
             f"   [{call.confidence}]", COLORS.get(predicted, (180, 180, 180))),
        ]
        if decided:
            label = "not a rally" if decided == NOT_A_RALLY else f"{SIDE_LABELS[decided]} side wins"
            lines.append((f"Your decision so far: {label}", (120, 255, 120)))
        for flag in call.flags[:2]:
            lines.append((f"! {flag}", (80, 200, 255)))

        key = play_rally(cap, call, fps, lines, help_line)
        if key in KEY_ENTER or key == ord(" "):
            if not session.accept(index):
                continue  # nothing to accept: the reviewer has to pick a side
        elif key == ord("n"):
            session.decide(index, "lower")
        elif key == ord("f"):
            session.decide(index, "upper")
        elif key == ord("x"):
            session.decide(index, NOT_A_RALLY)
        elif key == ord("r"):
            continue
        elif key == ord("b"):
            index = max(0, index - 1)
            continue
        elif key in (ord("q"), KEY_ESC):
            break
        else:
            continue
        save_session(output_dir, session)
        index += 1

    cap.release()
    cv2.destroyAllWindows()
    save_session(output_dir, session)
    report = session.report()
    print(f"Reviewed {report['rallies_reviewed']}/{report['rallies_total']} rallies, "
          f"changed {report['calls_changed']} automatic calls")
    print(f"Score ({names[0]} - {names[1]}): {report['final_score']}")
    print(f"Saved: {os.path.join(output_dir, 'review_report.json')}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Review automatic rally calls and build the score")
    parser.add_argument("output_dir")
    parser.add_argument("--video", default=None, help="Video file, if it has moved since the analysis")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--player1-side", choices=["upper", "lower"], default="lower",
                        help="Side player 1 is on when the video starts (upper = far, lower = near)")
    parser.add_argument("--start-score", default="0-0", help="Points at the start of the video, player 1 first")
    parser.add_argument("--start-games", default="0-0", help="Games won at the start of the video, player 1 first")
    parser.add_argument("--names", default="Player 1,Player 2")
    parser.add_argument("--skip-high", action="store_true",
                        help="Accept high-confidence calls without showing them")
    args = parser.parse_args()
    run_review(
        args.output_dir,
        match_type=args.match_type,
        player0_side=args.player1_side,
        start_points=parse_pair(args.start_score),
        start_games=parse_pair(args.start_games),
        names=tuple(args.names.split(",", 1)),
        video_path=args.video,
        skip_high=args.skip_high,
    )


if __name__ == "__main__":
    main()
