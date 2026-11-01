"""Measure automatic scoring against the true point winners read from a scoreboard.

Each scoreboard point is matched to the last predicted rally that ended since
the previous scoreboard point (the broadcast updates the score a few seconds
after the rally). Which side each player is on comes from replaying the true
score, so one wrong call does not throw off the side mapping for the rest of
the match.

    python -m badminton_analysis.analysis.evaluate_scoring outputs/<video_name> scoreboard_truth.json --start-score 12-6
"""

import argparse
import json
import os

from .line_calls import LineCallConfig, compute_line_calls, load_run
from .scoring import MatchScore, parse_pair, rally_winner_side


def evaluate(calls, truth_points, player0_side, start_points=(0, 0), start_games=(0, 0), lead_tolerance=1.0):
    """Compare predicted rally winners with true points.

    truth_points: list of (time_sec, row) where row 0 = player 1 won the point.
    Returns a dict with counts and one entry per true point.
    """
    truth_match = MatchScore(player0_side=player0_side, start_points=start_points, start_games=start_games)
    used = set()
    rows = []
    previous_time = float("-inf")
    for time_sec, true_row in truth_points:
        window = [
            index for index, call in enumerate(calls)
            if previous_time < call.end_sec <= time_sec + lead_tolerance and index not in used
        ]
        entry = {"time_sec": time_sec, "true_winner": true_row, "rally": None,
                 "predicted_winner": None, "result": "missed", "confidence": None, "reason": ""}
        if window:
            index = window[-1]
            used.update(window)
            call = calls[index]
            entry["rally"] = call.rally
            entry["extra_rallies"] = [calls[i].rally for i in window[:-1]]
            entry["confidence"] = call.confidence
            winner_side, entry["reason"] = rally_winner_side(call)
            if winner_side is None:
                entry["result"] = "undecided"
            else:
                entry["predicted_winner"] = truth_match.player_on(winner_side)
                entry["result"] = "correct" if entry["predicted_winner"] == true_row else "wrong"
        rows.append(entry)
        if not truth_match.match_over:
            truth_match.award(true_row)
        previous_time = time_sec + lead_tolerance

    counts = {key: sum(1 for row in rows if row["result"] == key) for key in ("correct", "wrong", "missed", "undecided")}
    extra = sum(len(row.get("extra_rallies", [])) for row in rows) + sum(
        1 for index, call in enumerate(calls) if index not in used and call.end_sec <= previous_time
    )
    high = [row for row in rows if row["confidence"] == "high" and row["result"] in ("correct", "wrong")]
    total = len(rows)
    return {
        "player0_start_side": player0_side,
        "true_points": total,
        **counts,
        "extra_rallies": extra,
        "accuracy": round(counts["correct"] / total, 3) if total else None,
        "high_confidence_accuracy": round(sum(r["result"] == "correct" for r in high) / len(high), 3) if high else None,
        "high_confidence_points": len(high),
        "points": rows,
    }


def predicted_final_score(report, start_points, start_games):
    """Score the match using only the predictions, the way it would run with no scoreboard."""
    match = MatchScore(start_points=start_points, start_games=start_games)
    for row in report["points"]:
        if row["predicted_winner"] is not None and not match.match_over:
            match.award(row["predicted_winner"])
    return match.score_text()


def true_final_score(report, start_points, start_games):
    match = MatchScore(start_points=start_points, start_games=start_games)
    for row in report["points"]:
        if not match.match_over:
            match.award(row["true_winner"])
    return match.score_text()


def load_truth(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [(change["time_sec"], change["rows"][0]) for change in data["changes"] if change["kind"] == "point"]


def run_evaluation(output_dir, truth_path, match_type="singles", player0_side="auto",
                   start_points=(0, 0), start_games=(0, 0)):
    metadata, records = load_run(output_dir)
    calls = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    truth_points = load_truth(truth_path)

    sides = ["upper", "lower"] if player0_side == "auto" else [player0_side]
    reports = [evaluate(calls, truth_points, side, start_points, start_games) for side in sides]
    report = max(reports, key=lambda item: item["correct"])
    report["side_chosen_automatically"] = player0_side == "auto"
    report["predicted_final_score"] = predicted_final_score(report, start_points, start_games)
    report["true_final_score"] = true_final_score(report, start_points, start_games)
    report["predicted_rallies"] = len(calls)

    path = os.path.join(output_dir, "scoring_accuracy.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")

    print(f"\nScoring accuracy ({len(truth_points)} scoreboard points, {len(calls)} predicted rallies)")
    print(f"  Player 1 started on the {report['player0_start_side']} side"
          f"{' (chosen automatically)' if report['side_chosen_automatically'] else ''}")
    print(f"  Correct winner: {report['correct']}/{report['true_points']}"
          f" ({(report['accuracy'] or 0) * 100:.0f}%)")
    print(f"  Wrong: {report['wrong']}  Missed rally: {report['missed']}  "
          f"Undecided: {report['undecided']}  Extra rallies: {report['extra_rallies']}")
    if report["high_confidence_points"]:
        print(f"  High-confidence calls: {report['high_confidence_accuracy'] * 100:.0f}% of "
              f"{report['high_confidence_points']}")
    print(f"  Final score: predicted {report['predicted_final_score']}, true {report['true_final_score']}")
    for row in report["points"]:
        if row["result"] != "correct":
            print(f"    {row['time_sec']:8.2f}s  {row['result']:<9} rally {row['rally']}  {row['reason']}")
    print(f"Accuracy report: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare automatic scoring with scoreboard truth")
    parser.add_argument("output_dir")
    parser.add_argument("truth_path", help="scoreboard_truth.json from badminton_analysis.analysis.scoreboard")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--player1-side", choices=["auto", "upper", "lower"], default="auto",
                        help="Side the top scoreboard row's player is on when the video starts")
    parser.add_argument("--start-score", default="0-0", help="Points at the start of the video, top row first")
    parser.add_argument("--start-games", default="0-0")
    args = parser.parse_args()
    run_evaluation(args.output_dir, args.truth_path, args.match_type, args.player1_side,
                   parse_pair(args.start_score), parse_pair(args.start_games))


if __name__ == "__main__":
    main()
