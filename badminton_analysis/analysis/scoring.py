"""Automatic badminton scoring from line calls.

Each rally's winner comes from its line call: a shuttle that lands in on one
side wins the rally for the other side, and a shuttle that lands out is a
point against whoever hit it (assumed to be the player across the net). The
winners then run through the badminton rules: rally-point scoring to 21, win
by 2, capped at 30, best of 3 games, ends changed after each game and at 11 in
the deciding game.

Run it on an existing output directory after line calls:

    python -m badminton_analysis.analysis.scoring outputs/<video_name> --start-score 12-6
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field

from .line_calls import SIDE_LABELS, LineCallConfig, compute_line_calls, load_run


OTHER_SIDE = {"upper": "lower", "lower": "upper"}


class MatchScore:
    """Badminton score keeper for two players (or pairs) indexed 0 and 1."""

    def __init__(self, player0_side="lower", points_to_win=21, point_cap=30, games_to_win=2,
                 interval_points=11, start_points=(0, 0), start_games=(0, 0), server=None):
        self.points_to_win = points_to_win
        self.point_cap = point_cap
        self.games_to_win = games_to_win
        self.interval_points = interval_points
        self.points = list(start_points)
        self.games = list(start_games)
        self.completed_games = []
        self.server = server
        self.match_over = False
        # Player 0's side in game 1; sides flip after every game and at the deciding-game interval.
        self._player0_start_side = player0_side
        self._deciding_game_switched = self._is_deciding_game() and max(self.points) >= interval_points

    @property
    def game_number(self):
        return sum(self.games) + 1

    def _is_deciding_game(self):
        return self.games[0] == self.games[1] == self.games_to_win - 1

    def side_of(self, player):
        flips = sum(self.games) + (1 if self._deciding_game_switched else 0)
        side = self._player0_start_side if flips % 2 == 0 else OTHER_SIDE[self._player0_start_side]
        return side if player == 0 else OTHER_SIDE[side]

    def player_on(self, side):
        return 0 if self.side_of(0) == side else 1

    def service_court(self):
        if self.server is None:
            return None
        return "right" if self.points[self.server] % 2 == 0 else "left"

    def _game_won_by(self):
        for player in (0, 1):
            mine, theirs = self.points[player], self.points[1 - player]
            if mine >= self.point_cap or (mine >= self.points_to_win and mine - theirs >= 2):
                return player
        return None

    def award(self, player):
        """Give the rally to a player. Returns a dict describing what happened."""
        if self.match_over:
            raise RuntimeError("Match is already over")
        event = {
            "game": self.game_number,
            "winner": player,
            "score_before": list(self.points),
            "change_ends": False,
            "game_over": False,
            "match_over": False,
        }
        self.points[player] += 1
        self.server = player
        event["score_after"] = list(self.points)

        game_winner = self._game_won_by()
        if game_winner is not None:
            self.games[game_winner] += 1
            self.completed_games.append(list(self.points))
            event["game_over"] = True
            event["games_after"] = list(self.games)
            self.points = [0, 0]
            self._deciding_game_switched = False
            if self.games[game_winner] >= self.games_to_win:
                self.match_over = True
                event["match_over"] = True
            else:
                event["change_ends"] = True
        elif self._is_deciding_game() and not self._deciding_game_switched and max(self.points) == self.interval_points:
            self._deciding_game_switched = True
            event["change_ends"] = True
        return event

    def score_text(self):
        games = " ".join(f"{a}-{b}" for a, b in self.completed_games)
        current = f"{self.points[0]}-{self.points[1]}"
        return f"{games} | {current}" if games else current


@dataclass
class RallyResult:
    rally: int
    end_sec: float
    winner_side: str = None
    winner: int = None
    reason: str = ""
    confidence: str = "low"
    score_after: list = None
    game: int = None
    flags: list = field(default_factory=list)


def rally_winner_side(call):
    """Return (winner_side, reason) for a line call, or (None, reason) when it cannot be decided."""
    side = call.landing_side
    if side is None or call.call not in ("in", "out"):
        return None, "no landing found"
    if call.call == "in":
        return OTHER_SIDE[side], f"landed in on the {SIDE_LABELS[side]} side"
    return side, f"{SIDE_LABELS[OTHER_SIDE[side]]} player hit it out"


def score_rallies(calls, player0_side="lower", start_points=(0, 0), start_games=(0, 0), server=None):
    """Run line calls through the scoring rules. Returns (results, MatchScore)."""
    match = MatchScore(player0_side=player0_side, start_points=start_points, start_games=start_games, server=server)
    results = []
    for call in calls:
        result = RallyResult(rally=call.rally, end_sec=call.end_sec, flags=list(call.flags))
        winner_side, result.reason = rally_winner_side(call)
        if winner_side is None or match.match_over:
            if match.match_over:
                result.reason = "match already over"
            results.append(result)
            continue
        result.winner_side = winner_side
        result.winner = match.player_on(winner_side)
        result.confidence = call.confidence
        if any("from the net" in flag for flag in call.flags):
            result.confidence = "low"
        event = match.award(result.winner)
        result.game = event["game"]
        result.score_after = event["score_after"]
        if event["change_ends"]:
            result.flags.append("players change ends")
        results.append(result)
    return results, match


def parse_pair(text):
    a, b = text.split("-")
    return int(a), int(b)


def run_scoring(output_dir, match_type="singles", player0_side="lower", start_points=(0, 0),
                start_games=(0, 0), names=("Player 1", "Player 2")):
    metadata, records = load_run(output_dir)
    calls = compute_line_calls(metadata, records, LineCallConfig(match_type=match_type))
    results, match = score_rallies(calls, player0_side=player0_side, start_points=start_points, start_games=start_games)

    report = {
        "players": {"0": names[0], "1": names[1]},
        "player0_start_side": player0_side,
        "start_score": {"points": list(start_points), "games": list(start_games)},
        "final_score": match.score_text(),
        "games": list(match.games),
        "rallies": [asdict(result) for result in results],
    }
    path = os.path.join(output_dir, "score_report.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"\nScore ({names[0]} - {names[1]}):")
    for result in results:
        who = names[result.winner] if result.winner is not None else "?"
        score = "-".join(str(p) for p in result.score_after) if result.score_after else "-"
        print(f"  #{result.rally:<3} {result.end_sec:7.2f}s  {score:>6}  point to {who:<12} "
              f"{result.reason} [{result.confidence}]")
    print(f"Final: {match.score_text()}")
    print(f"Score report: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Score a match automatically from an analysis output directory")
    parser.add_argument("output_dir")
    parser.add_argument("--match-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--player1-side", choices=["upper", "lower"], default="lower",
                        help="Side player 1 is on when the video starts (upper = far, lower = near)")
    parser.add_argument("--start-score", default="0-0", help="Points at the start of the video, player 1 first")
    parser.add_argument("--start-games", default="0-0", help="Games won at the start of the video, player 1 first")
    parser.add_argument("--names", default="Player 1,Player 2")
    args = parser.parse_args()
    run_scoring(
        args.output_dir,
        match_type=args.match_type,
        player0_side=args.player1_side,
        start_points=parse_pair(args.start_score),
        start_games=parse_pair(args.start_games),
        names=tuple(args.names.split(",", 1)),
    )


if __name__ == "__main__":
    main()
