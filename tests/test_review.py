import os
import tempfile
import unittest

from badminton_analysis.analysis.line_calls import RallyCall
from badminton_analysis.analysis.review import NOT_A_RALLY, ReviewSession, load_session, save_session
from badminton_analysis.analysis.review_web import ReviewApp
from badminton_analysis.analysis.scoring import score_rallies


def call(rally, result, side, confidence="high"):
    return RallyCall(rally=rally, start_frame=rally * 100, end_frame=rally * 100 + 50, start_sec=rally * 10.0,
                     end_sec=rally * 10.0 + 5, landing_side=side, call=result, confidence=confidence)


CALLS = [
    call(1, "in", "upper"),    # near side wins
    call(2, "out", "upper"),   # far side wins
    call(3, "in", "lower"),    # far side wins
    call(4, "unknown", None),  # no landing found
]


class ReviewSessionTests(unittest.TestCase):
    def test_unreviewed_session_scores_like_automatic_scoring(self):
        session = ReviewSession(CALLS, player0_side="lower")
        _, match = session.replay()
        _, auto = score_rallies(CALLS, player0_side="lower")
        self.assertEqual(match.points, auto.points)
        self.assertEqual(match.points, [1, 2])

    def test_override_and_not_a_rally_change_the_score(self):
        session = ReviewSession(CALLS, player0_side="lower")
        session.decide(1, "lower")          # reviewer: near side actually won rally 2
        session.decide(2, NOT_A_RALLY)      # rally 3 was a replay
        session.decide(3, "upper")          # undecided rally: far side won
        rows, match = session.replay()
        self.assertEqual(match.points, [2, 1])
        self.assertEqual([row["changed"] for row in rows], [False, True, True, True])
        self.assertIsNone(rows[2]["winner"])

    def test_accept_needs_an_automatic_call(self):
        session = ReviewSession(CALLS)
        self.assertTrue(session.accept(0))
        self.assertFalse(session.accept(3))
        self.assertEqual(session.first_unreviewed(), 1)

    def test_replay_upto_gives_score_before_a_rally(self):
        session = ReviewSession(CALLS, player0_side="lower")
        _, match = session.replay(2)
        self.assertEqual(match.points, [1, 1])

    def test_winner_maps_to_players_after_ends_change(self):
        calls = [call(i, "in", "upper") for i in range(1, 23)]  # near side wins every rally
        session = ReviewSession(calls, player0_side="lower")
        rows, match = session.replay()
        self.assertEqual(rows[20]["score_after"], [21, 0])
        self.assertEqual(rows[21]["winner"], 1)  # player 1 is on the near side in game 2
        self.assertEqual(match.games, [1, 0])

    def test_save_and_resume(self):
        out = tempfile.mkdtemp()
        session = ReviewSession(CALLS, player0_side="upper", start_points=(5, 3), names=("Lee", "Tan"))
        session.decide(0, "upper")
        session.decide(1, NOT_A_RALLY)
        save_session(out, session)
        self.assertTrue(os.path.exists(os.path.join(out, "review_report.json")))

        resumed = load_session(out, CALLS, "lower", (0, 0), (0, 0), ("A", "B"))
        self.assertEqual(resumed.decisions, {1: "upper", 2: NOT_A_RALLY})
        self.assertEqual((resumed.player0_side, resumed.start_points, resumed.names), ("upper", (5, 3), ("Lee", "Tan")))
        self.assertEqual(resumed.first_unreviewed(), 2)

    def test_decisions_for_rallies_that_no_longer_exist_are_dropped(self):
        session = ReviewSession(CALLS, decisions={"1": "upper", "99": "lower"})
        self.assertEqual(session.decisions, {1: "upper"})


class ReviewWebTests(unittest.TestCase):
    def app(self, court_frames):
        metadata = {"video": {"fps": 10.0, "width": 1280, "height": 720}}
        calls = [RallyCall(rally=1, start_frame=11, end_frame=101, start_sec=1.0, end_sec=10.0,
                           landing_frame=104, landing_image=[600.0, 400.0], landing_side="upper",
                           call="in", confidence="high")]
        return ReviewApp(tempfile.mkdtemp(), "video.mp4", ReviewSession(calls), metadata, calls,
                         court_frames=court_frames)

    def test_clip_covers_the_end_of_the_rally_and_the_landing(self):
        rally = self.app(range(1, 200)).state()["rallies"][0]
        self.assertEqual((rally["clip_start"], rally["landing_time"], rally["clip_end"]), (7.3, 10.3, 11.8))

    def test_clip_stops_when_the_broadcast_cuts_away_from_the_court(self):
        rally = self.app(range(1, 107)).state()["rallies"][0]
        self.assertEqual(rally["clip_end"], 10.5)

    def test_decisions_are_saved(self):
        app = self.app(range(1, 200))
        app.decide(0, "far")
        state = app.state()
        self.assertEqual((state["reviewed"], state["changed"]), (1, 1))
        self.assertTrue(os.path.exists(os.path.join(app.output_dir, "review.json")))
        with self.assertRaises(ValueError):
            app.decide(0, "sideways")


if __name__ == "__main__":
    unittest.main()
