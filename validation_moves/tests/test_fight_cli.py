import unittest

from validation_moves.fight_against_engine import (
    DEFAULT_HANDICAP_DEPTH,
    DEFAULT_HANDICAP_FACTOR,
    MIN_CLOCK_MS,
    _scale_handicapped_time,
    parse_args,
)


class FightAgainstEngineParseArgsTests(unittest.TestCase):
    def test_default_depth_mode(self) -> None:
        args = parse_args([])
        self.assertEqual(args.depth, 9)
        self.assertIsNone(args.time_per_move)
        self.assertIsNone(args.clock)
        self.assertIsNone(args.opponent_time_per_move)
        self.assertTrue(args.handicap_enabled)
        self.assertAlmostEqual(args.handicap_factor, DEFAULT_HANDICAP_FACTOR)
        self.assertEqual(args.handicap_depth, DEFAULT_HANDICAP_DEPTH)

    def test_explicit_depth(self) -> None:
        args = parse_args(["--depth", "7"])
        self.assertEqual(args.depth, 7)
        self.assertIsNone(args.time_per_move)
        self.assertIsNone(args.clock)

    def test_depth_and_time_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--depth", "8", "--time-per-move", "1.0"])

    def test_time_requires_positive_value(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--time-per-move", "0"])

    def test_time_mode_with_opponent_override(self) -> None:
        args = parse_args(["--time-per-move", "2.0", "--opponent-time-per-move", "1.0"])
        self.assertIsNone(args.depth)
        self.assertAlmostEqual(args.time_per_move, 2.0)
        self.assertAlmostEqual(args.opponent_time_per_move, 1.0)

    def test_clock_mode_defaults(self) -> None:
        args = parse_args(["--clock", "60"])
        self.assertIsNone(args.depth)
        self.assertIsNone(args.time_per_move)
        self.assertAlmostEqual(args.clock, 60.0)
        self.assertIsNone(args.opponent_clock)
        self.assertIsNone(args.increment)
        self.assertIsNone(args.opponent_increment)

    def test_clock_override_options(self) -> None:
        args = parse_args(
            [
                "--clock",
                "30",
                "--opponent-clock",
                "12",
                "--increment",
                "0.5",
                "--opponent-increment",
                "0.25",
                "--moves-to-go",
                "35",
            ]
        )
        self.assertAlmostEqual(args.clock, 30.0)
        self.assertAlmostEqual(args.opponent_clock, 12.0)
        self.assertAlmostEqual(args.increment, 0.5)
        self.assertAlmostEqual(args.opponent_increment, 0.25)
        self.assertEqual(args.moves_to_go, 35)

    def test_clock_dependencies(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--opponent-clock", "20"])
        with self.assertRaises(SystemExit):
            parse_args(["--increment", "1.0"])

    def test_no_handicap_flag(self) -> None:
        args = parse_args(["--no-handicap"])
        self.assertFalse(args.handicap_enabled)
        self.assertEqual(args.handicap_factor, 1.0)
        self.assertEqual(args.handicap_depth, 0)

    def test_custom_handicap_settings(self) -> None:
        args = parse_args(["--handicap-factor", "0.25", "--handicap-depth", "3"])
        self.assertTrue(args.handicap_enabled)
        self.assertAlmostEqual(args.handicap_factor, 0.25)
        self.assertEqual(args.handicap_depth, 3)

    def test_stockfish_flag_conflict(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--stockfish", "--opponent", "other-engine"])


class ScaleHandicapTests(unittest.TestCase):
    def test_scale_handicap_applies_factor(self) -> None:
        self.assertEqual(_scale_handicapped_time(1000, 0.5), 500)

    def test_scale_handicap_never_below_min(self) -> None:
        self.assertEqual(_scale_handicapped_time(1, 0.01), MIN_CLOCK_MS)


if __name__ == "__main__":
    unittest.main()
