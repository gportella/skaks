import unittest

from move_validation.fight_against_engine import (
    STOCKFISH_TIME_FACTOR,
    _scale_stockfish_movetime,
    parse_args,
)


class FightAgainstEngineParseArgsTests(unittest.TestCase):
    def test_default_depth_mode(self) -> None:
        args = parse_args([])
        self.assertEqual(args.depth, 9)
        self.assertIsNone(args.time_per_move)
        self.assertIsNone(args.stockfish_time_per_move)
        self.assertFalse(args.stockfish_time_overridden)
        self.assertIsNone(args.clock)

    def test_explicit_depth(self) -> None:
        args = parse_args(["--depth", "7"])
        self.assertEqual(args.depth, 7)
        self.assertIsNone(args.time_per_move)
        self.assertIsNone(args.clock)

    def test_time_mode_sets_defaults(self) -> None:
        args = parse_args(["--time-per-move", "1.5"])
        self.assertIsNone(args.depth)
        self.assertAlmostEqual(args.time_per_move, 1.5)
        self.assertAlmostEqual(args.stockfish_time_per_move, 1.5)
        self.assertFalse(args.stockfish_time_overridden)
        self.assertIsNone(args.clock)

    def test_time_mode_allows_stockfish_override(self) -> None:
        args = parse_args(
            ["--time-per-move", "2.0", "--stockfish-time-per-move", "1.0"]
        )
        self.assertAlmostEqual(args.time_per_move, 2.0)
        self.assertAlmostEqual(args.stockfish_time_per_move, 1.0)
        self.assertTrue(args.stockfish_time_overridden)
        self.assertIsNone(args.clock)

    def test_depth_and_time_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--depth", "8", "--time-per-move", "1.0"])

    def test_time_requires_positive_value(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--time-per-move", "0"])

    def test_clock_mode_defaults(self) -> None:
        args = parse_args(["--clock", "60"])
        self.assertIsNone(args.depth)
        self.assertIsNone(args.time_per_move)
        self.assertAlmostEqual(args.clock, 60.0)
        self.assertIsNone(args.stockfish_clock)
        self.assertIsNone(args.increment)
        self.assertIsNone(args.stockfish_increment)

    def test_clock_override_options(self) -> None:
        args = parse_args(
            [
                "--clock",
                "30",
                "--stockfish-clock",
                "12",
                "--increment",
                "0.5",
                "--stockfish-increment",
                "0.25",
                "--moves-to-go",
                "35",
            ]
        )
        self.assertAlmostEqual(args.clock, 30.0)
        self.assertAlmostEqual(args.stockfish_clock, 12.0)
        self.assertAlmostEqual(args.increment, 0.5)
        self.assertAlmostEqual(args.stockfish_increment, 0.25)
        self.assertEqual(args.moves_to_go, 35)

    def test_stockfish_clock_requires_primary_clock(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--stockfish-clock", "20"])

    def test_increment_requires_clock(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--increment", "1.0"])

    def test_stockfish_time_scaling_without_override(self) -> None:
        skaks_ms = 2000
        expected = max(int(round(skaks_ms * STOCKFISH_TIME_FACTOR)), 1)
        self.assertEqual(
            _scale_stockfish_movetime(skaks_ms, skaks_ms, override=False), expected
        )

    def test_stockfish_time_respects_override(self) -> None:
        self.assertEqual(
            _scale_stockfish_movetime(2000, 750, override=True),
            750,
        )


if __name__ == "__main__":
    unittest.main()
