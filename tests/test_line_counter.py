import unittest

from app.models_ai.line_counter import LineCounter


def track(track_id, cy, cx=50):
    return {
        "track_id": track_id,
        "cx": cx,
        "cy": cy,
        "bbox": (40, int(cy) - 5, 60, int(cy) + 5),
        "cls_name": "truck",
        "conf": 0.9,
    }


class LineCounterTests(unittest.TestCase):
    def make_counter(self, rule="down"):
        return LineCounter((0, 50, 100, 50), rule, dedup_seconds=45,
                           hysteresis_px=5)

    def test_crossing_down_counts_one_in(self):
        counter = self.make_counter("down")
        self.assertEqual(counter.update([track(1, 35)]), [])
        events = counter.update([track(1, 65)])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["direction"], "IN")
        self.assertEqual(counter.count_in, 1)
        self.assertEqual(counter.count_out, 0)

    def test_deadband_ignores_jitter_around_line(self):
        counter = self.make_counter()
        counter.update([track(7, 40)])

        self.assertEqual(counter.update([track(7, 48)]), [])
        self.assertEqual(counter.update([track(7, 52)]), [])
        events = counter.update([track(7, 60)])

        self.assertEqual(len(events), 1)
        self.assertEqual(counter.count_in, 1)

    def test_same_side_is_not_counted(self):
        counter = self.make_counter()
        counter.update([track(2, 30)])
        counter.update([track(2, 38)])
        counter.update([track(2, 42)])

        self.assertEqual(counter.count_in, 0)
        self.assertEqual(counter.count_out, 0)

    def test_up_rule_reverses_direction(self):
        counter = self.make_counter("up")
        counter.update([track(3, 35)])
        events = counter.update([track(3, 65)])

        self.assertEqual(events[0]["direction"], "OUT")
        self.assertEqual(counter.count_out, 1)

    def test_reset_starts_a_clean_session(self):
        counter = self.make_counter()
        counter.update([track(4, 35)])
        counter.update([track(4, 65)])
        counter.reset()

        self.assertEqual(counter.count_in, 0)
        self.assertEqual(counter.count_out, 0)
        self.assertEqual(counter.update([track(4, 65)]), [])

    def test_moving_line_clears_previous_track_side(self):
        counter = self.make_counter()
        counter.update([track(5, 35)])
        counter.set_line((0, 20, 100, 20), "down")

        self.assertEqual(counter.update([track(5, 35)]), [])
        self.assertEqual(counter.count_in, 0)


if __name__ == "__main__":
    unittest.main()
