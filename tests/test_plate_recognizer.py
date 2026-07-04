import unittest

from app.models_ai.plate_recognizer import select_best_plate


class PlateVotingTests(unittest.TestCase):
    def test_repeated_plate_wins_across_frames(self):
        result = select_best_plate([
            {"plate": "77C-135.58", "confidence": 0.81, "plate_img": "a"},
            {"plate": "77C13558", "confidence": 0.90, "plate_img": "b"},
            {"plate": "77C13556", "confidence": 0.98, "plate_img": "wrong"},
        ])

        self.assertEqual(result["plate"], "77C13558")
        self.assertEqual(result["votes"], 2)
        self.assertEqual(result["confidence"], 0.90)
        self.assertEqual(result["plate_img"], "b")

    def test_empty_candidates_return_none(self):
        self.assertIsNone(select_best_plate([]))
        self.assertIsNone(select_best_plate([{"plate": "", "confidence": 1}]))

    def test_invalid_vietnamese_plate_shape_is_rejected(self):
        self.assertIsNone(select_best_plate([
            {"plate": "58071A4", "confidence": 0.99, "plate_img": "noise"},
        ]))
        self.assertIsNone(select_best_plate([
            {"plate": "79H0272", "confidence": 0.99, "plate_img": "missing_digit"},
        ]))


if __name__ == "__main__":
    unittest.main()
