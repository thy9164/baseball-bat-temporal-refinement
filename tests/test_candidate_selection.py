import copy
import math
import unittest

from yolo.candidate_selection import SelectionConfig, select_sequence_candidates
from yolo.export_detector_evaluation_rows import select_for_evaluation_records


CONFIG = SelectionConfig(
    width=1000.0,
    height=500.0,
    bbox_conf_weight=100.0,
    anchor_radius=300.0,
    anchor_weight=1.0,
)


def detection(rank, bbox_conf, head, tail, bbox=(0.5, 0.5, 0.2, 0.3)):
    return {
        "bbox_conf": bbox_conf,
        "bbox": bbox,
        "head": (*head, 0.9),
        "tail": (*tail, 0.9),
        "detection_rank": rank,
    }


def records_fixture():
    records = []
    for frame in range(1, 7):
        stable_x = 0.30 + frame * 0.01
        records.append(
            {
                "prediction_stem": f"swing_001_{frame:06d}",
                "image_stem": f"swing_001_{frame:06d}",
                "swing_id": "swing_001",
                "frame_no": frame,
                "frame_name": f"{frame:06d}.jpg",
                "detections": [
                    detection(1, 0.95, (0.70, 0.40), (0.82, 0.42)),
                    detection(2, 0.80, (stable_x, 0.40), (stable_x + 0.12, 0.42)),
                ],
            }
        )
    return records


def legacy_point(item, keypoint):
    x, y, _ = item[keypoint]
    return float(x) * CONFIG.width, float(y) * CONFIG.height


def legacy_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def legacy_direction(item):
    head, tail = legacy_point(item, "head"), legacy_point(item, "tail")
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    return None if dx == 0 and dy == 0 else math.degrees(math.atan2(dy, dx))


def legacy_angle_delta(a, b):
    if a is None or b is None:
        return 0.0
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def legacy_score(item, previous, frame_gap, anchor):
    bbox_center = (float(item["bbox"][0]) * CONFIG.width, float(item["bbox"][1]) * CONFIG.height)
    score = -CONFIG.bbox_conf_weight * float(item["bbox_conf"])
    if previous is not None:
        dt = max(1, frame_gap)
        score += legacy_distance(legacy_point(item, "tail"), legacy_point(previous, "tail")) / dt
        score += legacy_distance(legacy_point(item, "head"), legacy_point(previous, "head")) / dt
        score += abs(
            legacy_distance(legacy_point(item, "head"), legacy_point(item, "tail"))
            - legacy_distance(legacy_point(previous, "head"), legacy_point(previous, "tail"))
        )
        score += legacy_angle_delta(legacy_direction(item), legacy_direction(previous))
        previous_center = (
            float(previous["bbox"][0]) * CONFIG.width,
            float(previous["bbox"][1]) * CONFIG.height,
        )
        score += legacy_distance(bbox_center, previous_center) / dt
    score += CONFIG.anchor_weight * max(0.0, legacy_distance(bbox_center, anchor) - CONFIG.anchor_radius)
    return score


def legacy_rank_sequence(records):
    centers = [
        (
            float(record["detections"][0]["bbox"][0]) * CONFIG.width,
            float(record["detections"][0]["bbox"][1]) * CONFIG.height,
        )
        for record in records
    ]
    anchor = (
        sorted(center[0] for center in centers)[len(centers) // 2 - 1 : len(centers) // 2 + 1][0]
        if len(centers) % 2
        else sum(sorted(center[0] for center in centers)[len(centers) // 2 - 1 : len(centers) // 2 + 1]) / 2,
        sum(sorted(center[1] for center in centers)[len(centers) // 2 - 1 : len(centers) // 2 + 1]) / 2,
    )
    previous = None
    previous_frame = None
    ranks = []
    for record in records:
        gap = record["frame_no"] - previous_frame if previous_frame is not None else 1
        valid = [
            item for item in record["detections"]
            if float(item["head"][2]) >= 0.1 and float(item["tail"][2]) >= 0.1
        ]
        selected = min(
            valid,
            key=lambda item: (legacy_score(item, previous, gap, anchor), int(item["detection_rank"])),
        )
        ranks.append(int(selected["detection_rank"]))
        previous, previous_frame = selected, record["frame_no"]
    return ranks


def selection_summary(items):
    return [
        (
            item["image_stem"],
            None if item["selected_pred"] is None else int(item["selected_pred"]["detection_rank"]),
            item.get("drop_bbox_reason"),
            item.get("drop_keypoints_reason"),
        )
        for item in items
    ]


class CandidateSelectionTests(unittest.TestCase):
    def test_extracted_selector_matches_legacy_scoring_path(self):
        records = records_fixture()
        expected = legacy_rank_sequence(copy.deepcopy(records))
        actual = select_sequence_candidates(copy.deepcopy(records), CONFIG)
        self.assertEqual([int(item["selected_pred"]["detection_rank"]) for item in actual], expected)

    def test_evaluation_ground_truth_fields_cannot_change_selection(self):
        base = records_fixture()
        evaluation_records = []
        for record in base:
            evaluation_records.append(
                {
                    **record,
                    "gt_row": {
                        "swing_id": record["swing_id"],
                        "frame": record["frame_name"],
                        "head_x": "1",
                        "head_y": "2",
                        "tail_x": "3",
                        "tail_y": "4",
                        "tail_visibility": "fully_visible",
                        "status": "matched",
                        "error_px": "0",
                    },
                    "clip_meta": {"view": "side"},
                }
            )
        changed = copy.deepcopy(evaluation_records)
        for record in changed:
            record["gt_row"].update(
                {
                    "head_x": "9999",
                    "head_y": "-9999",
                    "tail_x": "nan",
                    "tail_y": "inf",
                    "tail_visibility": "ignore",
                    "status": "missing_prediction",
                    "error_px": "1000000",
                }
            )
        first = select_for_evaluation_records(copy.deepcopy(evaluation_records), CONFIG)
        second = select_for_evaluation_records(copy.deepcopy(changed), CONFIG)
        self.assertEqual(selection_summary(first), selection_summary(second))

    def test_selector_is_deterministic_and_rejects_evaluation_fields(self):
        records = records_fixture()
        first = selection_summary(select_sequence_candidates(copy.deepcopy(records), CONFIG))
        second = selection_summary(select_sequence_candidates(copy.deepcopy(records), CONFIG))
        self.assertEqual(first, second)
        contaminated = copy.deepcopy(records)
        contaminated[0]["visibility"] = "fully_occluded"
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            select_sequence_candidates(contaminated, CONFIG)

    def test_local_length_outlier_keeps_historical_drop_behavior(self):
        records = []
        for frame in range(1, 8):
            tail_x = 0.31 if frame == 4 else 0.42
            records.append(
                {
                    "prediction_stem": f"swing_002_{frame:06d}",
                    "image_stem": f"swing_002_{frame:06d}",
                    "swing_id": "swing_002",
                    "frame_no": frame,
                    "frame_name": f"{frame:06d}.jpg",
                    "detections": [detection(1, 0.9, (0.30, 0.40), (tail_x, 0.42))],
                }
            )
        selected = select_sequence_candidates(records, CONFIG)
        self.assertEqual(selected[3].get("drop_keypoints_reason"), "length_outlier")
        self.assertNotIn("drop_bbox_reason", selected[3])


if __name__ == "__main__":
    unittest.main()
