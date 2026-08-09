from ocr_asr_benchmark.utils.geometry import bbox_iou, polygon_to_xyxy


def test_polygon_to_bbox() -> None:
    assert polygon_to_xyxy([[2, 3], [8, 3], [8, 9], [2, 9]]) == [2.0, 3.0, 8.0, 9.0]


def test_iou() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 2, 2], [3, 3, 5, 5]) == 0.0
