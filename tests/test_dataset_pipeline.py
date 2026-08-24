"""
Dataset Pipeline, Format Converters & Validation Regression Tests.
Verifies VisDrone, UAVDT, and UA-DETRAC annotation converters,
YOLO normalization bounds, and dataset leakage detection.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from data.converters.visdrone_to_yolo import convert_visdrone_annotation, VISDRONE_TO_UNIFIED
from data.converters.uavdt_to_yolo import convert_uavdt_sequence, UAVDT_TO_UNIFIED
from data.converters.ua_detrac_to_yolo import convert_ua_detrac_xml
from data.validate_dataset import validate_yolo_label_file, check_data_leakage, validate_entire_dataset
from config.settings import BASE_DIR


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Creates a temporary workspace for converter testing."""
    test_dir = tmp_path / "dataset_test"
    test_dir.mkdir()
    return test_dir


def test_visdrone_annotation_converter(temp_dataset_dir):
    """Verifies that VisDrone annotations convert accurately to normalized YOLO format."""
    img_path = temp_dataset_dir / "test_frame.jpg"
    ann_path = temp_dataset_dir / "test_frame.txt"
    out_lbl = temp_dataset_dir / "out_yolo.txt"

    # Create 1000x500 test image
    img = Image.fromarray(np.zeros((500, 1000, 3), dtype=np.uint8))
    img.save(img_path)

    # VisDrone format: x,y,w,h,score,category,truncation,occlusion
    ann_content = """100,50,200,100,1,4,0,0
300,100,50,80,1,1,0,0
500,200,60,70,1,3,0,0
700,300,100,100,0,4,0,0
"""
    ann_path.write_text(ann_content, encoding="utf-8")

    count = convert_visdrone_annotation(ann_path, img_path, out_lbl)
    assert count == 3, "Expected 3 valid annotations (4th line has score=0 and should be filtered)"

    lines = out_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    # Check vehicle (cat 4 -> class 0)
    c0, xc0, yc0, w0, h0 = map(float, lines[0].split())
    assert int(c0) == 0
    assert pytest.approx(xc0, rel=1e-3) == (100 + 100) / 1000.0  # 0.20
    assert pytest.approx(yc0, rel=1e-3) == (50 + 50) / 500.0     # 0.20
    assert pytest.approx(w0, rel=1e-3) == 200 / 1000.0           # 0.20
    assert pytest.approx(h0, rel=1e-3) == 100 / 500.0            # 0.20

    # Check pedestrian (cat 1 -> class 1)
    c1, _, _, _, _ = map(float, lines[1].split())
    assert int(c1) == 1

    # Check cyclist (cat 3 -> class 2)
    c2, _, _, _, _ = map(float, lines[2].split())
    assert int(c2) == 2


def test_uavdt_sequence_converter(temp_dataset_dir):
    """Verifies that UAVDT sequence tracking annotations convert correctly."""
    seq_dir = temp_dataset_dir / "M0101"
    seq_dir.mkdir()
    gt_file = temp_dataset_dir / "M0101_gt_whole.txt"
    out_lbl_dir = temp_dataset_dir / "uavdt_labels"

    img_path = seq_dir / "img000001.jpg"
    img = Image.fromarray(np.zeros((600, 800, 3), dtype=np.uint8))
    img.save(img_path)

    # UAVDT format: frame_idx,target_id,bbox_left,bbox_top,bbox_width,bbox_height,out_of_view,occlusion,category
    gt_content = """1,1,100,120,80,60,0,0,1
1,2,200,240,120,90,0,0,2
1,3,400,300,50,50,1,0,1
"""
    gt_file.write_text(gt_content, encoding="utf-8")

    convert_uavdt_sequence(gt_file, seq_dir, out_lbl_dir, "M0101")

    out_file = out_lbl_dir / "M0101_img000001.txt"
    assert out_file.exists(), "Converted UAVDT label file must exist"

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2, "Expected 2 visible targets (3rd target out_of_view=1 must be filtered)"

    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        assert cls_id == 0, "UAVDT cars/trucks must map to vehicle (class 0)"
        xc, yc, w, h = map(float, parts[1:])
        assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_ua_detrac_xml_converter(temp_dataset_dir):
    """Verifies that UA-DETRAC XML annotations parse and output valid YOLO boxes."""
    seq_dir = temp_dataset_dir / "MVI_20011"
    seq_dir.mkdir()
    xml_path = temp_dataset_dir / "MVI_20011.xml"
    out_lbl_dir = temp_dataset_dir / "detrac_labels"

    img_path = seq_dir / "img00001.jpg"
    img = Image.fromarray(np.zeros((540, 960, 3), dtype=np.uint8))
    img.save(img_path)

    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_20011">
    <sequence_attribute camera_state="unstable"/>
    <frame density="1" num="1">
        <target_list>
            <target id="1">
                <box height="80.0" left="200.0" top="150.0" width="120.0"/>
                <attribute vehicle_type="car"/>
            </target>
        </target_list>
    </frame>
</sequence>
"""
    xml_path.write_text(xml_content, encoding="utf-8")

    convert_ua_detrac_xml(xml_path, seq_dir, out_lbl_dir, "MVI_20011")

    out_file = out_lbl_dir / "MVI_20011_img00001.txt"
    assert out_file.exists(), "Converted UA-DETRAC label must exist"

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    cls_id, xc, yc, w, h = map(float, lines[0].split())
    assert int(cls_id) == 0, "UA-DETRAC vehicles must map to class 0"
    assert pytest.approx(xc, rel=1e-3) == (200.0 + 60.0) / 960.0
    assert pytest.approx(yc, rel=1e-3) == (150.0 + 40.0) / 540.0


def test_yolo_label_validator_logic(temp_dataset_dir):
    """Verifies the YOLO label syntax validator detects malformed or out-of-bounds annotations."""
    valid_lbl = temp_dataset_dir / "valid.txt"
    valid_lbl.write_text("0 0.5 0.5 0.2 0.3\n1 0.1 0.2 0.05 0.08\n", encoding="utf-8")
    is_valid, errs, counts = validate_yolo_label_file(valid_lbl)
    assert is_valid is True
    assert len(errs) == 0
    assert counts[0] == 1 and counts[1] == 1

    # Invalid class ID
    bad_cls_lbl = temp_dataset_dir / "bad_cls.txt"
    bad_cls_lbl.write_text("9 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    is_valid, errs, _ = validate_yolo_label_file(bad_cls_lbl)
    assert is_valid is False
    assert any("Invalid class_id" in e for e in errs)

    # Coordinate out of bounds (> 1.0)
    bad_coord_lbl = temp_dataset_dir / "bad_coord.txt"
    bad_coord_lbl.write_text("0 1.5 0.5 0.2 0.3\n", encoding="utf-8")
    is_valid, errs, _ = validate_yolo_label_file(bad_coord_lbl)
    assert is_valid is False
    assert any("out of [0.0, 1.0]" in e for e in errs)


def test_dataset_manifest_and_leakage_check():
    """Verifies that the master dataset validation passes and reports zero exact image leakage."""
    manifest = validate_entire_dataset()
    assert manifest["validation_passed"] is True, "Dataset validation must pass"
    assert manifest["split_counts"]["train"] > 0, "Train split must have images"
    assert manifest["split_counts"]["val"] > 0, "Val split must have images"
    assert manifest["split_counts"]["test"] > 0, "Test split must have images"
    assert manifest["leakage_analysis"]["exact_image_overlap_train_test"] == [], "Train and test splits must not overlap exact images"
