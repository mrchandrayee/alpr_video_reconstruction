from alpr.ocr import PlateOCR

def test_normalize_plate_text():
    assert PlateOCR.normalize("ka-01 AB 1234") == "KA01AB1234"

def test_normalize_removes_symbols():
    assert PlateOCR.normalize(" MH.12-XY@9876 ") == "MH12XY9876"
