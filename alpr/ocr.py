import re

class PlateOCR:
    def __init__(self, languages=None, gpu=False):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("Install OCR dependencies: pip install -r requirements.txt") from exc
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    @staticmethod
    def normalize(text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def read(self, image):
        if image is None or image.size == 0:
            return []
        results = self.reader.readtext(image, detail=1, paragraph=False)
        found = []
        for _, text, confidence in results:
            clean = self.normalize(text)
            if len(clean) >= 4:
                found.append({"text": clean, "confidence": float(confidence)})
        return sorted(found, key=lambda x: x["confidence"], reverse=True)
