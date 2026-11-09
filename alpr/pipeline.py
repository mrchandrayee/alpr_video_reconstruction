from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from .ocr import PlateOCR

class ALPRPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        v = cfg["vehicle"]
        self.vehicle_model = YOLO(v["model"])
        self.plate_model = None
        p = cfg["plate"]
        if p.get("enabled", True):
            weight = Path(p["model"])
            if not weight.exists():
                raise FileNotFoundError(
                    f"Plate model not found: {weight}. Add a license-plate YOLO weights file "
                    "trained for your region, or set plate.enabled=false."
                )
            self.plate_model = YOLO(str(weight))
        o = cfg["ocr"]
        self.ocr = PlateOCR(o.get("languages", ["en"]), o.get("gpu", False))
        self.last_seen = {}

    def _detect_plates(self, frame):
        if self.plate_model is None:
            return []
        result = self.plate_model.predict(
            frame, conf=self.cfg["plate"]["confidence"], verbose=False
        )[0]
        items = []
        if result.boxes is None:
            return items
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            # Upscale small plates before OCR.
            if crop.size:
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            readings = self.ocr.read(crop)
            best = readings[0] if readings else {"text": "READING...", "confidence": 0.0}
            items.append({"box": (x1, y1, x2, y2), **best})
        return items

    def process(self, frame):
        cfg = self.cfg
        result = self.vehicle_model.track(
            frame,
            persist=cfg["tracking"].get("persist", True),
            tracker=cfg["vehicle"].get("tracker", "bytetrack.yaml"),
            conf=cfg["vehicle"]["confidence"],
            classes=cfg["vehicle"].get("classes"),
            verbose=False
        )[0]
        annotated = frame.copy()
        plates = self._detect_plates(frame)
        # Associate each detected plate to the nearest vehicle box by center distance.
        vehicles = []
        if result.boxes is not None:
            for b in result.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                tid = int(b.id[0]) if b.id is not None else -1
                vehicles.append({"id": tid, "box": (x1,y1,x2,y2)})
                if cfg["display"].get("show_vehicle_boxes", True):
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,180,255), 2)
                    cv2.putText(annotated, f"ID {tid}", (x1,max(20,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0,180,255), 2)
        for plate in plates:
            x1,y1,x2,y2 = plate["box"]
            pcx, pcy = (x1+x2)//2, (y1+y2)//2
            owner = None
            best_dist = float("inf")
            for v in vehicles:
                a,b,c,d = v["box"]
                if a <= pcx <= c and b <= pcy <= d:
                    dist = ((pcx-(a+c)//2)**2 + (pcy-(b+d)//2)**2)**.5
                    if dist < best_dist:
                        owner, best_dist = v["id"], dist
            plate["vehicle_id"] = owner
            if plate["text"] not in ("", "READING..."):
                self.last_seen[plate["text"]] = plate["confidence"]
            if cfg["display"].get("show_plate_boxes", True):
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,255),2)
                cv2.putText(annotated, f'{plate["text"]} {plate["confidence"]:.0%}',
                            (x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),2)
        if cfg["display"].get("show_line", True):
            ly = int(frame.shape[0] * cfg["tracking"].get("line_y_ratio", .55))
            cv2.line(annotated,(0,ly),(frame.shape[1],ly),(255,100,0),2)
        self._draw_panel(annotated, plates)
        return annotated, plates

    def _draw_panel(self, frame, plates):
        title = self.cfg["display"].get("panel_title", "Plates")
        x0, y0, width = max(8, frame.shape[1]-190), 12, 180
        rows = min(6, len(plates))
        height = 34 + rows*42
        overlay = frame.copy()
        cv2.rectangle(overlay,(x0,y0),(x0+width,y0+height),(15,15,15),-1)
        cv2.addWeighted(overlay,.78,frame,.22,0,frame)
        cv2.putText(frame,title,(x0+8,y0+22),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2)
        for i,p in enumerate(plates[:6]):
            y = y0+48+i*42
            cv2.putText(frame,p["text"][:18],(x0+8,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
            cv2.putText(frame,f'{p["confidence"]:.0%}',(x0+8,y+15),
                        cv2.FONT_HERSHEY_SIMPLEX,.35,(220,220,220),1)
