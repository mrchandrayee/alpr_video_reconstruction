import argparse, os, cv2
from pathlib import Path
from .config import load_config
from .pipeline import ALPRPipeline

def main():
    ap = argparse.ArgumentParser(description="ALPR vehicle + license plate recognition")
    ap.add_argument("--source", required=True, help="Video path, webcam index (0), or RTSP URL")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")
    pipeline = ALPRPipeline(cfg)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if cfg["output"].get("save_video", True):
        out = Path(cfg["output"]["path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            annotated, _ = pipeline.process(frame)
            if writer: writer.write(annotated)
            if cfg["output"].get("show_window", True):
                cv2.imshow("ALPR - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
    if writer: print(f"Saved: {cfg['output']['path']}")

if __name__ == "__main__":
    main()
