"""
main.py -- Entry point for the Retail Shelf Intelligence system.

Ties together detection, tracking, zone management, interaction logic,
event logging, and analytics into a single real-time pipeline that
processes frames from a webcam or video file.

Usage:
    python src/main.py                            # webcam (device 0)
    python src/main.py --source path/to/video.mp4 # video file
    python src/main.py --config configs/custom.json
"""

import argparse
import json
import os
import sys
import time

import cv2

# Resolve project root so imports work regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.detect import ObjectDetector, draw_detections
from src.track import CentroidTracker
from src.zone_manager import ZoneManager
from src.logic import InteractionTracker
from src.logger import EventLogger
from src.analytics import generate_summary


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retail Shelf Intelligence System — Real-time shelf "
                    "monitoring with YOLOv8 detection and interaction tracking."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source. Use an integer for a webcam index (e.g. 0) "
             "or a file path for recorded video (e.g. data/demo_video.mp4). "
             "Defaults to the value in the config file (usually webcam 0)."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "default_config.json"),
        help="Path to the JSON configuration file "
             "(default: configs/default_config.json)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # Determine video source
    source = args.source if args.source is not None else config.get("video_source", 0)
    try:
        source = int(source)  # webcam index
    except (ValueError, TypeError):
        # Treat as file path -- validate existence
        if not os.path.isabs(source):
            source = os.path.join(PROJECT_ROOT, source)
        if not os.path.exists(source):
            print(f"[main] ERROR: Video file not found: {source}")
            print("[main] Please check the path and try again.")
            print("[main] Example: python src/main.py --source data/demo_video.mp4")
            sys.exit(1)

    # Initialize modules
    detector = ObjectDetector(config)
    person_tracker = CentroidTracker(
        max_disappeared=config.get("tracker_max_disappeared", 25),
        max_distance=config.get("tracker_max_distance", 120),
    )
    product_tracker = CentroidTracker(
        max_disappeared=config.get("tracker_max_disappeared", 25),
        max_distance=config.get("tracker_max_distance", 120),
    )
    zone_mgr = ZoneManager(config)
    interaction = InteractionTracker(config)

    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    event_logger = EventLogger(
        output_dir=output_dir,
        snapshot_enabled=config.get("snapshot_on_events", True),
        debounce_sec=config.get("log_debounce_sec", 2.0),
    )

    # Ensure screenshots directory exists
    screenshots_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    # Open video
    print(f"[main] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[main] ERROR: Could not open video source.")
        if isinstance(source, int):
            print(f"[main] Webcam index {source} is not available. "
                  "Check that your camera is connected.")
        else:
            print(f"[main] Could not decode: {source}")
        sys.exit(1)

    display_w = config.get("display_width", 960)
    display_h = config.get("display_height", 540)
    fps_display = 0.0
    frame_idx = 0

    print("[main] System running. Press 'q' to quit.\n")

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("[main] End of video stream.")
                break

            frame_idx += 1

            # Resize for consistent processing
            frame = cv2.resize(frame, (display_w, display_h))

            # --- Detection ---
            detections = detector.detect(frame)

            # Split persons from products
            person_dets = [d for d in detections if d["is_person"]]
            product_dets = [d for d in detections if not d["is_person"]]

            # --- Tracking ---
            person_tracks = person_tracker.update(person_dets)
            product_tracks = product_tracker.update(product_dets)

            # --- Interaction and anomaly logic ---
            status = interaction.update(
                person_tracks=person_tracks,
                product_detections=product_dets,
                zone_manager=zone_mgr,
                logger=event_logger,
            )

            # --- Drawing ---
            # Draw shelf zone overlays
            frame = zone_mgr.draw_zones(
                frame,
                product_counts=status["product_counts"],
                zone_status=status.get("zone_status"),
            )

            # Draw detections (bboxes and labels)
            frame = draw_detections(frame, detections)

            # Draw track IDs on persons
            for tid, pdet in person_tracks.items():
                x1, y1, x2, y2 = pdet["bbox"]
                cv2.putText(frame, f"ID:{tid}",
                            (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 120, 0), 1, cv2.LINE_AA)

            # Draw dwell time labels (offset vertically to prevent overlap)
            dwell_offset = {}
            for pid, zid, secs in status.get("active_dwells", []):
                if pid in person_tracks:
                    x1, y1, _, _ = person_tracks[pid]["bbox"]
                    # Stack labels vertically if multiple dwells for same person
                    offset = dwell_offset.get(pid, 0)
                    dwell_offset[pid] = offset + 1
                    y_pos = y1 - 10 - (offset * 18)
                    cv2.putText(frame, f"dwell:{secs}s @{zid}",
                                (x1, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 200, 255), 1, cv2.LINE_AA)

            # Print events to console (limit to avoid flooding)
            for evt in status.get("events", [])[:5]:
                print(f"  >> {evt}")

            # --- HUD: semi-transparent banner at top ---
            hud_h = 30
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_w, hud_h),
                          (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # FPS counter
            t_elapsed = time.time() - t_start
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_elapsed, 1e-6))
            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Event counter
            cv2.putText(frame, f"Events: {event_logger.event_count}",
                        (150, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # Frame counter
            cv2.putText(frame, f"Frame: {frame_idx}",
                        (350, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Retail Shelf Intelligence", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[main] Quit signal received.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        event_logger.finalize()

        # Generate post-run analytics
        log_path = os.path.join(output_dir, "logs.csv")
        summary_path = os.path.join(output_dir, "analytics_summary.csv")
        generate_summary(log_path=log_path, output_path=summary_path)

        print("[main] Session complete.")


if __name__ == "__main__":
    main()
