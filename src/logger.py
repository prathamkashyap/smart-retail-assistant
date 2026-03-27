"""
logger.py -- Event logging to CSV using Pandas.

All significant events (interactions, removals, anomalies) are appended
to a CSV file in real time.  The logger also supports saving snapshot
images when important events occur.

Includes a debounce mechanism to suppress duplicate events (same type,
zone, and object) that fire within a short time window.
"""

import os
import time
from datetime import datetime
import pandas as pd
import cv2


class EventLogger:
    """
    Appends event rows to a CSV file and optionally saves frame snapshots
    for notable events.  Duplicate events are suppressed via debounce.
    """

    COLUMNS = [
        "timestamp",
        "event_type",
        "zone_id",
        "object_id",
        "confidence",
        "details",
    ]

    # These event types will trigger a snapshot save when enabled
    SNAPSHOT_EVENTS = {
        "PRODUCT_REMOVED",
        "ANOMALY_STOCK_DROP",
        "ANOMALY_EMPTY_ZONE",
        "REPEATED_ATTENTION",
    }

    def __init__(self, output_dir="outputs", snapshot_enabled=True,
                 debounce_sec=2.0):
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "logs.csv")
        self.snapshot_dir = os.path.join(output_dir, "snapshots")
        self.snapshot_enabled = snapshot_enabled

        os.makedirs(output_dir, exist_ok=True)
        if self.snapshot_enabled:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Buffer rows and flush periodically to reduce disk I/O
        self.buffer = []
        self.flush_interval = 10  # flush every N events
        self.event_count = 0

        # Debounce: suppress same event within this many seconds
        self.debounce_sec = debounce_sec
        # Key: (event_type, zone_id, object_id) -> last_timestamp (float)
        self._recent_events = {}

        # Write header if the log file does not exist yet
        if not os.path.exists(self.log_path):
            pd.DataFrame(columns=self.COLUMNS).to_csv(
                self.log_path, index=False
            )
            print(f"[logger] Created log file: {self.log_path}")

    def log_event(self, event_type, zone_id, object_id,
                  confidence=0, details="", frame=None):
        """
        Record a single event. If a frame is provided and the event is
        notable, a snapshot of the frame is saved alongside the log entry.

        Duplicate events (same type, zone, object) are suppressed if
        they occur within ``debounce_sec`` seconds of each other.
        """
        now = time.time()

        # --- Debounce check ---
        dedup_key = (event_type, zone_id, object_id)
        last_time = self._recent_events.get(dedup_key)
        if last_time is not None and (now - last_time) < self.debounce_sec:
            return  # suppress duplicate
        self._recent_events[dedup_key] = now

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp": timestamp,
            "event_type": event_type,
            "zone_id": zone_id,
            "object_id": object_id,
            "confidence": round(confidence, 3),
            "details": details,
        }

        self.buffer.append(row)
        self.event_count += 1

        # Periodic flush
        if len(self.buffer) >= self.flush_interval:
            self._flush()

        # Optional snapshot
        if (self.snapshot_enabled and frame is not None and
                event_type in self.SNAPSHOT_EVENTS):
            self._save_snapshot(frame, event_type)

    def _flush(self):
        """Write buffered rows to the CSV file."""
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer, columns=self.COLUMNS)
        df.to_csv(self.log_path, mode="a", header=False, index=False)
        self.buffer = []

    def _save_snapshot(self, frame, event_type):
        """Save the current frame as a JPEG for post-run review."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{event_type}_{ts}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(path, frame)

    def finalize(self):
        """Flush any remaining buffered events -- call on shutdown."""
        self._flush()
        print(f"[logger] Finalized. {self.event_count} events written "
              f"to {self.log_path}")
