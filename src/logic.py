"""
logic.py -- Interaction detection, dwell-time estimation, and anomaly logic.

This is the central "brain" of the system. Each frame it receives the
current set of person tracks, product tracks, and zone product counts,
and it determines:
  - Which people are interacting with which shelf zones
  - How long each person has been near a zone (dwell time)
  - Whether products have been removed from a zone
  - Whether a person has repeatedly visited the same zone
  - Simple anomaly flags (sudden stock drop, prolonged empty zone,
    missing product after customer interaction)
"""

import time
from collections import deque


class InteractionTracker:
    """
    Keeps per-person, per-zone state to detect interactions,
    dwell time, repeated visits, and product-related anomalies.
    """

    # Temporal smoothing parameters
    HISTORY_LENGTH = 5          # frames of product-count history to keep
    CONFIRM_FRAMES = 3          # consecutive low-count frames to confirm removal

    def __init__(self, config):
        self.dwell_threshold = config.get("dwell_time_threshold_sec", 3.0)
        self.empty_alert_sec = config.get("empty_zone_alert_sec", 10.0)
        self.repeat_threshold = config.get("repeated_attention_threshold", 3)
        self.stock_drop_ratio = config.get("stock_drop_ratio", 0.5)

        # Tracks when each person first entered each zone's proximity
        # Key: (person_track_id, zone_id) -> entry_timestamp
        self.active_interactions = {}

        # Counts how many times each person has visited each zone
        # Key: (person_track_id, zone_id) -> visit_count
        self.visit_counts = {}

        # Previous frame product counts per zone for change detection
        self.prev_product_counts = {}

        # Temporal smoothing: per-zone product count history (last N frames)
        # Key: zone_id -> deque of recent counts
        self.count_history = {}

        # Tracks when a zone first became empty
        # Key: zone_id -> timestamp when it became empty
        self.empty_since = {}

        # Baseline product counts (set from the first few frames)
        self.baseline_counts = {}
        self.baseline_set = False
        self.warmup_frames = 10
        self.frame_count = 0

    def update(self, person_tracks, product_detections, zone_manager, logger):
        """
        Main per-frame update. Analyses interactions and anomalies.

        Parameters
        ----------
        person_tracks : dict
            track_id -> detection dict, for persons only.
        product_detections : list
            List of product detection dicts (non-person).
        zone_manager : ZoneManager
            Provides zone lookup and product counting.
        logger : EventLogger
            For recording events.

        Returns
        -------
        dict
            Status information used for on-screen display:
            - active_dwells: list of (person_id, zone_id, seconds)
            - zone_status: dict zone_id -> status string
            - events: list of event descriptions generated this frame
        """
        now = time.time()
        self.frame_count += 1
        events = []
        active_dwells = []
        zone_status = {}

        # Count products per zone this frame
        current_counts = zone_manager.count_products_per_zone(product_detections)

        # Set baseline from early frames (gives the model time to stabilize)
        if not self.baseline_set and self.frame_count == self.warmup_frames:
            self.baseline_counts = dict(current_counts)
            self.baseline_set = True
            print("[logic] Baseline product counts established:",
                  self.baseline_counts)

        # --- Person-zone proximity and dwell time ---
        currently_active = set()

        for pid, pdet in person_tracks.items():
            nearby_zones = zone_manager.get_zones_near_bbox(pdet["bbox"])
            for zone in nearby_zones:
                key = (pid, zone.zone_id)
                currently_active.add(key)

                if key not in self.active_interactions:
                    # Person just entered this zone's proximity
                    self.active_interactions[key] = now
                    self.visit_counts[key] = self.visit_counts.get(key, 0) + 1

                    event_msg = (f"Person #{pid} entered proximity of "
                                 f"{zone.zone_id}")
                    events.append(event_msg)
                    logger.log_event(
                        event_type="INTERACTION_START",
                        zone_id=zone.zone_id,
                        object_id=pid,
                        confidence=pdet.get("confidence", 0),
                        details=event_msg,
                    )

                    # Check repeated attention
                    if self.visit_counts[key] >= self.repeat_threshold:
                        rep_msg = (f"Repeated attention: Person #{pid} has "
                                   f"visited {zone.zone_id} "
                                   f"{self.visit_counts[key]} times")
                        events.append(rep_msg)
                        logger.log_event(
                            event_type="REPEATED_ATTENTION",
                            zone_id=zone.zone_id,
                            object_id=pid,
                            confidence=pdet.get("confidence", 0),
                            details=rep_msg,
                        )
                else:
                    # Person still near the zone -- compute dwell
                    dwell = now - self.active_interactions[key]
                    active_dwells.append((pid, zone.zone_id, round(dwell, 1)))

                    # Fire dwell alert once when threshold is first crossed
                    if (dwell >= self.dwell_threshold and
                            dwell < self.dwell_threshold + 0.5):
                        dwell_msg = (f"Dwell alert: Person #{pid} has been "
                                     f"near {zone.zone_id} for "
                                     f"{dwell:.1f}s")
                        events.append(dwell_msg)
                        logger.log_event(
                            event_type="DWELL_ALERT",
                            zone_id=zone.zone_id,
                            object_id=pid,
                            confidence=pdet.get("confidence", 0),
                            details=dwell_msg,
                        )

        # Clean up interactions that ended (person left zone)
        ended = set(self.active_interactions.keys()) - currently_active
        for key in ended:
            pid, zid = key
            dwell = now - self.active_interactions[key]
            end_msg = (f"Person #{pid} left {zid} after {dwell:.1f}s")
            events.append(end_msg)
            logger.log_event(
                event_type="INTERACTION_END",
                zone_id=zid,
                object_id=pid,
                confidence=0,
                details=end_msg,
            )
            del self.active_interactions[key]

        # --- Product disappearance and stock changes (temporally smoothed) ---
        if self.baseline_set:
            for zone in zone_manager.zones:
                zid = zone.zone_id
                curr = current_counts.get(zid, 0)
                baseline = self.baseline_counts.get(zid, 0)

                # Update per-zone count history buffer
                if zid not in self.count_history:
                    self.count_history[zid] = deque(maxlen=self.HISTORY_LENGTH)
                self.count_history[zid].append(curr)

                # Smoothed count = minimum over the history window
                # This prevents single-frame flickers from triggering events
                hist = self.count_history[zid]
                smoothed = min(hist) if len(hist) >= self.CONFIRM_FRAMES else curr
                prev_smoothed = self.prev_product_counts.get(zid, smoothed)

                # Product removed -- confirmed only when smoothed count drops
                if smoothed < prev_smoothed:
                    rem_msg = (f"Product removal in {zid}: "
                               f"count dropped {prev_smoothed} -> {smoothed}")
                    events.append(rem_msg)
                    logger.log_event(
                        event_type="PRODUCT_REMOVED",
                        zone_id=zid,
                        object_id=-1,
                        confidence=0,
                        details=rem_msg,
                    )

                # Sudden stock drop relative to baseline
                if baseline > 0 and smoothed / baseline < self.stock_drop_ratio:
                    stock_msg = (f"Stock drop anomaly in {zid}: "
                                 f"baseline={baseline}, current={smoothed}")
                    zone_status[zid] = "LOW STOCK"
                    if prev_smoothed / baseline >= self.stock_drop_ratio:
                        events.append(stock_msg)
                        logger.log_event(
                            event_type="ANOMALY_STOCK_DROP",
                            zone_id=zid,
                            object_id=-1,
                            confidence=0,
                            details=stock_msg,
                        )

                # Empty zone tracking (uses raw count for responsiveness)
                if curr == 0:
                    if zid not in self.empty_since:
                        self.empty_since[zid] = now
                    empty_duration = now - self.empty_since[zid]
                    if empty_duration >= self.empty_alert_sec:
                        zone_status[zid] = f"EMPTY {empty_duration:.0f}s"
                        if empty_duration < self.empty_alert_sec + 0.5:
                            empty_msg = (f"Prolonged empty zone: {zid} has "
                                         f"been empty for "
                                         f"{empty_duration:.0f}s")
                            events.append(empty_msg)
                            logger.log_event(
                                event_type="ANOMALY_EMPTY_ZONE",
                                zone_id=zid,
                                object_id=-1,
                                confidence=0,
                                details=empty_msg,
                            )
                    else:
                        zone_status[zid] = "EMPTY"
                else:
                    if zid in self.empty_since:
                        del self.empty_since[zid]
                    if curr > 0:
                        zone_status[zid] = "OK"

                # Store smoothed count for next frame comparison
                self.prev_product_counts[zid] = smoothed

        return {
            "active_dwells": active_dwells,
            "zone_status": zone_status,
            "events": events,
            "product_counts": current_counts,
        }
