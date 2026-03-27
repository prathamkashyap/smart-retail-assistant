"""
zone_manager.py -- Shelf zone definitions and spatial logic.

Each "zone" is an axis-aligned rectangle representing a shelf region on
the camera frame.  The manager loads zone definitions from the config file
and provides helpers to:
  - check if a bounding box overlaps or is near a zone
  - count how many product detections fall inside each zone
  - draw zone overlays on the frame
"""

import cv2


class ShelfZone:
    """Data class for a single shelf zone."""

    def __init__(self, zone_id, label, bbox, color):
        self.zone_id = zone_id
        self.label = label
        # bbox stored as (x1, y1, x2, y2) in pixel coordinates
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.color = tuple(color)

    def contains_point(self, cx, cy):
        """Return True if the centroid (cx, cy) is inside this zone."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def overlaps_bbox(self, bbox):
        """
        Return True if the given [x1, y1, x2, y2] bounding box has
        any overlap with this zone.
        """
        bx1, by1, bx2, by2 = bbox
        return not (bx2 < self.x1 or bx1 > self.x2 or
                    by2 < self.y1 or by1 > self.y2)

    def is_near(self, bbox, margin):
        """
        Check if a bounding box is within 'margin' pixels of the zone.
        Used for proximity detection (e.g. customer standing near shelf).
        """
        bx1, by1, bx2, by2 = bbox
        expanded = (
            self.x1 - margin,
            self.y1 - margin,
            self.x2 + margin,
            self.y2 + margin
        )
        return not (bx2 < expanded[0] or bx1 > expanded[2] or
                    by2 < expanded[1] or by1 > expanded[3])


class ZoneManager:
    """
    Manages all shelf zones and provides batch spatial queries.
    """

    def __init__(self, config):
        self.zones = []
        self.proximity_margin = config.get("proximity_margin", 80)

        for z in config.get("shelf_zones", []):
            self.zones.append(ShelfZone(
                zone_id=z["zone_id"],
                label=z["label"],
                bbox=z["bbox"],
                color=z["color"],
            ))

        print(f"[zones] Loaded {len(self.zones)} shelf zones.")

    def get_zone_for_bbox(self, bbox):
        """
        Return the zone whose area contains the centroid of bbox,
        or None if the bbox is outside all zones.
        """
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        for zone in self.zones:
            if zone.contains_point(cx, cy):
                return zone
        return None

    def get_zones_near_bbox(self, bbox):
        """
        Return a list of zones that the bbox is near (within margin).
        Useful for person-proximity checks.
        """
        return [z for z in self.zones if z.is_near(bbox, self.proximity_margin)]

    def count_products_per_zone(self, product_detections):
        """
        Given a list of product detections, count how many fall inside
        each shelf zone.

        Returns a dict: zone_id -> count
        """
        counts = {z.zone_id: 0 for z in self.zones}
        for det in product_detections:
            zone = self.get_zone_for_bbox(det["bbox"])
            if zone is not None:
                counts[zone.zone_id] += 1
        return counts

    def draw_zones(self, frame, product_counts=None, zone_status=None):
        """
        Draw semi-transparent zone overlays on the frame with labels.

        product_counts : dict  zone_id -> int  (products in each zone)
        zone_status    : dict  zone_id -> str  (extra status text)
        """
        overlay = frame.copy()
        for zone in self.zones:
            # Semi-transparent fill
            cv2.rectangle(overlay,
                          (zone.x1, zone.y1),
                          (zone.x2, zone.y2),
                          zone.color, -1)

        # Blend overlay at 25% opacity
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        for zone in self.zones:
            # Zone border
            cv2.rectangle(frame,
                          (zone.x1, zone.y1),
                          (zone.x2, zone.y2),
                          zone.color, 2)

            # Label
            label = zone.label
            if product_counts and zone.zone_id in product_counts:
                label += f" | items: {product_counts[zone.zone_id]}"
            if zone_status and zone.zone_id in zone_status:
                label += f" | {zone_status[zone.zone_id]}"

            cv2.putText(frame, label,
                        (zone.x1 + 4, zone.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone.color, 1)

        return frame
