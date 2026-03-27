"""
track.py -- Centroid-based object tracker with velocity prediction.

Maintains persistent IDs for detected objects across consecutive frames.
Uses Euclidean distance between centroids (with optional velocity-based
prediction) to match new detections to existing tracks.  When an object
is not matched for several frames it gets deregistered automatically.

Velocity prediction helps maintain IDs through brief occlusions or
detection gaps by estimating where each track *should* be in the next
frame based on its recent motion.
"""

import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict


class CentroidTracker:
    """
    Track objects across frames using their bounding-box centroids.

    Parameters
    ----------
    max_disappeared : int
        Number of consecutive frames an object can be missing before
        the track is dropped.
    max_distance : float
        Maximum pixel distance for a centroid match.  If the closest
        candidate is farther than this the detection is treated as new.
    velocity_smoothing : float
        Exponential moving average factor for velocity updates.
        0.0 = no smoothing (use latest), 1.0 = never update.
        Default 0.4 provides a good balance between responsiveness and
        stability.
    """

    def __init__(self, max_disappeared=25, max_distance=120,
                 velocity_smoothing=0.4):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()        # id -> [x1, y1, x2, y2]
        self.disappeared = OrderedDict()   # id -> frames missed
        self.velocities = OrderedDict()    # id -> (vx, vy) per-frame velocity
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.velocity_smoothing = velocity_smoothing

    def register(self, centroid, bbox):
        """Create a new track."""
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.velocities[obj_id] = (0, 0)
        self.next_id += 1
        return obj_id

    def deregister(self, obj_id):
        """Remove a track that has been missing too long."""
        del self.objects[obj_id]
        del self.bboxes[obj_id]
        del self.disappeared[obj_id]
        del self.velocities[obj_id]

    def _predicted_centroids(self):
        """
        Return predicted centroids for all current tracks by adding
        their velocity to their last known position.  This is used as
        the reference point for matching, which improves ID retention
        when objects are briefly occluded or moving steadily.
        """
        predicted = OrderedDict()
        for obj_id, centroid in self.objects.items():
            vx, vy = self.velocities[obj_id]
            predicted[obj_id] = (centroid[0] + vx, centroid[1] + vy)
        return predicted

    def update(self, detections):
        """
        Match incoming detections to existing tracks.

        Parameters
        ----------
        detections : list[dict]
            Each dict must contain a "bbox" key with [x1, y1, x2, y2].

        Returns
        -------
        dict
            Mapping of track_id -> detection dict (augmented with "track_id").
        """
        # Compute centroids from bounding boxes
        input_centroids = []
        input_bboxes = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))
            input_bboxes.append(det["bbox"])

        # If no detections, mark every existing track as disappeared
        if len(input_centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return {}

        input_centroids = np.array(input_centroids)

        # If we have no existing tracks, register everything
        if len(self.objects) == 0:
            matched = {}
            for i, det in enumerate(detections):
                tid = self.register(input_centroids[i], input_bboxes[i])
                det_copy = dict(det)
                det_copy["track_id"] = tid
                matched[tid] = det_copy
            return matched

        # Use velocity-predicted positions for matching
        predicted = self._predicted_centroids()
        object_ids = list(predicted.keys())
        predicted_centroids = np.array(list(predicted.values()))

        D = dist.cdist(predicted_centroids, input_centroids)

        # Greedy matching: smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        matched = {}

        alpha = self.velocity_smoothing

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            old_centroid = self.objects[obj_id]
            new_centroid = input_centroids[col]

            # Update velocity with exponential moving average
            raw_vx = new_centroid[0] - old_centroid[0]
            raw_vy = new_centroid[1] - old_centroid[1]
            old_vx, old_vy = self.velocities[obj_id]
            self.velocities[obj_id] = (
                alpha * old_vx + (1 - alpha) * raw_vx,
                alpha * old_vy + (1 - alpha) * raw_vy,
            )

            self.objects[obj_id] = tuple(new_centroid)
            self.bboxes[obj_id] = input_bboxes[col]
            self.disappeared[obj_id] = 0

            det_copy = dict(detections[col])
            det_copy["track_id"] = obj_id
            matched[obj_id] = det_copy

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing tracks
        for row in range(len(object_ids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        # Handle unmatched new detections -- register them
        for col in range(len(input_centroids)):
            if col not in used_cols:
                tid = self.register(input_centroids[col], input_bboxes[col])
                det_copy = dict(detections[col])
                det_copy["track_id"] = tid
                matched[tid] = det_copy

        return matched
