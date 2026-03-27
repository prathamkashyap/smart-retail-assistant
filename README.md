# 🛒 Smart Retail Store Assistant

### Retail Shelf Intelligence and Customer Interaction Analytics

A real-time Computer Vision system that monitors retail shelves using a webcam or recorded video. It detects customers and products, tracks their interactions with configurable shelf zones, identifies anomalies, and generates structured analytics logs—all using a single camera and a pretrained YOLOv8 model.

Built with **Python**, **OpenCV**, **YOLOv8 (Ultralytics)**, **NumPy**, and **Pandas**. No custom model training required.

---

## ✨ Key Features

- **Real-time object detection** using YOLOv8 nano (pretrained on COCO) for persons, bottles, cups, bags, books, and more
- **Velocity-predicted centroid tracking** with EMA smoothing for stable, persistent identity assignment across frames
- **Configurable shelf zones** defined as rectangular regions in a JSON config file—each zone is monitored independently
- **Customer proximity detection** — determines when a person is near a shelf zone
- **Dwell-time estimation** — tracks how long a customer stays near a specific zone
- **Product removal detection** with **temporal smoothing** — uses a 5-frame history buffer to eliminate false positives from detection flickering
- **Repeated attention detection** — counts how many times the same person revisits a zone
- **Anomaly detection** — sudden stock drops relative to baseline, prolonged empty zones
- **Debounced CSV event logging** — suppresses duplicate events within a configurable time window
- **Post-session analytics** — summary statistics printed to console and saved as CSV
- **On-screen HUD** — zone overlays, bounding boxes, track IDs, dwell counters, FPS, and event counts
- **Snapshot saving** — frames captured automatically when notable events occur

---

## 🏗️ System Architecture

The system follows a linear pipeline where each video frame passes through five processing stages:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Video Input │───▶│   YOLOv8     │───▶│   Centroid   │
│ (Webcam/File)│    │  Detection   │    │   Tracking   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐    ┌───────▼──────┐
                    │    Event     │◀───│  Interaction │
                    │   Logging    │    │    Logic     │
                    └──────┬───────┘    └──────┬───────┘
                           │                   │
                    ┌──────▼───────┐    ┌──────▼───────┐
                    │   Analytics  │    │ Zone Analysis│
                    │   Summary    │    │ & Visualise  │
                    └──────────────┘    └──────────────┘
```

Each stage is implemented as an independent Python class communicating through well-defined interfaces.

---

## 📁 Folder Structure

```
smart-retail-assistant/
├── configs/
│   └── default_config.json        # Zone definitions, thresholds, model settings
├── src/
│   ├── __init__.py
│   ├── detect.py                  # YOLOv8 detection wrapper
│   ├── track.py                   # Centroid tracker with velocity prediction
│   ├── zone_manager.py            # Shelf zone logic and drawing
│   ├── logic.py                   # Interaction, dwell-time, anomaly logic
│   ├── logger.py                  # CSV event logger with debounce
│   ├── analytics.py               # Post-session summary statistics
│   └── main.py                    # Entry point — real-time pipeline
├── data/                          # Place demo videos here (e.g., demo_video.mp4)
├── models/
│   └── yolov8n.pt                 # YOLOv8 nano weights (~6 MB)
├── outputs/
│   ├── logs.csv                   # Generated during a session
│   ├── analytics_summary.csv      # Generated after a session
│   ├── snapshots/                 # Auto-captured frames for notable events
│   └── screenshots/               # Manual or example screenshots
├── report.tex                     # LaTeX project report
├── report_guide.txt               # Guide for writing the project report
├── requirements.txt
├── run.sh                         # Linux/macOS launcher
├── run.bat                        # Windows launcher
└── README.md
```

---

## 🔧 Installation

```bash
# Clone or download the repository
cd smart-retail-assistant

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥8.0.0 | YOLOv8 inference |
| `opencv-python` | ≥4.8.0 | Frame capture & visualisation |
| `numpy` | ≥1.24.0 | Numerical operations |
| `pandas` | ≥2.0.0 | CSV logging & analytics |
| `scipy` | ≥1.10.0 | Distance computation for tracking |

---

## 🧠 Model Setup

The system expects the YOLOv8 nano weights at `models/yolov8n.pt`.

**Option A** — The weights are already included in the repository. No action needed.

**Option B** — If the file is missing, download it:

```bash
# Using the Ultralytics CLI
pip install ultralytics
yolo export model=yolov8n.pt
mv yolov8n.pt models/

# Or download directly
curl -L -o models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
```

The system will display a clear error message with download instructions if the model file is not found.

---

## 🚀 How to Run

### Using webcam (default)

```bash
python src/main.py
```

### Using a recorded video file

```bash
python src/main.py --source data/demo_video.mp4
```

### Using a custom config

```bash
python src/main.py --config configs/custom_config.json
```

### Using the launcher scripts

```bash
# Linux/macOS
chmod +x run.sh && ./run.sh

# Windows
run.bat
```

Press **`q`** to quit. On exit, the system prints a session analytics summary and saves it to `outputs/analytics_summary.csv`.

---

## 📊 Example Outputs

After a session, the following outputs are generated:

| Output | Location | Description |
|--------|----------|-------------|
| Event log | `outputs/logs.csv` | All detected events with timestamps |
| Analytics summary | `outputs/analytics_summary.csv` | Aggregated statistics |
| Event snapshots | `outputs/snapshots/` | Frame captures for removals, anomalies |
| Screenshots | `outputs/screenshots/` | Manual or reference screenshots |

### Running Analytics Separately

```bash
python src/analytics.py                        # uses default log path
python src/analytics.py outputs/logs.csv       # specify custom log
```

### Event Types

| Event | Trigger |
|-------|---------|
| `INTERACTION_START` | Person enters proximity of a shelf zone |
| `INTERACTION_END` | Person leaves the zone's proximity |
| `DWELL_ALERT` | Person stays near a zone longer than the threshold |
| `REPEATED_ATTENTION` | Person revisits the same zone ≥3 times |
| `PRODUCT_REMOVED` | Product count in a zone drops (confirmed over 3+ frames) |
| `ANOMALY_STOCK_DROP` | Zone product count falls below 50% of baseline |
| `ANOMALY_EMPTY_ZONE` | Zone continuously empty beyond alert duration |

---

## ⚙️ Configuration

All parameters are in `configs/default_config.json`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.weights` | YOLO model file | `models/yolov8n.pt` |
| `model.confidence_threshold` | Minimum detection confidence | 0.35 |
| `proximity_margin` | Pixel margin for proximity detection | 80 |
| `dwell_time_threshold_sec` | Seconds before dwell alert | 3.0 |
| `empty_zone_alert_sec` | Seconds before empty zone anomaly | 10.0 |
| `repeated_attention_threshold` | Visit count for repeated attention | 3 |
| `stock_drop_ratio` | Baseline ratio for stock drop anomaly | 0.5 |
| `tracker_max_disappeared` | Frames before dropping a track | 25 |
| `tracker_max_distance` | Max pixel distance for centroid matching | 120 |
| `log_debounce_sec` | Duplicate event suppression window | 2.0 |

### Adjusting Shelf Zones

Edit the `shelf_zones` array in the config. Each zone has:
- `zone_id`: unique string identifier
- `label`: display name shown on screen
- `bbox`: `[x1, y1, x2, y2]` in pixel coordinates (relative to display resolution)
- `color`: `[B, G, R]` colour for the overlay

Zone coordinates must be adjusted to match your camera's view of the shelf.

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Object Detection | YOLOv8 nano (Ultralytics, pretrained on COCO) |
| Image Processing | OpenCV |
| Object Tracking | Custom centroid tracker with velocity prediction (SciPy) |
| Data Handling | Pandas, NumPy |
| Configuration | JSON |

---

## ⚠️ Known Limitations

- **Zone calibration**: Zone coordinates are in pixel space (960×540 default). Changing the camera angle or resolution requires manual recalibration.
- **Limited product vocabulary**: The pretrained COCO model covers ~10 retail-relevant categories. Many packaged goods are not recognised. Fine-tuning on a retail dataset would expand coverage.
- **Single camera**: The system monitors one view. Blind spots are possible.
- **CPU-bound by default**: Runs on CPU without GPU acceleration. A CUDA-capable GPU and appropriate PyTorch build will improve throughput.
- **Prototype scope**: Designed for academic demonstration, not production deployment.

---

## 🔮 Future Improvements

- **Custom-trained model** on retail-specific datasets (e.g., SKU-110K) for broader product recognition
- **Deep SORT integration** with ReID features for robust tracking through full occlusions
- **Automatic zone suggestion** using edge detection or product clustering
- **Real-time web dashboard** with push notifications for stock-outs and anomalies
- **Spatial heat maps** showing customer attention patterns over time
- **Multi-camera support** with cross-camera tracking for full-store coverage
- **Edge deployment** on NVIDIA Jetson or Raspberry Pi with Coral TPU

---

## 📝 Demo Workflow

1. Set up a shelf with a few objects (bottles, cups, books, bags).
2. Point your webcam at the shelf.
3. Run `python src/main.py`.
4. Walk up to the shelf, pick up an item, put it back.
5. The system will log interaction start/end, dwell time, and product removal events.
6. Leave an area empty for more than 10 seconds to trigger an empty-zone anomaly.
7. Press **`q`** to stop and review the analytics summary.

---

## 📜 License

This project is for academic and educational purposes.
