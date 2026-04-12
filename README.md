# Picker-Bot

**Developers:** `<Developer Name 1>`, `<Developer Name 2>`
**Module:** `<Module Name>` | **University:** `<University Name>`

---

## About

Picker-Bot is a computer-vision-guided pick-and-place system built around an **EPSON VT6-A901S** 6-axis industrial manipulator. It detects microelectronic modules on a work surface using a **YOLOv8 Oriented Bounding Box (OBB)** model, maps their pixel locations to real-world robot coordinates via **homography calibration**, and commands the robot to pick each module and deposit it in a collection tray — all over a **TCP/IP** socket link.

Key technologies: Python, OpenCV, Ultralytics YOLOv8, NumPy, Epson RC8 robot programming.

---

## Codebase Structure

```
picker-bot/
├── Scripts/                            # Core application scripts
│   ├── pickerbot.py                    # Main orchestrator — runs the full pipeline
│   ├── pickerbot_sender.py             # TCP/IP robot communication module
│   ├── detect_and_classify.py          # YOLOv8-OBB detection & annotation module
│   ├── teleop_mouse.py                 # Interactive mouse control & calibration utilities
│   └── cv_discovery.py                 # (Legacy) Threshold-based contour detection
│
├── Utils/
│   └── Calibration/                    # Calibration tools and data
│       ├── calibration.py              # Homography calibration loader
│       ├── camera_alignment.py         # Visualises saved calibration points
│       ├── sort_and_tag_pixels.py      # Sorts raw pixel clicks & maps world coords
│       ├── calibration_pixels.csv      # Base 77-point pixel calibration
│       ├── calibration_pixels_scaled.csv # Scaled calibration with world coordinates
│       └── graph_paper.jpg             # Calibration grid target image (1280x720)
│
├── Epson/
│   └── Pickerbot_Receiver/             # Robot-side program (Epson RC8)
│       └── Main.prg                    # TCP listener — executes JUMP/GO/MOVE/PICK/STANDBY
│
├── models/
│   ├── best.pt                         # YOLOv8-OBB model for module detection
│   ├── cw_keras/                       # (Legacy) Keras classifier — 5 classes
│   └── pir_daytime_sample_model/       # (Legacy) Keras classifier — 3 classes
│
├── Legacy/
│   ├── main_orchestrator.py            # (Legacy) Earlier orchestration script
│   └── keras_inference.py              # (Legacy) Keras classification module
│
├── test-dataset/
│   └── test-samples/                   # Test images for offline development
│
├── config.json                         # Runtime configuration
├── LICENSE
└── README.md
```

| Folder | Purpose |
|--------|---------|
| `Scripts/` | All active Python scripts — the main entry point and its supporting modules. |
| `Utils/Calibration/` | Calibration data files, grid images, and helper scripts for the pixel-to-world mapping pipeline. |
| `Epson/` | Robot-side Epson RC8 program that listens for TCP commands and drives the manipulator. |
| `models/` | Trained model weights. `best.pt` is the active YOLOv8-OBB model. |
| `Legacy/` | Superseded scripts kept for reference. Not used in the current pipeline. |
| `test-dataset/` | Sample images for testing detection without a live camera. |

---

## Content Libraries

The following modules are designed to be imported by other scripts:

### `Scripts/pickerbot_sender.py` — Robot TCP Interface

Manages the TCP/IP socket connection to the EPSON controller.

| Function | Description |
|----------|-------------|
| `connect(ip, port)` | Opens a TCP socket to the robot controller. |
| `disconnect()` | Closes the socket. |
| `epsonGo(x, y, z, u)` | Sends a GO (linear move) command. |
| `epsonJump(x, y, z, u)` | Sends a JUMP command. |
| `epsonMove(x, y, z, u)` | Sends a MOVE command. |
| `epsonPick(x, y, z, u)` | Sends a PICK command (move + grip + deposit). |
| `epsonStandby()` | Returns the robot to its home position. |
| `epsonPickAll(locations)` | Batch-picks a list of locations; aborts on first failure. |

### `Scripts/detect_and_classify.py` — YOLO Detection Module

Wraps the YOLOv8-OBB model for inference and annotation.

| Function | Description |
|----------|-------------|
| `detect_and_annotate(frame, confidence)` | Runs detection on a frame. Returns `(annotated_frame, detections)` where each detection is `(cx, cy, angle, label, conf)`. Filters out the `"noise"` class. |

### `Scripts/teleop_mouse.py` — Calibration Utilities

Provides homography calibration loading and the height-recalibration GUI.

| Function | Description |
|----------|-------------|
| `load_calibration_data(csv_filename)` | Loads pixel and world coordinate arrays from a calibration CSV. |
| `calculate_homography(src_pts, dst_pts)` | Computes a homography matrix from point correspondences. |
| `pixel_to_world(H, pixel_x, pixel_y)` | Transforms a pixel coordinate to world coordinates using the homography. |
| `run_calibration_gui()` | Opens the ruler-based height recalibration GUI. |

### `Utils/Calibration/calibration.py` — Homography Loader

Standalone calibration utility with the same `load_calibration_data`, `calculate_homography`, and `pixel_to_world` functions.

### `Legacy/keras_inference.py` — Keras Classifier

Provides `ModuleClassifier(model_dir)` which loads a Keras `.h5` model and `labels.txt`. Call `.predict(cv2_image)` to get `(label, confidence)`. Compatible with Google Teachable Machine exports.

---

## Standard Operation Steps

The typical workflow follows four stages: **Calibration**, **TCP Connection**, **Detection**, and **Pick Operation**.

### Step 1 — Calibration

Calibration maps pixel coordinates from the camera image to real-world millimetre coordinates on the work surface using a 77-point homography grid (11 columns x 7 rows, 20 mm spacing).

**Initial setup (one-time):**

1. Capture `graph_paper.jpg` with the camera positioned over the calibration grid.
2. Click all 77 grid intersections to produce `calibration_pixels.csv`.
3. Run the sorting and mapping script:
   ```bash
   python Utils/Calibration/sort_and_tag_pixels.py
   ```
   This sorts the points left-to-right, top-to-bottom and appends world coordinates.

4. (Optional) Verify the points visually:
   ```bash
   python Utils/Calibration/camera_alignment.py
   ```

**Height recalibration (when camera height changes):**

```bash
python Scripts/pickerbot.py --calibrate
```

This opens a GUI where you click two points 20 mm apart on `ruler.jpg`. The script computes a scale factor, rescales all 77 calibration points, saves the result to `calibration_pixels_scaled.csv`, and updates `config.json`.

### Step 2 — TCP Connection

Configure the connection in `config.json`:

```json
{
  "enable_epson_tcp": true,
  "epson_ip": "192.168.150.2",
  "epson_port": 2001
}
```

| Setting | Description |
|---------|-------------|
| `enable_epson_tcp` | Set `true` to send commands to the robot. When `false`, detection runs but no commands are sent. |
| `epson_ip` | `127.0.0.1` for the EPSON simulator, `192.168.150.2` for the physical robot. |
| `epson_port` | TCP port (default `2001`). |

On the robot side, deploy `Epson/Pickerbot_Receiver/Main.prg` to the controller. The program starts a TCP server, listens for commands, and executes them.

### Step 3 — Detection

Detection uses the YOLOv8-OBB model (`models/best.pt`) to locate microelectronic modules and estimate their orientation angle.

- **Live camera:** Set `"input_mode": "camera"` and `"webcam_id": 1` in `config.json`.
- **Static image:** Set `"input_mode": "image"` and `"test_image_path": "<path>"`.

The confidence threshold is controlled by `"min_confidence"` (default `0.7`).

### Step 4 — Pick Operation

Once detections are obtained, `pickerbot.py` converts each pixel centroid to world coordinates using the homography matrix, then sends batch PICK commands to the robot via TCP. Each PICK triggers the following sequence on the manipulator:

1. Lift to clearance height
2. Move above the target (x, y)
3. Descend to pick height (z) with rotation (u)
4. Activate gripper
5. Lift and move to deposit tray
6. Release gripper
7. Return to standby

---

## Usage Examples

### Run the full pipeline (camera input)

```bash
python Scripts/pickerbot.py --src camera
```

### Run the full pipeline (image input)

```bash
python Scripts/pickerbot.py --src test-dataset/test-samples/31.jpg
```

### Run height recalibration only

```bash
python Scripts/pickerbot.py --calibrate
```

### Run standalone detection on an image

```bash
python Scripts/detect_and_classify.py test-dataset/test-samples/31.jpg
```

### Run standalone detection on the live camera

```bash
python Scripts/detect_and_classify.py
```

### Interactive mouse control (teleop)

Click anywhere on the calibration image to move the robot to that world coordinate:

```bash
python Scripts/teleop_mouse.py
```

Press `q` to quit.

### Visualise calibration points

```bash
python Utils/Calibration/camera_alignment.py
```

---

## Command-Line Reference

### `Scripts/pickerbot.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--src` | `"camera"` | Input source. Use `"camera"` for live feed or provide a path to an image file. |
| `--calibrate` | off | Run the height recalibration GUI and exit. |

### `Scripts/detect_and_classify.py`

| Argument | Default | Description |
|----------|---------|-------------|
| positional `image_path` | *(none — uses camera)* | Path to an image file. If omitted, opens the live camera feed with a confidence trackbar. |

---

## Configuration Reference (`config.json`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_mode` | string | `"image"` | `"image"` to use a test image, `"camera"` for live feed. |
| `webcam_id` | int | `1` | Camera device index (`0` = built-in, `1` = first USB camera). |
| `test_image_path` | string | `"test-dataset/test-samples/31.jpg"` | Path to the test image when `input_mode` is `"image"`. |
| `min_confidence` | float | `0.7` | YOLO detection confidence threshold (0.0–1.0). |
| `enable_epson_tcp` | bool | `false` | Enable/disable TCP commands to the robot. |
| `epson_ip` | string | `"127.0.0.1"` | Robot controller IP address. |
| `epson_port` | int | `2001` | Robot controller TCP port. |
| `robot_z` | int | `360` | Default Z-axis pick height in mm. |
| `calibration_file` | string | `"Utils/Calibration/calibration_pixels_scaled.csv"` | Path to the active calibration CSV. |
