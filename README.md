# Leo Data Pipeline

Extracts and undistorts frames from ROS2 mcap bag files for YOLO training.

Records `/camera/image_color/compressed` and `/camera/camera_info` topics, undistorts frames inline using the camera intrinsics, and writes clean JPEGs ready for labelling.

No ROS installation required — uses the pure-Python `mcap` library to read bag files.

---

## Setup

### Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda/Anaconda)

### Create the environment

```bash
conda env create -f environment.yml
conda activate leo_pipeline
```

---

## Usage

```bash
python scripts/parse_mcap.py <path/to/bag> [options]
```

Accepts either a bag directory or a direct path to a `.mcap` file.

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | `frames/` | Output directory |
| `-n`, `--every-n` | `1` | Save every Nth frame (e.g. `5` = 20% of frames) |
| `--alpha` | `0.0` | Undistortion crop: `0.0` crops black borders (best for YOLO), `1.0` keeps all pixels |

### Examples

```bash
# Extract every frame, crop black borders (YOLO-ready)
python scripts/parse_mcap.py rosbags/rosbag2_2026_04_22-11_02_50/ -o frames/

# Extract every 5th frame
python scripts/parse_mcap.py rosbags/rosbag2_2026_04_22-11_02_50/ -o frames/ -n 5

# Keep all pixels (adds black borders)
python scripts/parse_mcap.py rosbags/rosbag2_2026_04_22-11_02_50/ -o frames/ --alpha 1.0
```

---

## Output

```
frames/
├── camera_info.json        # Corrected intrinsics (zero distortion)
├── frame_00000000000000001.jpg
├── frame_00000000000000002.jpg
└── ...
```

`camera_info.json` contains the updated camera matrix `K` after undistortion, with distortion coefficients zeroed out.
