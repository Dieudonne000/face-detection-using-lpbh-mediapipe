# Face Detection & Recognition System

A real-time face detection and recognition system using **MediaPipe** for face detection and **LBPH (Local Binary Patterns Histograms)** for face recognition. This project provides a complete pipeline from dataset creation to real-time recognition.

## Features

- 🎥 **Real-time face detection** using MediaPipe
- 👤 **Face recognition** using LBPH algorithm
- 📸 **Dataset creation** with automatic face cropping
- 🔍 **Dataset review and cleanup** tool
- 🎯 **Model training** with validation support
- 🖥️ **Real-time webcam recognition** with face mesh visualization
- 📷 **Single image recognition** mode

## Technologies

- **OpenCV** - Image processing and LBPH face recognition
- **MediaPipe** - Face detection and mesh visualization
- **Python 3** - Core programming language

## Project Structure

```
face-detection/
├── 01_create_dataset.py      # Dataset creation tool
├── 02_review_dataset.py       # Dataset preview and cleanup
├── 03_train_model.py          # Model training script
├── 04_predict.py              # Real-time recognition
├── dataset/                   # Face image dataset
│   ├── Person1/              # Person 1 images
│   └── Person2/              # Person 2 images
├── models/                    # Trained models
│   ├── lbph_face_model.pkl   # LBPH recognizer model
│   └── lbph_label_map.pkl    # Label to name mapping
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (for dataset creation and real-time recognition)

### Dependencies

Install the required packages:

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

Or create a `requirements.txt` file:

```txt
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
mediapipe>=0.10.0
numpy>=1.19.0
```

Then install:

```bash
pip install -r requirements.txt
```

**Note:** `opencv-contrib-python` is required for the LBPH face recognizer (`cv2.face` module).

## Usage

### 1. Create Dataset

Capture face images for each person:

```bash
python 01_create_dataset.py
```

- Enter the person's ID/name when prompted
- Position yourself in front of the webcam
- The script will automatically capture up to 250 images at 100ms intervals
- Images are saved to `dataset/<person_id>/`
- Press 'q' to quit early

**Tips:**
- Move your head slightly during capture for better training data
- Ensure good lighting
- Capture from different angles if possible

### 2. Review Dataset (Optional)

Preview and clean your dataset:

```bash
python 02_review_dataset.py
```

**Controls:**
- `←` / `P` - Previous image
- `→` / `N` - Next image
- `Space` / `S` - Play/Pause slideshow
- `+` / `-` - Adjust slideshow speed
- `D` - Delete current image
- `Q` / `ESC` - Quit

### 3. Train Model

Train the LBPH face recognition model:

```bash
python 03_train_model.py
```

**Options:**
```bash
python 03_train_model.py --val-split 0.2    # 20% validation split (default)
python 03_train_model.py --val-split 0      # No validation (use all data)
python 03_train_model.py --radius 1 --neighbors 8 --grid-x 8 --grid-y 8
```

**Parameters:**
- `--val-split` - Validation split ratio (0.0 to 1.0, default: 0.2)
- `--radius` - LBPH radius parameter (default: 1)
- `--neighbors` - LBPH neighbors parameter (default: 8)
- `--grid-x` - LBPH grid X parameter (default: 8)
- `--grid-y` - LBPH grid Y parameter (default: 8)

The trained model will be saved to `models/lbph_face_model.pkl` and label mapping to `models/lbph_label_map.pkl`.

### 4. Run Recognition

#### Real-time Webcam Recognition

```bash
python 04_predict.py
```

**Options:**
```bash
python 04_predict.py --camera 0                    # Use camera index 0 (default)
python 04_predict.py --confidence 0.7              # Detection confidence (default: 0.7)
python 04_predict.py --threshold 70.0              # Recognition threshold (default: 70.0, lower=stricter)
python 04_predict.py --show-mesh                   # Show MediaPipe face mesh overlay
```

#### Single Image Recognition

```bash
python 04_predict.py --image photo.jpg
python 04_predict.py --image photo.jpg --show-mesh
```

**Parameters:**
- `--camera` - Camera index (default: 0)
- `--image` - Process single image instead of webcam
- `--confidence` - Minimum face detection confidence (0-1, default: 0.7)
- `--threshold` - LBPH recognition threshold (lower = stricter, default: 70.0)
- `--show-mesh` - Enable MediaPipe face mesh visualization

**Controls:**
- Press `q` to quit

## How It Works

1. **Face Detection**: MediaPipe detects faces in real-time with high accuracy
2. **Face Cropping**: Detected faces are automatically cropped and saved during dataset creation
3. **Model Training**: LBPH algorithm learns facial patterns from the collected dataset
4. **Recognition**: New faces are compared against the trained model to identify known persons

### LBPH Algorithm

LBPH (Local Binary Patterns Histograms) is a texture-based face recognition algorithm that:
- Works well with grayscale images
- Is robust to lighting variations
- Provides fast recognition
- Returns confidence scores (lower = better match)

## Configuration

### Dataset Creation (`01_create_dataset.py`)

Edit these constants in the script:
- `MAX_IMAGES` - Maximum images to capture (default: 250)
- `INTERVAL_MS` - Capture interval in milliseconds (default: 100)
- `DATASET_DIR` - Dataset directory (default: 'dataset')

### Recognition Threshold

The `--threshold` parameter controls recognition strictness:
- **Lower values (e.g., 50.0)**: Stricter, fewer false positives, more false negatives
- **Higher values (e.g., 100.0)**: More lenient, more false positives, fewer false negatives
- **Default (70.0)**: Balanced setting

Adjust based on your needs and lighting conditions.

## Troubleshooting

### "Model not found" Error

Make sure you've trained the model first:
```bash
python 03_train_model.py
```

### "No images loaded" Error

- Check that `dataset/<person_name>/*.jpg` files exist
- Verify images are readable (not corrupted)
- Ensure proper file extensions (.jpg, .jpeg, .png)

### Webcam Not Working

- Check camera permissions
- Try different camera index: `--camera 1`
- Verify camera is not being used by another application

### Poor Recognition Accuracy

- Collect more training images (aim for 100+ per person)
- Ensure good lighting during both training and recognition
- Capture images from different angles and expressions
- Adjust `--threshold` parameter
- Review and clean dataset using `02_review_dataset.py`

### OpenCV Face Module Not Found

Make sure you have `opencv-python` installed:
```bash
pip install opencv-python
```

## License

This project is provided as-is for educational and personal use.

## Authors

- MUNEZA Jean Dieudonne

## Acknowledgments

- **MediaPipe** by Google for face detection
- **OpenCV** for computer vision tools
- LBPH algorithm implementation

