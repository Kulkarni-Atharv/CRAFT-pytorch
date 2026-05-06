# CRAFT Text Detection — Raspberry Pi CM5 + Global Shutter Camera

Real-time text **region detection** on a Raspberry Pi Compute Module 5 using the [CRAFT](https://arxiv.org/abs/1904.01941) (Character Region Awareness For Text detection) model and the Raspberry Pi Global Shutter Camera via Picamera2.

> **Detection vs Recognition** — CRAFT finds *where* text is (bounding boxes / polygons). It does not read the characters. Pipe each cropped region into a recognition model (EasyOCR, TrOCR, Tesseract) downstream if you need the actual string.

<img width="1000" alt="CRAFT example" src="./figures/craft_example.gif">

---

## Hardware

| Component | Details |
|---|---|
| Compute Module | Raspberry Pi CM5 |
| Camera | Raspberry Pi Global Shutter Camera (IMX296) |
| Interface | CSI-2 via CM5 IO Board |

---

## Repository Layout

```
CRAFT-pytorch/
├── capture_cm5.py          ← live camera capture + CRAFT detection  (main script)
├── craft.py                ← CRAFT model architecture
├── craft_utils.py          ← post-processing (thresholding, box extraction)
├── imgproc.py              ← image pre/post-processing helpers
├── refinenet.py            ← optional link refiner for curved/dense text
├── file_utils.py           ← file I/O helpers
├── test.py                 ← batch inference on a folder of images
├── basenet/                ← VGG16-BN backbone
├── weights/                ← place downloaded .pth model files here
├── results_cm5/            ← annotated output images saved here
├── requirements_cm5.txt    ← CM5 Python dependencies
└── requirements.txt        ← original dependencies reference
```

---

## Setup on CM5

### 1 — System packages

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-venv python3-dev
```

### 2 — Virtual environment (system site-packages required for picamera2)

```bash
cd ~/CRAFT-pytorch
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 3 — Install PyTorch (CPU-only, ARM64)

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> This downloads ~250 MB and can take 10–15 min on CM5.

### 4 — Install remaining dependencies

```bash
pip install opencv-python scikit-image Pillow numpy scipy
```

### 5 — Verify

```bash
python -c "import torch, cv2, picamera2, craft; print('All imports OK')"
```

---


## Running the Live Camera Script

Every new terminal session needs:
```bash
source .venv/bin/activate
```

### Basic run

```bash
python capture_cm5.py --model weights/craft_mlt_25k.pth
```

### With link refiner (better on curved or closely-packed text)

```bash
python capture_cm5.py --model weights/craft_mlt_25k.pth \
  --refine --refiner_model weights/craft_refiner_CTW1500.pth
```

### Restrict to a sensor region of interest (x y w h)

```bash
python capture_cm5.py --model weights/craft_mlt_25k.pth --roi 100 50 1200 900
```

### Camera controls

| Key | Action |
|---|---|
| `SPACE` | Capture clean frame, run detection, draw boxes, save result, exit |
| `Q` | Quit without capturing |

Annotated images and heatmaps are saved to `results_cm5/` automatically.

---


## Batch Inference on Images (no camera)

```bash
python test.py \
  --trained_model weights/craft_mlt_25k.pth \
  --test_folder /path/to/images \
  --cuda False
```

Results are saved to `./result/`.

---

## How CRAFT Works

1. **Backbone** — VGG16-BN extracts multi-scale features.
2. **U-Net decoder** — fuses features into two score maps:
   - **Region score** — probability that a pixel belongs to a character centre.
   - **Affinity score** — probability that two adjacent characters belong to the same word.
3. **Post-processing** — threshold both maps, run connected components, fit minimum-area rectangles (or tight polygons with `--poly`).
4. **Output** — quad bounding boxes / polygons in original image coordinates.

---

## Tuning Tips for the Global Shutter Camera

- Lower `--text_threshold` (e.g. `0.5`) if text regions are missed in low-contrast scenes.
- Raise `--canvas_size` (e.g. `1600`) for small or dense text — at the cost of slower inference.
- Use `--roi` to limit processing to the area of interest and reduce inference time.
- The global shutter eliminates rolling-shutter distortion on fast-moving targets, improving box accuracy on conveyor belts, robotics, or handheld capture.

---


