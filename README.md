# Stereo Mosaicing & X-Slit Forward Panoramas

Implementation of stereo mosaicing from sideways-panning video, plus a bonus X-Slit projection method that creates the illusion of forward/backward camera motion from the same input.

![Boat X-Slit Demo](outputs/boat_xslit_mincut.gif)

## Background

**Stereo Mosaicing** takes a video panning horizontally across a scene and generates multiple panoramas from different virtual viewpoints. When played back, this creates a parallax/3D effect — objects at different depths appear to shift relative to each other.

**X-Slit Forward Panoramas** (bonus) use non-stationary mosaicing to simulate forward motion. Instead of stitching fixed vertical strips, we sample columns whose temporal position shifts linearly across the frame, effectively slicing the space-time volume diagonally. This creates a "dolly zoom" effect from a camera that only moved sideways.

## Key Techniques

- **Horizon Pairs Homography**: Estimates rotation only from point pairs that are horizontally distant but vertically close, avoiding the "banana effect" caused by parallax at different depths
- **Barcode Blending**: Builds separate odd/even frame mosaics with double-width strips, then blends with Laplacian pyramids to eliminate blinking artifacts
- **Min-Cut Seam Finding** (bonus): Dynamic programming to find optimal seams between strips, preserving object boundaries better than averaging
- **X-Slit Projection** (bonus): Diagonal space-time slicing with configurable slope to control apparent forward/backward motion

## Repository Structure
```
├── ex4.py                 # Main stereo mosaicing implementation
├── ex4_bonus.py           # X-Slit forward panorama implementation
├── requirements.txt
├── reports/
│   ├── ex4_report.pdf     # Main exercise report
│   └── ex4_bonus_report.pdf
├── inputs/
│   ├── viewpoint_input.mp4
│   ├── dynamic_input.mp4
│   ├── good_input.mp4
│   └── bad_input.mp4
└── outputs/
    ├── viewpoint_result.mp4
    ├── dynamic_result.mp4
    ├── good_result.mp4
    ├── bad_result.mp4
    ├── boat_xslit_barcode.mp4
    └── boat_xslit_mincut.mp4
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Stereo Mosaicing (Main Exercise)
```python
from ex4 import generate_panorama

# API required by exercise spec
panoramas = generate_panorama("path/to/frames/", n_out_frames=24)
# Returns list of PIL images
```

For video input directly:
```python
from ex4 import generate_dynamic_panorama
import imageio.v3 as iio

frames = generate_dynamic_panorama("inputs/boat.mp4", num_viewpoints=24)
iio.imwrite("outputs/boat_stereo.mp4", frames, fps=30)
```

### X-Slit Forward Panorama (Bonus)
```python
from ex4_bonus import generate_forward_panorama_video
import imageio.v3 as iio

frames = generate_forward_panorama_video(
    "inputs/boat.mp4",
    num_output_frames=120,
    slope_start=0.0,          # No X-Slit effect at start
    slope_end=1.5,            # Diagonal slicing at end
    zoom_start=1.0,
    zoom_end=1.5,
    vertical_stretch_start=1.0,
    vertical_stretch_end=0.85,  # Compensate for X-Slit aspect ratio distortion
    blend_mode='pyramid'      # Or 'mincut' for sharper seams (~5x slower)
)

# Loop forward and backward for smooth playback
frames = np.concatenate([frames, frames[::-1]], axis=0)
iio.imwrite("outputs/boat_xslit.mp4", frames, fps=60)
```

**Note on vertical stretch**: The X-Slit projection inherently distorts aspect ratio (as noted by Peleg et al. in "Mosaicing New Views"). The `vertical_stretch` parameters compensate for this by scaling the image height. Since we lack real depth data, these values are tuned per video.

## Results

### Stereo Mosaicing

| Input | Output (24 viewpoints) |
|-------|------------------------|
| viewpoint_input.mp4 | Smooth parallax effect, barcode blending mitigates pole blinking |
| dinamic_input.mp4 | Vertical motion filter handles waterfall dynamics |

### X-Slit Forward Panorama (Bonus)

| Method | Quality | Time | Notes |
|--------|---------|------|-------|
| Barcode (pyramid) | Good, some blinking on poles | ~30s | Default choice for speed |
| Min-cut seams | Sharper, blockier transitions | ~150s | Better object preservation |

## Limitations

The algorithm assumes objects are distant relative to their internal depth variation. Scenes with multiple distinct depth planes (like ground tiles + distant buildings) break the single-homography assumption, causing ghosting and stretching artifacts. See `bad_input.mp4` for an example.

## References

- Vivet, D., Peleg, S., Binefa, X. "Real-Time Stereo Mosaicing using Feature Tracking" (2011) — Barcode blending with double-width strips
- Zomet, A., Feldman, D., Peleg, S. "Mosaicing New Views: The Crossed-Slits Projection" (2003) — X-Slit projection model and space-time slicing
- Peleg, S., Ben-Ezra, M., Pritch, Y. "Omnistereo: Panoramic Stereo Imaging" (2001) — Foundations of stereo mosaicing