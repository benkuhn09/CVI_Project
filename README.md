# CVI Project — Analysis of Student Activities on a University Campus

Pedestrian detection and trajectory analysis on the PETS S2.L1 View001 dataset (795 frames).

## Dataset & Setup

- **Frames:** `frames/View_001/` — 795 images from the PETS S2.L1 View001 sequence
- **Ground truth:** `data/gt/gt.txt` — manual annotations, one row per detection
- **Detections:** `data/det/det.txt` — pre-computed ACF detector output, same format as gt.txt
- **Format:** `[frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z]`

---

## Parts

### Part 1 — Ground Truth Visualization
Loads `gt.txt` and draws color-coded bounding boxes on each frame. Each pedestrian ID gets a unique color from the HSV palette, consistent across all frames.

### Part 2 — Pedestrian Detector
Background subtraction detector, using methods from lab classes
1. Estimates background via median across 100 sampled frames — moving pedestrians don't persist in the median so the result is a clean empty-scene image
2. Per-frame absolute difference against background, thresholded to get a binary foreground mask
3. Morphological cleanup with `imclose` + `imerode` to fill gaps and remove noise
4. `bwlabel` + `regionprops` to find regions, filtered by minimum area, with bounding boxes drawn on each detection

### Part 3 — Trajectory Plotting
Uses the same background subtraction detector as Part 2. Detections are matched across frames using nearest-neighbour assignment on centroid distance. Each track stores a history of centroid positions which is drawn as a coloured trail dynamically as the video plays.

### Part 4 — Consistent Labels Through Time
Extends Part 3 with more robust tracking to keep pedestrian IDs stable across the sequence. Key improvements over Part 3:

- **Velocity estimation** — each track maintains a smoothed velocity estimate, used to predict where the pedestrian will be in the next frame before matching
- **Appearance model** — average RGB colour of each bounding box is tracked and used as an additional matching cost alongside distance
- **IoU term** — bounding box overlap between predicted and detected boxes is included in the cost matrix
- **Blob splitting** — wide blobs where two pedestrians are merged get split horizontally before matching
- **Looser parameters** — higher `maxMatchDist` (55 vs 45) and longer `maxInvisible` (18 vs 10) to survive occlusions and missed detections

**Discussion of approaches at different accuracy levels:**
- *Low accuracy:* nearest-neighbour on centroid distance only — fast but label switching occurs when pedestrians cross or come close together
- *Medium accuracy (Part 3):* nearest-neighbour with appearance model and max distance threshold — reduces switching but still struggles with occlusion
- *Higher accuracy (Part 4):* motion prediction + appearance + IoU in a combined cost matrix — more stable IDs through the sequence, handles brief occlusions via the invisible count tolerance

### Part 5 — Occupancy Heatmap
Two heatmaps showing where pedestrians spent the most time across the sequence, using centroid positions from `det.txt`:

- **Static heatmap** — accumulates all centroid hits across all 795 frames into a 2D grid, then convolves with a Gaussian kernel (`fspecial('gaussian')` + `filter2`) to spread each hit into a smooth blob. Displayed as a semi-transparent hot colormap overlay on the first frame.
- **Dynamic heatmap** — same accumulation but updated and displayed live each frame, so you watch the heatmap grow in real time as pedestrians walk through the scene.

### Part 6 — Statistical Analysis via EM
Fits a Gaussian Mixture Model (GMM) to all pedestrian centroid positions using the Expectation-Maximization algorithm (`fitgmdist`). Each Gaussian component represents a distinct zone of activity in the scene.

Detections within 100px of the image border are removed before fitting to exclude false positives from the ACF detector firing on scene edges.

Two outputs:
- **Ellipse plot** — each component drawn as a 2-sigma ellipse (via eigendecomposition of the covariance matrix) overlaid on the scene, with its centre marked
- **Density map** — the full GMM probability density evaluated at every pixel, showing the statistical model's view of pedestrian occupancy

Printed stats per component: mixing proportion (how much of total activity), mean position (x, y in pixels), and standard deviation in x and y

### Part 7 — Evaluation Performance of Algorithm

### Part 8 — Deep Neural Network Comparison

---

## Requirements

MATLAB with Image Processing Toolbox and Statistics and Machine Learning Toolbox (required for `fitgmdist` in Part 6).