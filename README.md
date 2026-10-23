# AMPCA

Attention Matrix Principal Component Analysis (AMPCA) jointly learns row and column projections with spatial attention. This implementation uses the OSM algorithm for image reconstruction.

## Requirements

- Python
- CUDA-enabled PyTorch and a compatible NVIDIA GPU
- NumPy, Matplotlib, and Pillow

The current script requires CUDA and uses double-precision computations.

## Usage

Edit the configuration in `main()` of `OSM_GPU_surrogate.py`, then run:

```bash
python OSM_GPU_surrogate.py
```

Set `IMAGE_DIR` and `OUTPUT_DIR` before running. Images are loaded recursively from the input directory. With the default settings, images are converted to grayscale and must have identical dimensions.

## Main Parameters

| Parameter |  Description |
| `IMAGE_DIR` |  Directory containing input images. |
| `OUTPUT_DIR` |  Output directory prefix; a timestamped run directory is created automatically. |
| `K1`, `K2` |  Row and column projection dimensions. |
| `BETA` |  Attention penalty coefficient; must be nonnegative. |
| `ROW_ROI` | Row attention interval. |
| `COL_ROI` |  Column attention interval. |
| `C_MODE` | Surrogate coefficient: `"ZERO"` for 0, `"LE"` for LE, or `"LE_SQUARED"` for LE². |
| `LAMBDA`, `MU_PROX` | Proximal parameters. `None` selects `0.9 / (2*LE + 1)` automatically. |
| `TOL` | Stopping tolerance for both subspace distances. |
| `MAX_ITER_QR` | Maximum number of iterations. |
| `SEED` | Random initialization seed. |
| `TO_GRAY` | Convert images to grayscale. |
| `TARGET_SIZE` | Optional image resizing; `None` preserves the original size. |
| `QR_SNAP_ITERS` | Iterations selected for reconstruction snapshots. |
| `SAVE_TOP_N` | Maximum number of images saved per snapshot; final reconstructions include all input images. |

ROI intervals use **1-based, inclusive indexing**. Attention-matrix diagonal entries are zero inside these intervals and one elsewhere.

`LE` is the sum of squared spectral norms of the centered images. Iteration stops when both `dist_U` and `dist_V` are below `TOL`, or the iteration limit is reached.

## Outputs

- `QR_reconstructed/`: final reconstructions and iteration snapshots.
- `results/`: subspace distances, iteration times, parameter settings, stopping status, and plots.
- `matrices/`: `U_qr.npy`, `V_qr.npy`, and `Psi_train.npy`.
