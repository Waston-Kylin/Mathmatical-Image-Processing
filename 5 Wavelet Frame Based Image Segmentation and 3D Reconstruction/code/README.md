# HW5

## Dependencies
* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-image](https://scikit-image.org/)
* [h5py](https://www.h5py.org/)

```bash
pip install numpy scipy matplotlib scikit-image h5py
```

## Execution & Evaluation
This project implements wavelet frame-based 2D image segmentation and 3D implicit surface reconstruction from unorganized point clouds, reproducing the results in *Dong & Shen (2012)*.

All tasks can be configured and executed through the command-line interface in `main.py`:

```bash
python main.py \
    --task $TASK \
    --noise_level $NOISE_LEVEL \
    --models $MODELS \
    --alpha_3d $ALPHA_3D \
    --mu_3d $MU_3D \
    --max_iter_3d $MAX_ITER_3D \
    --tol_3d $TOL_3D \
    --output_seg_dir $OUTPUT_SEG_DIR \
    --output_surf_dir $OUTPUT_SURF_DIR
```

- ```$TASK``` specifies the pipeline to run (choose from `"seg2d"` for 2D image segmentation comparison, `"surf3d"` for 3D surface reconstruction and Figure 4 reproduction, or `"all"` to execute both pipelines sequentially, default: `"all"`).
- ```$NOISE_LEVEL``` is the standard deviation $\sigma_n$ of additive zero-mean Gaussian white noise applied during 2D segmentation evaluation (runs both $\sigma_n = 0.0$ clean and $\sigma_n > 0$ noisy comparisons, default: `0.02`).
- ```$MODELS``` specifies the target 3D datasets located in `Data/` to reconstruct (choose one or more from `"cndragon"`, `"lucy"`, `"statuette"`, default: `"cndragon" "lucy" "statuette"`).
- ```$ALPHA_3D``` is the regularization parameter $\alpha$ scaling the spatially adaptive distance-based threshold field $\lambda(x) = \alpha \cdot \text{dist}(x, X)$ in Algorithm 3 (default: `0.8`).
- ```$MU_3D``` is the penalty parameter $\mu$ balancing data fidelity $\langle 2f - 1, u \rangle$ against framelet sparsity in the ADMM splitting scheme (default: `1.0`).
- ```$MAX_ITER_3D``` is the maximum iteration count for the 3D surface reconstruction solver (default: `1000`).
- ```$TOL_3D``` is the relative change stopping tolerance $\|u^{k+1} - u^k\|_2 / \|u^k\|_2$ for early convergence in 3D surface reconstruction (default: `1e-4`).
- ```$OUTPUT_SEG_DIR``` is the directory where 2D segmentation comparison figures (`seg2d_*.png`) are saved (default: `"../doc/segmentation_results"`).
- ```$OUTPUT_SURF_DIR``` is the directory where reconstructed 3D surface models, `.npy` solution caches, single-model comparisons (`surf3d_*.png`), and the complete Figure 4 reproduction panel (`Figure4_Reproduction.png`) are saved (default: `"../doc/reconstruction_results"`).

---

## Quick Start Examples

### 1. Run 2D Image Segmentation Comparison Only
Evaluates and benchmarks **GAC**, **Chan-Vese**, and **Wavelet Frame-based Segmentation (Algorithm 2)** on `shapes`, `weak_edge`, `kanizsa`, and `coins`:
```bash
python main.py --task seg2d --noise_level 0.02
```

### 2. Test 3D Surface Reconstruction on a Single Model
Reconstructs the `cndragon` model and verifies marching cubes isosurface rendering without processing the remaining models:
```bash
python main.py --task surf3d --models cndragon
```

### 3. Full Benchmark & Figure 4 Reproduction
Runs the complete test suite across all 2D benchmark images and reproduces the full $2 \times 3$ panel of Figure 4 across all three 3D models (`cndragon`, `lucy`, `statuette`):
```bash
python main.py --task all
```

*Note: 3D reconstruction results are cached automatically as `*_u_rec_v.npy` in `$OUTPUT_SURF_DIR`. Re-running visual rendering or figure generation with existing cache files completes in seconds without recomputing the ADMM iterations.*