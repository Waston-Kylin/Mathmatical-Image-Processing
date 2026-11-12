# Mathematical Image Processing
## Requirements
* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-image](https://scikit-image.org/)
* [Pillow](https://python-pillow.org/)
* [h5py](https://www.h5py.org/)
* [Gradio](https://gradio.app/)

```bash
pip install numpy scipy matplotlib scikit-image pillow h5py gradio
```

## TV-Based Image Deblurring and Denoising
```bash
cd "1 TV-based Image Deblurring and Denoising"/code
```
To perform TV-based image deblurring and denoising on test images, you can run the following command:
```bash
python main.py \
    --image_path $IMAGE_PATH \
    --lambda_weight $LAMBDA_WEIGHT \
    --output_image $OUTPUT_IMAGE \
    --kernel_size $KERNEL_SIZE \
    --gaussian_sigma$GAUSSIAN_SIGMA \
    --noise_level $NOISE_LEVEL 
```
- ```$NAME``` is the image name (choose from "barbara", "peppers256").
- ```$LAMBDA_WEIGHT``` is the regularization parameter $\lambda$ balancing TV regularization and data fidelity (default: 0.001).
- ```$KERNEL_SIZE``` is the spatial dimension $s$ of the blur kernel (default: 15).
- ```$GAUSSIAN_SIGMA``` is the standard deviation $\sigma$ of the Gaussian blur kernel (default: 1.0).
- ```$NOISE_LEVEL``` is the standard deviation $\sigma_1$ of additive Gaussian noise (choose from 0.01, 0.02, default: 0.01).
- ```$MU``` is the penalty parameter $\mu$ in the augmented Lagrangian (ADMM / Split Bregman algorithm), which controls the balance between the fidelity term and the constraint enforcement (default: 0.012), you can change this in `mysubroutines.py` Line 33.
- ```$MAX_ITER``` is the maximum number of iterations for the solver to converge (default: 50), you can change this in `mysubroutines.py` Line 33.

## PDE-Based Image Restoration
```bash
cd "2 PDE-Based Image Restoration"/code
```
It is recommended to test the code with the UI provided in `UI.py`, given that there are many hyperparameters and thus it is hard to run the code manually in terminal.
```bash
python UI.py
```
To perform PDE-based image deblurring and denoising on test images, you can run the following command in terminal:

```bash
python main.py \
    --name $IMAGE_NAME \
    --method $PDE_EQUATION \
    --T $TERMINATION_TIME \
    --dt $TIME_STEP \
    --noise_level $NOISE_LEVEL \
    --blur $BLUR_OR_NOT \
    --K $PM_EDGE_THRESHOLD \
    --scheme $PM_SCHEME \
    --L_type $SHOCK_FILTER_TYPE
```
- ```$IMAGE_NAME``` is the image name (choose from "barbara", "peppers256", default: "barbara").
- ```$PDE_EQUATION``` is the PDE-based restoration model to be solved (choose from "heat", "pm", "shock", default: "heat").
- ```$TERMINATION_TIME``` is the stopping/evolution time $T$ of the PDE (default: 0.2).
- ```$TIME_STEP``` is the discrete time step size $\Delta t$ for explicit numerical integration (default: 0.01).
- ```$NOISE_LEVEL``` is the standard deviation $\sigma_1$ of additive Gaussian noise (set to 0 for noise-free, default: 0.02).
- ```$BLUR_OR_NOT``` is a boolean flag indicating whether to apply Gaussian blur degradation (kernel size $15 \times 15$, $\sigma = 1.0$) to synthesize blurred observations for deblurring tasks (default: False).
- ```$PM_EDGE_THRESHOLD``` is the edge threshold parameter $K$ in the Perona-Malik diffusivity function $c(\vert{}\nabla u\vert{}^2) = \frac{1}{1 + \vert{}\nabla u\vert{}^2 / K^2}${}, controlling the boundary between smoothing and sharpening (default: 0.04).
- ```$PM_SCHEME``` is the finite difference discretization scheme for the Perona-Malik divergence operator (choose from "symmetric" for half-point flux-conserving central differences, "asymmetric" for standard one-sided differences, default: "symmetric").
- ```$SHOCK_FILTER_TYPE``` is the second-order derivative operator $L(u)$ used in the Shock Filter $\partial u / \partial t = -\vert{}\nabla u\vert{} \operatorname{sign}(L(u))${}{}  to determine the local edge convexity/inflection (choose from "laplacian" for isotropic $\Delta u$, "u_nn" for directional second derivative $u_{NN}$ along the gradient direction, default: "u_nn").

## PDE-Based Image Deblurring and Denoising
```bash
cd "3 PDE-Based Image Deblurring and Denoising"/code
```
It is recommended to test the code with the UI provided in `UI.py`, given that there are many hyperparameters and thus it is hard to run the code manually in terminal.
```bash
python UI.py
```
To perform PDE-based image deblurring and denoising on test images, you can run the following command in terminal:
```bash
python main.py \
    --name $IMAGE_NAME \
    --method $SEGMENTATION_MODEL \
    --init_type $INITIAL_CONTOUR \
    --noise_level $NOISE_LEVEL \
    --blur \
    --blur_sigma $GAUSSIAN_BLUR_SIGMA \
    --max_iter $MAX_ITERATIONS \
    --dt $TIME_STEP \
    --reinit_int $REINIT_INTERVAL \
    --alpha $BALLOON_FORCE \
    --edge_k $EDGE_THRESHOLD \
    --mu $LENGTH_PENALTY \
    --lam_admm $ADMM_PENALTY
```

- ```$IMAGE_NAME``` is the image name located in image/ without extension (choose from "shapes", "weak_edge", "kanizsa", "coins", default: "shapes").
- ```$SEGMENTATION_MODEL``` is the variational/level-set segmentation model to be evaluated (choose from "gac" for Geodesic Active Contours, "cv" for Chan-Vese, "convex_cv" for Convexified Binary Mumford-Shah, default: "cv").
- ```$INITIAL_CONTOUR``` is the geometric configuration of the initial level set $\phi_0$ (choose from "box" for outer bounding rectangle, "circle" for center circle, "corner_circle" for off-center local circle, "multi_circles" for small grid bubbles, default: "box").
- ```$NOISE_LEVEL``` is the standard deviation $\sigma_n$ of additive zero-mean Gaussian white noise (set to 0.0 for noise-free, default: 0.00).
- ```--blur``` is a boolean flag indicating whether to apply Gaussian blur degradation ($15 \times 15$ kernel) to synthesize blurred observations for boundary leakage evaluation (default: False).
- ```$GAUSSIAN_BLUR_SIGMA``` is the standard deviation $\sigma_b$ of the Gaussian blur kernel when --blur is enabled (default: 2.0).
- ```$MAX_ITERATIONS``` is the total number of iterations for the solver (recommended: 1000 for GAC, 3000 for CV, 500 for Convex CV, default: 1000).
- ```$TIME_STEP``` is the discrete time step size $\Delta t$ for explicit PDE evolution schemes in GAC and CV (default: 0.4).
- ```$REINIT_INTERVAL``` is the iteration interval for calling Reinitial2D to restore $\phi$ to a signed distance function ($\vert{}\nabla \phi\vert{} = 1${}{} ) via the Godunov upwind scheme (default: 5).
- ```$BALLOON_FORCE``` is the inflation/deflation speed parameter $\alpha$ in the GAC model $\alpha g(\vert{}\nabla I\vert{})\vert{}\nabla u\vert{}${}{}{}{}, where negative values shrink the contour inward and positive values expand it outward (choose from -0.3, -0.2, -0.1, default: -0.2).
- ```$EDGE_THRESHOLD``` is the sensitivity parameter $k$ in the GAC boundary stopping function $g(\vert{}\nabla I\vert{}) = \frac{1}{1 + \vert{}\nabla (G_\sigma * I)\vert{}^2 / k^2}${}{} (default: 0.05).
- ```$LENGTH_PENALTY``` is the regularization parameter $\mu$ penalizing contour length/TV norm to control boundary smoothness (default: 1.0).
- ```$ADMM_PENALTY``` is the augmented Lagrangian penalty parameter $\lambda_{\text{admm}}$ for the auxiliary variable $\vec{d} = \nabla u$ in the Split Bregman / ADMM solver of Convex CV (default: 1.0).

## Wavelet Frame Based Image Restoration
```bash
cd "4 Wavelet Frame Based Image Restoration"/code
```
### Execution & Evaluation
It is recommended to test the code with the interactive Web UI provided in `UI.py`, given that there are multiple algorithms and hyperparameters to configure:
```bash
python UI.py
```
To perform image restoration (deblurring and denoising) via the command-line interface, you can run `main.py` in the terminal:

```bash
python main.py \
    --image_path $IMAGE_PATH \
    --method $RESTORATION_MODEL \
    --noise_level $NOISE_LEVEL \
    --lambda_weight $LAMBDA_REGULARIZATION \
    --max_iter $MAX_ITERATIONS \
    --kernel_size $KERNEL_SIZE \
    --gaussian_sigma $GAUSSIAN_SIGMA \
    --frame $FRAMELET_ORDER \
    --Level $DECOMPOSITION_LEVEL \
    --thresh_type $THRESHOLD_TYPE \
    --mu $ADMM_PENALTY \
    --delta $DUAL_STEP_SIZE \
    --kappa $BALANCED_WEIGHT \
    --use_apg \
    --output_image $OUTPUT_IMAGE_PATH
```

- ```$IMAGE_PATH``` is the relative path to the input target image (e.g., "image/aircraft.jpg", "image/peppers256.png", "image/barbara.png", default: "aircraft.jpg").
- ```$RESTORATION_MODEL``` is the image restoration framework to be evaluated (choose from "analysis" for Analysis-based Split Bregman / ADMM, "balanced" for Balanced Approach via PFBS/APG, "tv" for Total Variation Split Bregman baseline, default: "analysis").
- ```$NOISE_LEVEL``` is the standard deviation $\sigma_1$ of additive Gaussian white noise (choose from 0.01, 0.02, default: 0.01).
- ```$LAMBDA_REGULARIZATION``` is the regularization parameter $\lambda$ balancing the $\ell_1$-norm sparsity of framelet coefficients against data fidelity (default: 0.0008).
- ```$MAX_ITERATIONS``` is the maximum number of iterations for the solver (default: 50).
- ```$KERNEL_SIZE``` is the spatial window size $s$ of the 2D Gaussian blur kernel $[s \times s]$ (default: 15).
- ```$GAUSSIAN_SIGMA``` is the standard deviation $\sigma$ of the Gaussian blur kernel (default: 1.0).
- ```$FRAMELET_ORDER``` is the B-spline tight framelet filter type (choose from 0 for piecewise constant Haar framelet, 1 for piecewise linear B-spline framelet, 3 for piecewise cubic B-spline framelet, default: 1).
- ```$DECOMPOSITION_LEVEL``` is the number of multi-scale framelet decomposition levels $L$ (default: 2).
- ```$THRESHOLD_TYPE``` is the soft-thresholding shrinkage operator applied to high-frequency framelet coefficients (choose from "s" for standard anisotropic coordinate-wise soft-thresholding, "vs" for isotropic vector soft-thresholding across directional subbands, default: "s").
- ```$ADMM_PENALTY``` is the augmented Lagrangian penalty parameter $\mu$ for the split constraint $d = Wu$ in the Analysis-ADMM solver and the TV solver (default: 0.015).
- ```$DUAL_STEP_SIZE``` is the step size $\delta$ for dual variable updating $b_{k+1} = b_k + \delta(Wu_{k+1} - d_{k+1})$ in Analysis-ADMM ($0 < \delta \le 1.0$, default: 1.0).
- ```$BALANCED_WEIGHT``` is the penalization parameter $\kappa$ enforcing canonical frame coefficient consistency $\frac{\kappa}{2} \|(I - WW^T)\alpha\|_2^2$ in the Balanced model (default: 1.0).
- ```--use_apg``` is a boolean flag indicating whether to enable Nesterov's Accelerated Proximal Gradient (APG / FISTA momentum extrapolation) for the Balanced PFBS solver (default: False in CLI, recommended: True).
- ```$OUTPUT_IMAGE_PATH``` is the target file path to save the 3-panel visual comparison plot (`u_true`, `f`, `u`) (default: "../docs/image/aircraft_analysis_n0.01_f1_ts.jpg").

### Benchmark
To automatically run the entire benchmark pipeline comparing TV, Analysis (ADMM), and Balanced (PFBS) across all test images and noise levels, execute:
```bash
python main.py --benchmark
```

### Given `usege.py`
```bash
python usege.py
```

## Wavelet Frame Based Image Segmentation and 3D Reconstruction
```bash
cd "5 Wavelet Frame Based Image Segmentation and 3D Reconstruction"/code
```

### Execution & Evaluation
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

### Quick Start Examples

### Run 2D Image Segmentation Comparison Only
Evaluates and benchmarks **GAC**, **Chan-Vese**, and **Wavelet Frame-based Segmentation (Algorithm 2)** on `shapes`, `weak_edge`, `kanizsa`, and `coins`:
```bash
python main.py --task seg2d --noise_level 0.02
```

### Test 3D Surface Reconstruction on a Single Model
Reconstructs the `cndragon` model and verifies marching cubes isosurface rendering without processing the remaining models:
```bash
python main.py --task surf3d --models cndragon
```

### Full Benchmark & Figure 4 Reproduction
Runs the complete test suite across all 2D benchmark images and reproduces the full $2 \times 3$ panel of Figure 4 across all three 3D models (`cndragon`, `lucy`, `statuette`):
```bash
python main.py --task all
```

*Note: 3D reconstruction results are cached automatically as `*_u_rec_v.npy` in `$OUTPUT_SURF_DIR`. Re-running visual rendering or figure generation with existing cache files completes in seconds without recomputing the ADMM iterations.*
