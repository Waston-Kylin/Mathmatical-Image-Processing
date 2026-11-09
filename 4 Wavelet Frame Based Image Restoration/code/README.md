# HW4

## Dependencies
* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Pillow](https://python-pillow.org/)
* [Scikit-image](https://scikit-image.org/)
* [Gradio](https://gradio.app/)

```bash
pip install numpy scipy matplotlib pillow scikit-image gradio
```

## Execution & Evaluation
```bash
cd code
```
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

## Benchmark
To automatically run the entire benchmark pipeline comparing TV, Analysis (ADMM), and Balanced (PFBS) across all test images and noise levels, execute:
```bash
python main.py --benchmark
```

## Given `usege.py`
```bash
python usege.py
```