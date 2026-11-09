# HW3
## Dependencies
* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-image](https://scikit-image.org/)
* [Gradio](https://gradio.app/)
```bash
pip install numpy scipy matplotlib scikit-image gradio
```

## Execution & Evaluation
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