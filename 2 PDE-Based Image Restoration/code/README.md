# HW2
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