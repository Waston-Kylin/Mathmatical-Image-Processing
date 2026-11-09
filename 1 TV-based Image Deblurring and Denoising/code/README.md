# HW1
## Dependencies
* [Numpy](https://numpy.org/)
* [Scipy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scikit-image](https://scikit-image.org/)

## Execution & Evaluation
To perform TV-based image deblurring and denoising on test images, you can run the following command:

```
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