# High-Performance GMM References

This list is scoped to implementations and papers that are useful for `GMMXX`, which is an EM-based GMM clustering package. It excludes unrelated learned-compression projects with similar naming as implementation baselines.

## Best Baseline Candidates

### TorchGMM

- Link: https://pypi.org/project/torchgmm/
- Type: PyTorch/PyTorch Lightning GMM package.
- Fit for this project: good external GPU baseline.
- Notes: supports single/multiple CPU/GPU training, mini-batch training, and a scikit-learn-style estimator API. Current PyPI metadata supports Python `>=3.10,<3.14`, so it fits Python 3.12.

### tgmm

- Link: https://adriansousapoza.github.io/tgmm/
- Type: PyTorch GMM package.
- Fit for this project: useful correctness/feature baseline, especially for covariance modes.
- Notes: supports EM/MAP, full/diagonal/spherical/tied covariance variants, GPU via PyTorch, several initializers, and clustering metrics.

### scikit-learn GaussianMixture

- Link: https://sklearn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- Type: standard CPU GMM implementation.
- Fit for this project: required reference for correctness and quality.
- Notes: supports `full`, `tied`, `diag`, and `spherical` covariance types; use `covariance_type="spherical"` for the closest comparison to `GMMXX`.

## GPU/CUDA Design References

### Andrew Harp CUDA EM GMM

- Link: https://www.mathworks.com/matlabcentral/fileexchange/24020-expectation-maximization-of-gaussian-mixture-models-via-cuda
- Type: CUDA EM implementation with MATLAB integration.
- Fit for this project: useful old-school CUDA kernel reference for E-step/M-step decomposition and reductions.
- Notes: reports large speedups on a workload with 1,000,000 points, 16 dimensions, and 16 clusters. The interesting files are described as `gpugaumixmod.h` and `gpugaumixmod_kernel.h`.

### Fast Estimation of Gaussian Mixture Model Parameters on GPU Using CUDA

- Link: https://dspace.zcu.cz/items/04f982a8-7221-444a-9a54-c4b3658c5aa6
- Type: 2011 IEEE paper.
- Fit for this project: algorithmic reference for CUDA/SSE acceleration of EM GMM parameter estimation.
- Notes: useful for reduction strategy and CPU/GPU comparison, but not a drop-in Python baseline.

### dpgmm

- Link: https://pypi.org/project/dpgmm/
- Type: PyTorch + Triton implementation for Dirichlet Process GMM MCMC.
- Fit for this project: useful Triton reference, not a direct EM baseline.
- Notes: targets high-dimensional DPGMM sampling, so the inference objective is different from fixed-K EM GMM clustering.

## CPU/C++ and Streaming References

### VLFeat GMM

- Link: https://www.vlfeat.org/api/gmm.html
- Type: C implementation of diagonal-covariance GMM.
- Fit for this project: strong CPU numerical-stability and diagonal-covariance reference.
- Notes: supports `float` and `double`, is parallelized, and is tuned for visual-feature datasets. It restricts covariance matrices to diagonal.

### mlpack GMM

- Link: https://www.mlpack.org/doc/user/bindings/cli.html
- Type: fast C++ ML library with command-line and language bindings.
- Fit for this project: CPU baseline/reference for EM training workflow.
- Notes: trains parametric GMMs with EM, supports multiple random trials, convergence tolerance, max iterations, and saved/reused models.

### pomegranate GeneralMixtureModel

- Link: https://pomegranate.readthedocs.io/en/latest/tutorials/C_Feature_Tutorial_3_Out_Of_Core_Learning.html
- Type: PyTorch-backed probabilistic modeling library.
- Fit for this project: useful design reference for exact out-of-core EM updates.
- Notes: its `summarize()` and `from_summaries()` pattern mirrors the streaming sufficient-statistics direction we want for large-N GMM fitting.

### Spark MLlib GaussianMixture

- Link: https://spark.apache.org/docs/latest/api/scala/org/apache/spark/mllib/clustering/GaussianMixture.html
- Type: distributed EM GMM.
- Fit for this project: distributed-data reference, not a GPU baseline.
- Notes: useful for understanding distributed sufficient-statistics aggregation. Spark documents the high-dimensional limitation from storing covariance matrices.

### OpenCV EM

- Link: https://docs.opencv.org/3.4/javadoc/org/opencv/ml/EM.html
- Type: C++/OpenCV EM model.
- Fit for this project: practical CPU implementation reference.
- Notes: supports training from E-step, M-step, or full EM entry points, and exposes labels, probabilities, likelihoods, means, covariances, and weights.

## Lower-Priority References

### ldeecke/gmm-torch

- Link: https://github.com/ldeecke/gmm-torch
- Type: older PyTorch EM implementation.
- Fit for this project: simple PyTorch reference.
- Notes: useful for API and sanity checks, but less relevant as a performance target than TorchGMM or tgmm.

### GMMPytorch

- Link: https://github.com/kylesayrs/GMMPytorch
- Type: PyTorch GMM optimized by gradient descent.
- Fit for this project: not an EM baseline.
- Notes: supports multiple covariance families and singularity guardrails, but the optimizer path differs from EM.

## Practical Ranking For `GMMXX`

1. Use `sklearn-spherical` as the mandatory CPU correctness baseline.
2. Use `flash-torch` as the internal ablation baseline.
3. Use `TorchGMM` as the most relevant external GPU baseline.
4. Use `tgmm` for broader covariance and PyTorch comparison.
5. Use VLFeat, mlpack, pomegranate, and the CUDA papers as implementation references rather than first-pass benchmark dependencies.
6. Treat dpgmm as a Triton design reference only, because DPGMM MCMC is not fixed-K EM.
