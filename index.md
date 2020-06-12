---
layout: landing
title: Improving Deconvolution of fMRI Signal with Sparse Paradigm Free Mapping Using Stability Selection
description: Eneko Uruñuela, Stephen Jones, Anna Crawford, Wanyong Shin, Sehong Oh, Mark Lowe, Cesar Caballero-Gaudes
publication: OHBM 2020
year: 2020
button:
  link: https://github.com/eurunuela/OHBM_2020/raw/master/OHBM_2020_poster.pdf
  name: Poster
owner:
  name: Eneko Uruñuela
social:
  twitter: https://twitter.com/eurunuela
  github: https://github.com/eurunuela
  email: e.urunuela@bcbl.eu
---


# [Scope of this work](#scope)

The main goal of this work is to estimate neuronal-related activity without any prior information about the timings of the BOLD events. To do this, several algorithms have been proposed in the literature and they all solve the problem by means of regularized least-squares estimators with L1- and L2-norms[^1][^2][^3][^4][^5]. However, there are no algorithms that select an optimal regularization parameter; which is key for accurate estimation. Thus, this work introduces two improvements over the previous Sparse Paradigm Free Mapping (SPFM)[^6] algorithm:

- The addition of a subsampling approach called Stability Selection that avoids the selection of the regularization parameter; 
- A modified formulation that allows us to estimate innovation signals.

# [Formulation](#formulation)

In the original SPFM formulation, for a given voxel, the fMRI signal $$y(t)$$ is explained as the convolution of the haemodynamic responde function (HRF) $$h(t)$$ and the activity-inducing signal $$s(t)$$. Gaussian noise $$n(t)$$ is also considered. As we know the neuronal-related signal is sparse, we estimate it by means of regularized least-squares with an L1 penalty.

$$
\mathbf{y} = \mathbf{H}\mathbf{s} + \mathbf{n} \; \; \; \; \longrightarrow \; \; \; \; \hat{\mathbf{s}}=\underset{\mathbf{s}}{\operatorname{argmin}} \frac{1}{2}\|\mathbf{y}-\mathbf{H s}\|_{2}^{2}+\lambda|\mathbf{s}|_{1}
$$

![](./images/demo_r2_colors.png){:width="100%"}
**Figure 1: Simulated signal with activity-inducing signal $$s$$ of the 5 different events.**

Yet, this work introduces a modification to this formulation: the estimation of the innovation signal $$u(t)$$, which is the derivative of the activity-inducing signal; i.e. $$u = Ds$$. In order to estimate the innovation signal, we add an integration operator $$L$$ into our design matrix H, and we solve the same regularized least-squares problem.

$$
\mathbf{y} = \mathbf{H}\mathbf{L}\mathbf{s} + \mathbf{n} \; \; \; \; \longrightarrow \; \; \; \; \widehat{\mathbf{u}}=\underset{\mathbf{u}}{\operatorname{argmin}} \frac{1}{2}\|\mathbf{y}-\mathbf{H} \mathbf{L u}\|_{2}^{2}+\lambda|\mathbf{u}|_{1}
$$

where

$$
\mathbf{D}=\left[\begin{array}{ccccc}
1 & 0 & \cdots & & \\
1 & -1 & 0 & \cdots & \\
0 & \ddots & \ddots & \ddots & \ldots \\
\vdots & \ddots & 0 & 1 & -1
\end{array}\right], \; \; \; \; \mathbf{L}=\left[\begin{array}{ccccc}
1 & 0 & \cdots & & \\
1 & 1 & 0 & \cdots & \\
1 & 1 & 1 & 0 & \cdots \\
\vdots & \ddots & \ddots & \ddots & \ddots
\end{array}\right] .
$$

![](./images/demo_innovation_colors.png){:width="100%"}
**Figure 2: Simulated signal with innovation signal $$u$$ of the 5 different events.**

# [Regularization paths](#regularization-paths)

# [Stability Selection](#stability-selection)

<video autoplay loop style="width=100%">
    <source src="./images/auc.mov" type="video/mp4">
</video>

# [Benchmarking](#benchmarking)

---

[^1]: D. R. Gitelman, W. D. Penny, J. Ashburner, and K. J. Friston, “Modeling regional and psychophysiologic interactions in fMRI: The importance of hemodynamic deconvolution”, Neuroimage, vol. 19, pp. 200–207, 2003.

[^2]: I. Khalidov, J. Fadili, F. Lazeyras, D. Van De Ville, and M. Unser, “Activelets: Wavelets for Sparse Representation of Hemodynamic Responses”, Signal Processing, vol. 91, pp. 2810–2821, 2011.

[^3]: F.I. Karahanoǧlu, C. Caballero-Gaudes, F. Lazeyras, and D. Van De Ville, “Total Activation: FMRI Deconvolution through Spatio- Temporal Regularization”, Neuroimage, vol. 73, pp. 121-134, 2013.

[^4]: L. Hernandez-Garcia and M.O. Ulfarsson, “Neuronal event detection in fMRI time series using iterative deconvolution techniques”, Magnetic Resonance Imaging, vol. 2, pp. 353-364, 2011.

[^5]: C. C. Gaudes, N. Petridou, I.L. Dryden, L. Bai, S.T. Francis, and P.A. Gowland, “Detection and Characterization of Single-Trial FMRI Bold Responses: Paradigm Free Mapping”, Human Brain Mapping, vol. 32, pp. 1400-1418, 2011.

[^6]: C. Caballero-Gaudes, N. Petridou, S.T. Francis, I.L. Dryden, and P.A. Gowland, “Paradigm Free Mapping with Sparse Regression Automatically Detects Single-Trial Functional Magnetic Resonance Imaging Blood Oxygenation Level Dependent Responses”, Human Brain Mapping, vol. 34, pp. 501-518, 2013.