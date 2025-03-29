# Project 1: Variational SAE for Superposition Study

*This document is intended for internal use within the SPAR Program. It summarizes the first-stage results of an example project, serving as prior knowledge for continued research and development.
*

## Background:
+ Builds on Toy Models of Superposition and Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.
+ Addressing concerns that correlated features remain entangled in current SAE setups, limiting interpretability.
## Goal:
+ Improve disentanglement of correlated features in SAEs.
+ Introduce variational inference techniques to structure feature extraction better.
## Proposal:
+ **Toy Model Study**
  + **Baseline Study:** Evaluate SAEs in handling correlated features.
    + **Toy Model 1:** Setwise correlation/anti-correlation.
    + **Toy Model 2:** General correlation matrix via Cholesky decomposition.
  + **Variational SAE (VSAE) Approach:**
    + **Normal VAE:** Isotropic Gaussian prior (benchmarking case).
    + **Gaussian Mixture VAE:** Targeting setwise correlation disentanglement.
    + **Multivariate Gaussian VAE:** General correlation-aware feature separation.
  + **Evaluation:** Compare reconstruction loss, sparsity metrics, and clustering entropy across these methods.
  + **Findings from My Prior Experiments:** 
    + **VSAE consistently outperforms standard SAEs** in feature disentanglement.
    + **Structured priors help extract interpretable latent representations**, reducing superposition entanglement.
    + **Correlation-based priors** improve feature sparsity without degrading reconstruction quality.
  + **Reference Code:** https://colab.research.google.com/drive/1CPfEXtGdpuTD9-hLNKeQ7EQBYA8xcIli#scrollTo=NtJqqkV8nV1H
+ **Real Model Study**
  + It’s all up to you! My advice would be to focus on finding an SAE approach that gets a good performance on a L0 vs change in cross-entropy loss plot on a small language model like gelu-1l or gpt-2 small. This should let you try a bunch of ideas and see which, if any, are any good.
  + You may also want to read up on various improvements to SAE training, such as Gated, Anthropic's, TopK, JumpReLU, to ensure you're beating and building on modern approaches.
